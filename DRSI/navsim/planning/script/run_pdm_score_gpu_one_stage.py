import logging
import os
import traceback
import uuid
from dataclasses import fields
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Union

import hydra
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch.distributed as dist
from hydra.utils import instantiate
from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.common.geometry.convert import relative_to_absolute_poses
from nuplan.planning.script.builders.logging_builder import build_logger
from nuplan.planning.utils.multithreading.worker_utils import worker_map
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import PDMResults, SensorConfig
from navsim.common.dataloader import MetricCacheLoader, SceneLoader
from navsim.evaluate.pdm_score import pdm_score
from navsim.planning.script.builders.worker_pool_builder import build_worker
from navsim.planning.script.run_pdm_score_one_stage import (
    compute_final_scores,
    create_scene_aggregators,
    infer_start_adjacent_mapping,
)
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
from navsim.planning.training.agent_lightning_module import AgentLightningModule
from navsim.planning.training.dataset import Dataset
from navsim.traffic_agents_policies.abstract_traffic_agents_policy import AbstractTrafficAgentsPolicy

logger = logging.getLogger(__name__)

CONFIG_PATH = "config/pdm_scoring"
CONFIG_NAME = "default_run_pdm_score_gpu_one_stage"


def run_pdm_score(args: List[Dict[str, Union[List[str], DictConfig]]]) -> List[pd.DataFrame]:
    """
    CPU worker: Compute PDM score using pre-computed GPU trajectories.
    """
    node_id = int(os.environ.get("NODE_RANK", 0))
    thread_id = str(uuid.uuid4())
    logger.info(f"Starting worker in thread_id={thread_id}, node_id={node_id}")

    tokens = [t for a in args for t in a["tokens"]]
    cfg: DictConfig = args[0]["cfg"]
    model_trajectory = {}
    for a in args:
        model_trajectory.update(a["model_trajectory"])

    simulator: PDMSimulator = instantiate(cfg.simulator)
    scorer: PDMScorer = instantiate(cfg.scorer)
    assert (
        simulator.proposal_sampling == scorer.proposal_sampling
    ), "Simulator and scorer proposal sampling has to be identical"

    if cfg.traffic_agents == "non_reactive":
        traffic_agents_policy: AbstractTrafficAgentsPolicy = instantiate(
            cfg.traffic_agents_policy.non_reactive, simulator.proposal_sampling
        )
    elif cfg.traffic_agents == "reactive":
        traffic_agents_policy: AbstractTrafficAgentsPolicy = instantiate(
            cfg.traffic_agents_policy.reactive, simulator.proposal_sampling
        )

    metric_cache_loader = MetricCacheLoader(Path(cfg.metric_cache_path))
    tokens_to_evaluate = list(
        set(tokens) & set(metric_cache_loader.tokens) & set(model_trajectory.keys())
    )

    pdm_results: List[pd.DataFrame] = []
    for idx, token in enumerate(tokens_to_evaluate):
        logger.info(
            f"Processing scenario {idx + 1} / {len(tokens_to_evaluate)} in thread_id={thread_id}, node_id={node_id}"
        )
        try:
            metric_cache = metric_cache_loader.get_from_token(token)
            trajectory = model_trajectory[token]["trajectory"]

            score_row, ego_simulated_states = pdm_score(
                metric_cache=metric_cache,
                model_trajectory=trajectory,
                future_sampling=simulator.proposal_sampling,
                simulator=simulator,
                scorer=scorer,
                traffic_agents_policy=traffic_agents_policy,
            )
            score_row["valid"] = True
            score_row["log_name"] = metric_cache.log_name
            score_row["frame_type"] = metric_cache.scene_type
            score_row["start_time"] = metric_cache.timepoint.time_s
            end_pose = StateSE2(
                x=trajectory.poses[-1, 0],
                y=trajectory.poses[-1, 1],
                heading=trajectory.poses[-1, 2],
            )
            absolute_endpoint = relative_to_absolute_poses(metric_cache.ego_state.rear_axle, [end_pose])[0]
            score_row["endpoint_x"] = absolute_endpoint.x
            score_row["endpoint_y"] = absolute_endpoint.y
            score_row["start_point_x"] = metric_cache.ego_state.rear_axle.x
            score_row["start_point_y"] = metric_cache.ego_state.rear_axle.y
            score_row["ego_simulated_states"] = [ego_simulated_states]

        except Exception:
            logger.warning(f"----------- Agent failed for token {token}:")
            traceback.print_exc()
            score_row = pd.DataFrame([PDMResults.get_empty_results()])
            score_row["valid"] = False
        score_row["token"] = token
        pdm_results.append(score_row)

    return pdm_results


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    """
    GPU batched inference + CPU one-stage scoring
    """
    build_logger(cfg)

    # =========================================================
    # Phase 1: GPU batched inference
    # =========================================================
    agent: AbstractAgent = instantiate(cfg.agent)
    agent.initialize()

    scene_filter = instantiate(cfg.train_test_split.scene_filter)

    scene_loader_inference = SceneLoader(
        data_path=Path(cfg.navsim_log_path),
        original_sensor_path=Path(cfg.original_sensor_path),
        scene_filter=scene_filter,
        sensor_config=agent.get_sensor_config(),
    )

    dataset = Dataset(
        scene_loader=scene_loader_inference,
        feature_builders=agent.get_feature_builders(),
        target_builders=agent.get_target_builders(),
        cache_path=None,
        force_cache_computation=False,
        append_token_to_batch=True,
        is_training=False,
    )
    dataloader = DataLoader(dataset, **cfg.dataloader.params, shuffle=False)

    trainer = pl.Trainer(**cfg.trainer.params, callbacks=agent.get_training_callbacks())
    predictions = trainer.predict(
        AgentLightningModule(agent=agent, combined=False),
        dataloader,
        return_predictions=True,
    )

    # DDP: Gather results from all GPUs
    if dist.is_initialized():
        dist.barrier()
        all_predictions = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(all_predictions, predictions)
        if dist.get_rank() != 0:
            return None
    else:
        all_predictions = [predictions]

    model_trajectory = {}
    for proc_prediction in all_predictions:
        for d in proc_prediction:
            model_trajectory.update(d)

    logger.info(f"GPU inference complete. Total trajectories: {len(model_trajectory)}")

    # =========================================================
    # Phase 2: CPU scoring (one-stage, original frames only)
    # =========================================================
    scene_loader_meta = SceneLoader(
        data_path=Path(cfg.navsim_log_path),
        original_sensor_path=None,
        scene_filter=scene_filter,
        sensor_config=SensorConfig.build_no_sensors(),
    )
    metric_cache_loader = MetricCacheLoader(Path(cfg.metric_cache_path))

    tokens_to_evaluate = list(set(scene_loader_meta.tokens) & set(metric_cache_loader.tokens))
    num_missing = len(set(scene_loader_meta.tokens) - set(metric_cache_loader.tokens))
    if num_missing > 0:
        logger.warning(f"Missing metric cache for {num_missing} tokens. Skipping these tokens.")
    logger.info(f"Starting pdm scoring of {len(tokens_to_evaluate)} scenarios...")

    worker = build_worker(cfg)
    data_points = [
        {
            "cfg": cfg,
            "log_file": log_file,
            "tokens": tokens_list,
            "model_trajectory": {t: model_trajectory[t] for t in tokens_list if t in model_trajectory},
        }
        for log_file, tokens_list in scene_loader_meta.get_tokens_list_per_log().items()
    ]
    score_rows: List[pd.DataFrame] = worker_map(worker, run_pdm_score, data_points)
    pdm_score_df = pd.concat(score_rows)

    # =========================================================
    # Phase 3: Score aggregation (same as one_stage)
    # =========================================================
    start_adjacent_mapping = infer_start_adjacent_mapping(pdm_score_df)
    pdm_score_df = create_scene_aggregators(
        start_adjacent_mapping, pdm_score_df, instantiate(cfg.simulator.proposal_sampling)
    )
    pdm_score_df = compute_final_scores(pdm_score_df)

    num_successful = pdm_score_df["valid"].sum()
    num_failed = len(pdm_score_df) - num_successful
    failed_tokens = pdm_score_df[~pdm_score_df["valid"]]["token"].to_list() if num_failed > 0 else []

    score_cols = [
        c
        for c in pdm_score_df.columns
        if (
            (any(score.name in c for score in fields(PDMResults)) or c == "two_frame_extended_comfort" or c == "score")
            and c != "pdm_score"
        )
    ]

    average_row = pdm_score_df[score_cols].mean(skipna=True)
    average_row["token"] = "average_all_frames"
    average_row["valid"] = pdm_score_df["valid"].all()

    pdm_score_df = pdm_score_df[["token", "valid"] + score_cols]
    pdm_score_df.loc[len(pdm_score_df)] = average_row

    save_path = Path(cfg.output_dir)
    timestamp = datetime.now().strftime("%Y.%m.%d.%H.%M.%S")
    pdm_score_df.to_csv(save_path / f"{timestamp}.csv")

    logger.info(
        f"""
        Finished running evaluation.
            Number of successful scenarios: {num_successful}.
            Number of failed scenarios: {num_failed}.
            Final average score of valid results: {pdm_score_df['score'].mean()}.
            Results are stored in: {save_path / f"{timestamp}.csv"}.
        """
    )

    if cfg.verbose:
        logger.info(f"\nDetailed results:\n{pdm_score_df.iloc[-3:].T}")
    if num_failed > 0:
        logger.info(f"\nList of failed tokens:\n{failed_tokens}")


if __name__ == "__main__":
    main()
