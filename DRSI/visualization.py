"""
Visualization module for DRSI (Trajectory Pruning Pipeline Visualization)

Visualizes the 4-stage trajectory pruning pipeline:
  Stage 1: Full Vocab       — all ~16384 trajectories from the vocabulary
  Stage 2: After GRC        — Global Route Compliance (filtered by driving command)
  Stage 3: After DRC        — Dynamic Reachability Compliance (filtered by kinematics)
  Stage 4: Final Selection  — Best trajectory selected by aggregated score

Each figure contains:
  - Top row   : Stitched front camera image (L0 + F0 + R0)
  - Middle row : 4 BEV panels showing each pruning stage
  - Bottom row : Per-metric predicted score bar chart for the selected trajectory
  - Badges     : Driving command (LEFT / STRAIGHT / RIGHT / UNKNOWN) on each panel

Per-metric score folder (OUTPUT_DIR / {metric_name} /):
  - One BEV figure per scene: Left = Predicted score, Right = GT score
  - viridis colormap: dark = low score, bright = high score
  - GT trajectory overlaid in green

Usage:
  python visualization.py agent=drsi_vov train_test_split=navtest

Required overrides (add to command line):
  agent=drsi_vov          — use DRSI VoV backbone agent
  train_test_split=navtest — use navtest split / scene filter
"""

import pickle
import torch
import hydra
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from pathlib import Path
from hydra.utils import instantiate
from typing import Dict, Tuple, Optional
import cv2

from navsim.planning.training.dataset import Dataset, CacheOnlyDataset
from navsim.common.dataloader import SceneLoader
from navsim.visualization.bev import add_configured_bev_on_ax
from navsim.visualization.plots import configure_bev_ax

# =========================================================
# User Configuration  (edit these to match your environment)
# =========================================================
CONFIG_PATH = "navsim/planning/script/config/training"
CONFIG_NAME = "competition_training"

# Checkpoint to load  (pick whichever GRC/DRC checkpoint you trained)
CKPT_PATH = "checkpoint_path"

# GT score pkl (token → {metric: array(16384,)})
GT_SCORE_PATH = Path("./dataset/traj_pdm_v2/ori/navtrain_16384.pkl")

# Output directory for visualizations
OUTPUT_DIR = Path("./visualization_outputs")

# Speed bins for scene selection: (min_speed_m_s, max_speed_m_s, label)
# Speed is computed as sqrt(vx² + vy²).  Scenes outside all bins are skipped.
SPEED_BINS = [
    (0.0, 1.0, "v_0to1"),   # ≤ 1 m/s  (stopped / very slow)
    (1.0, 3.0, "v_1to3"),   # 1–3 m/s  (slow)
    (3.0, 6.0, "v_3to6"),   # 3–6 m/s  (moderate)
    (6.0, 8.0, "v_6to8"),   # 6–8 m/s  (fast)
]

# Number of scenes to visualize per speed bin
NUM_SAMPLES_PER_BIN = 5

# =========================================================
# Constants
# =========================================================
DRIVING_COMMAND_NAMES  = {0: "LEFT", 1: "STRAIGHT", 2: "RIGHT", 3: "UNKNOWN"}
DRIVING_COMMAND_COLORS = {0: "#E74C3C", 1: "#27AE60",  2: "#3498DB", 3: "#95A5A6"}

# All 8 score metrics (including IMI)
SCORE_METRICS_ALL = [
    'imi',
    'no_at_fault_collisions',
    'drivable_area_compliance',
    'time_to_collision_within_bound',
    'ego_progress',
    'driving_direction_compliance',
    'lane_keeping',
    'traffic_light_compliance',
]

# Metrics shown in bar chart (all 8)
SCORE_METRICS = SCORE_METRICS_ALL

METRIC_DISPLAY_NAMES = {
    'imi':                           'IMI',
    'no_at_fault_collisions':        'No Fault\nCollision',
    'drivable_area_compliance':      'Drivable\nArea',
    'time_to_collision_within_bound':'TTC\nBound',
    'ego_progress':                  'Ego\nProgress',
    'driving_direction_compliance':  'Drive Dir\nCompliance',
    'lane_keeping':                  'Lane\nKeeping',
    'traffic_light_compliance':      'Traffic\nLight',
}

# Score weights in the final aggregated score formula
METRIC_WEIGHT_LABEL = {
    'imi':                           'w=0.03 (softmax)',
    'traffic_light_compliance':      'w=0.1',
    'no_at_fault_collisions':        'w=0.1',
    'drivable_area_compliance':      'w=0.9',
    'driving_direction_compliance':  'w=0.2',
    'time_to_collision_within_bound':'w=6×7',
    'ego_progress':                  'w=6×7',
    'lane_keeping':                  'w=6×3',
}


# =========================================================
# Helper utilities
# =========================================================

def get_ego_velocity(features: Dict) -> Tuple[float, float]:
    """
    Extract ego velocity (vx, vy) from status_feature indices [4, 5].
    status_feature layout: [cmd(0-3), vx(4), vy(5), ax(6), ay(7)]
    Returns (vx, vy) in m/s in ego-frame (vx=forward, vy=lateral).
    """
    sf = features.get('status_feature', None)
    if sf is None:
        return 0.0, 0.0
    tensor = sf[0] if isinstance(sf, list) else sf
    if isinstance(tensor, torch.Tensor):
        arr = tensor[0].detach().cpu().float()
        if arr.shape[-1] > 5:
            return float(arr[4].item()), float(arr[5].item())
    else:
        arr = np.array(tensor)[0]
        if len(arr) > 5:
            return float(arr[4]), float(arr[5])
    return 0.0, 0.0


def get_speed_bin_label(speed: float) -> Optional[str]:
    """
    Return the SPEED_BINS label for a given scalar speed (m/s).
    Uses half-open intervals: [0,1], (1,3], (3,6], (6,8].
    Returns None if speed is outside all defined bins (e.g. > 8 m/s).
    """
    for vmin, vmax, label in SPEED_BINS:
        if vmin <= speed <= vmax:
            return label
    return None


def add_velocity_overlay(ax, vx: float, vy: float, arrow_scale: float = 2.0):
    """
    Overlay ego velocity as an arrow + text badge on a BEV axis.
    BEV coord convention: vx = forward (up on screen), vy = lateral (right).
    matplotlib plot(lateral, forward) → arrow tip at (vy*scale, vx*scale).
    """
    speed = float(np.sqrt(vx ** 2 + vy ** 2))

    # Text badge at bottom-left of panel
    ax.text(
        0.03, 0.05,
        f"vx:{vx:+.2f}  vy:{vy:+.2f}\nspd:{speed:.2f} m/s",
        transform=ax.transAxes, fontsize=7.5, color='white', fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#1A252F', alpha=0.88),
        ha='left', va='bottom', zorder=200,
    )


def get_driving_command_info(features: Dict) -> Tuple[int, str]:
    """
    Extract the driving command (one-hot, first 4 dims of status_feature)
    and return (index, name).
      0 = LEFT, 1 = STRAIGHT, 2 = RIGHT, 3 = UNKNOWN
    """
    sf = features.get('status_feature', None)
    if sf is None:
        return 3, "UNKNOWN"

    # status_feature may be a list (multi-frame) or a tensor (B, D)
    tensor = sf[0] if isinstance(sf, list) else sf
    if isinstance(tensor, torch.Tensor):
        cmd_idx = int(tensor[0, :4].argmax().item())
    else:
        cmd_idx = int(np.argmax(np.array(tensor)[0, :4]))

    return cmd_idx, DRIVING_COMMAND_NAMES.get(cmd_idx, "UNKNOWN")


def get_stitched_camera(scene, frame_idx: int) -> np.ndarray:
    """Stitch left (L0) + front (F0) + right (R0) camera images into one wide image."""
    frame  = scene.frames[frame_idx]
    cam    = frame.cameras
    l0 = cam.cam_l0.image[28:-28, 416:-416] if cam.cam_l0.image is not None else None
    f0 = cam.cam_f0.image[28:-28]            if cam.cam_f0.image is not None else None
    r0 = cam.cam_r0.image[28:-28, 416:-416]  if cam.cam_r0.image is not None else None

    if l0 is not None and f0 is not None and r0 is not None:
        stitched = np.concatenate([l0, f0, r0], axis=1)
        return cv2.resize(stitched, (1400, 250))
    return np.zeros((250, 1400, 3), dtype=np.uint8)


def to_numpy_indices(t, total: int) -> np.ndarray:
    """Convert a tensor/array of indices to a 1-D numpy array, fallback to all indices."""
    if t is None:
        return np.arange(total)
    arr = t.cpu().numpy() if isinstance(t, torch.Tensor) else np.asarray(t)
    return arr.flatten()


def subsample_drc(drc_idx: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns all DRC indices for visualization (no subsampling).

    Returns
    -------
    sel  : 1-D int array — positions within drc_idx (used to index score arrays)
    vis  : 1-D int array — absolute vocab indices   (used to index vocab_np)
    """
    sel = np.arange(len(drc_idx))
    return sel, drc_idx[sel]


def load_gt_scores(path: Path) -> Dict:
    """
    Load GT score pkl file.
    Returns dict: { token(str) -> { metric(str) -> np.ndarray(16384,) } }
    Returns empty dict if file not found.
    """
    if not path.exists():
        print(f"  [WARN] GT score file not found: {path}")
        return {}
    with open(path, 'rb') as f:
        data = pickle.load(f)
    print(f"  GT scores loaded: {len(data)} tokens  ({path.name})")
    return data


def find_drc_final_pos(drc_idx: np.ndarray, final_vocab_idx: int) -> int:
    """
    Find the position of the final selected trajectory within the DRC-selected index array.
    Returns 0 as fallback if not found.
    """
    pos = np.where(drc_idx == final_vocab_idx)[0]
    return int(pos[0]) if len(pos) > 0 else 0


def get_drc_score_array(predictions: Dict, metric: str) -> Optional[np.ndarray]:
    """
    Returns per-trajectory score array (float32, shape: num_drc) for DRC-selected trajectories.
      - 'imi'   : softmax probability (sums to 1 over DRC trajectories)
      - others  : sigmoid activation  (each value in [0, 1])
    """
    st = predictions.get(metric)
    if st is None:
        return None

    if isinstance(st, torch.Tensor):
        arr = st[0].detach().cpu().float().numpy()
    else:
        arr = np.asarray(st, dtype=np.float32)[0]

    if metric == 'imi':
        # softmax: shift for numerical stability
        arr = arr - arr.max()
        e = np.exp(arr)
        return (e / e.sum()).astype(np.float32)
    else:
        return (1.0 / (1.0 + np.exp(-arr))).astype(np.float32)  # sigmoid


def plot_trajs(ax, vocab_np: np.ndarray, indices: np.ndarray,
               color, alpha: float = 0.5, lw: float = 1.0,
               zorder: int = 5):
    """
    Plot all vocabulary trajectories on a BEV axis.
    BEV coordinate convention: axis-x → longitudinal (forward), axis-y → lateral.
    matplotlib: plot(y, x) so that forward is up on screen.
    """
    if len(indices) == 0:
        return
    for idx in indices:
        pts = np.concatenate([np.zeros((1, 2)), vocab_np[idx, :, :2]])  # prepend (0,0)
        ax.plot(pts[:, 1], pts[:, 0], color=color, lw=lw, alpha=alpha, zorder=zorder)


def plot_trajs_by_score(ax, vocab_np: np.ndarray, indices: np.ndarray,
                        scores: np.ndarray, cmap_name: str = 'Greys_r',
                        alpha: float = 0.70, lw: float = 1.3,
                        zorder: int = 5):
    """
    Plot all DRC-selected trajectories on a BEV axis colored by their score value.
    scores: float array in [0, 1], same length as indices.
    Returns a ScalarMappable for adding a colorbar.
    """
    norm     = mcolors.Normalize(vmin=0.0, vmax=1.0)
    colormap = matplotlib.colormaps[cmap_name]

    if len(indices) == 0:
        sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
        sm.set_array([])
        return sm

    for idx, s in zip(indices, scores):
        pts   = np.concatenate([np.zeros((1, 2)), vocab_np[idx, :, :2]])
        color = colormap(norm(float(s)))
        ax.plot(pts[:, 1], pts[:, 0], color=color, lw=lw, alpha=alpha, zorder=zorder)

    sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
    sm.set_array([])
    return sm


def plot_endpoint_scatter(ax, vocab_np: np.ndarray, indices: np.ndarray,
                          scores: np.ndarray, cmap_name: str = 'viridis',
                          s: float = 8, alpha: float = 0.80,
                          zorder: int = 6):
    """
    Scatter plot of trajectory END POINTS colored by score (viridis by default).
    Each trajectory's last waypoint vocab_np[idx, -1, :2] is plotted as a dot.
    Returns a ScalarMappable for adding a colorbar.
    """
    norm     = mcolors.Normalize(vmin=0.0, vmax=1.0)
    colormap = matplotlib.colormaps[cmap_name]

    if len(indices) == 0:
        sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
        sm.set_array([])
        return sm

    endpoints = vocab_np[indices, -1, :2]          # (n, 2): [x_fwd, y_lat]
    colors    = colormap(norm(scores.astype(float)))

    ax.scatter(endpoints[:, 1], endpoints[:, 0],   # plot(lat, fwd) → forward=up
               c=colors, s=s, alpha=alpha, zorder=zorder, linewidths=0)

    sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
    sm.set_array([])
    return sm


def add_gt_traj(ax, gt_pts: Optional[np.ndarray]):
    """Overlay the GT (ground-truth) trajectory on a BEV axis."""
    if gt_pts is None:
        return
    ax.plot(gt_pts[:, 1], gt_pts[:, 0], 'g-', lw=2.5,
            marker='o', markersize=3, zorder=100, label='GT')


def badge(ax, text: str, x: float, y: float, ha: str, va: str, color: str, fontsize: int = 9):
    """Generic rounded-rectangle text badge."""
    ax.text(x, y, text, transform=ax.transAxes,
            fontsize=fontsize, fontweight='bold', color='white',
            bbox=dict(boxstyle='round,pad=0.35', facecolor=color, alpha=0.92),
            ha=ha, va=va, zorder=200)


def add_command_badge(ax, cmd_idx: int, cmd_name: str):
    """Top-left: driving command badge."""
    badge(ax, f"▶ {cmd_name}", 0.03, 0.97, 'left', 'top',
          DRIVING_COMMAND_COLORS.get(cmd_idx, '#95A5A6'))


def add_count_badge(ax, count: int, total: int, stage_label: str):
    """Top-right: trajectory count badge."""
    badge(ax, f"{stage_label}\n{count} / {total}", 0.97, 0.97,
          'right', 'top', '#2C3E50', fontsize=8)


# =========================================================
# Score bar chart
# =========================================================

def plot_score_bars(ax, predictions: Dict, drc_final_pos: int,
                    gt_token_scores: Optional[Dict] = None,
                    final_vocab_idx: int = 0):
    """
    Grouped bar chart: Predicted vs GT score for the final selected trajectory.

    Parameters
    ----------
    drc_final_pos   : position in DRC-scored array → index into predictions[metric]
    gt_token_scores : {metric: np.ndarray(16384,)} for this scene token (may be None)
    final_vocab_idx : original vocab index of selected trajectory → GT lookup key
      - 'imi'  : softmax probability (no GT equivalent → predicted only)
      - others : sigmoid prediction vs GT float score
    """
    labels, pred_scores, gt_scores_list = [], [], []

    for metric in SCORE_METRICS:
        score_arr = get_drc_score_array(predictions, metric)
        if score_arr is None or drc_final_pos >= len(score_arr):
            continue
        pred_s = float(score_arr[drc_final_pos])

        # GT score: indexed by original vocab index (full 16384 array)
        gt_s = None
        if gt_token_scores is not None and metric in gt_token_scores:
            gt_arr = gt_token_scores[metric]
            if final_vocab_idx < len(gt_arr):
                gt_s = float(gt_arr[final_vocab_idx])

        labels.append(METRIC_DISPLAY_NAMES.get(metric, metric))
        pred_scores.append(pred_s)
        gt_scores_list.append(gt_s)

    if not labels:
        ax.text(0.5, 0.5, 'No scores available',
                transform=ax.transAxes, ha='center', va='center', fontsize=12)
        return

    n     = len(labels)
    x     = np.arange(n)
    has_gt = any(g is not None for g in gt_scores_list)
    w     = 0.38 if has_gt else 0.55

    # ── Predicted bars (left) ────────────────────────────────────────────────
    pred_bars = ax.bar(x - w / 2 if has_gt else x,
                       pred_scores, width=w, alpha=0.88, zorder=3, label='Predicted')
    for bar, s in zip(pred_bars, pred_scores):
        bar.set_color('#27AE60' if s >= 0.8 else ('#F39C12' if s >= 0.5 else '#E74C3C'))
        ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.015,
                f'{s:.2f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

    # ── GT bars (right, grey) ─────────────────────────────────────────────────
    if has_gt:
        gt_vals = [g if g is not None else 0.0 for g in gt_scores_list]
        gt_bars = ax.bar(x + w / 2, gt_vals, width=w, alpha=0.70, zorder=3,
                         color='#5D6D7E', label='GT')
        for bar, orig in zip(gt_bars, gt_scores_list):
            if orig is None:
                bar.set_alpha(0.0)   # hide bar when GT unavailable (e.g. imi)
                continue
            ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.015,
                    f'{orig:.2f}', ha='center', va='bottom', fontsize=8, color='#2C3E50')

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, 1.22)
    ax.set_ylabel('Score  (sigmoid / softmax)', fontsize=10)
    ax.set_title(
        f'Predicted vs GT Scores — Selected Trajectory  (vocab idx: {final_vocab_idx})',
        fontsize=11, fontweight='bold',
    )
    ax.axhline(0.5, color='grey', ls='--', alpha=0.5, label='threshold = 0.5')
    ax.grid(axis='y', alpha=0.3, zorder=0)
    ax.legend(fontsize=9, loc='upper right')


# =========================================================
# Main visualization: 4-stage pruning figure
# =========================================================

def visualize_pruning_stages(
    scene,
    features:        Dict[str, torch.Tensor],
    targets:         Dict[str, torch.Tensor],
    predictions:     Dict[str, torch.Tensor],
    token:           str,
    output_path:     Path,
    cmd_idx:         int,
    cmd_name:        str,
    gt_token_scores: Optional[Dict] = None,
    vx:              float = 0.0,
    vy:              float = 0.0,
):
    """
    Create and save a single figure with 3 rows:
      Row 0  [Camera]   — stitched L0+F0+R0 image
      Row 1  [4 BEVs]   — Full Vocab | After GRC | After DRC | Final Selection
      Row 2  [Scores]   — per-metric score bar chart

    The BEV panels share the same color scheme:
      grey  = full vocab background
      blue  = GRC-selected trajectories
      orange= DRC-selected trajectories
      red   = final selected trajectory
      green = GT trajectory
    """
    frame_idx  = scene.scene_metadata.num_history_frames - 1
    frame      = scene.frames[frame_idx]
    stitched   = get_stitched_camera(scene, frame_idx)

    # --- Vocab trajectories ---------------------------------------------------
    vocab = predictions.get('trajectory_vocab')
    if vocab is None:
        print(f"  [SKIP] No trajectory_vocab for token {token}")
        return
    vocab_np = vocab.cpu().numpy() if isinstance(vocab, torch.Tensor) else np.asarray(vocab)
    total    = len(vocab_np)

    # --- Pruning indices  (all converted to 1-D numpy) -----------------------
    grc_idx         = to_numpy_indices(predictions.get('grc_selected_indices'), total)
    drc_idx         = to_numpy_indices(predictions.get('drc_selected_indices'), total)
    final_t         = predictions.get('selected_indices')
    final_vocab_idx = int(to_numpy_indices(final_t, total).flat[0])

    # Position of the selected trajectory within DRC-scored arrays
    drc_final_pos = find_drc_final_pos(drc_idx, final_vocab_idx)

    # --- GT trajectory (check features, targets, predictions) ----------------
    def _extract_traj(d, key='gt_trajectory'):
        t = d.get(key) if d else None
        if t is None:
            return None
        arr = t.cpu().numpy() if isinstance(t, torch.Tensor) else np.asarray(t)
        return arr[0] if arr.ndim == 3 else arr  # (T, 3)

    gt_np  = (_extract_traj(features)
               or _extract_traj(targets)
               or _extract_traj(predictions))
    gt_pts = (np.concatenate([np.zeros((1, 2)), gt_np[:, :2]])
              if gt_np is not None else None)

    # Final trajectory waypoints
    final_pts = np.concatenate([np.zeros((1, 2)), vocab_np[final_vocab_idx, :, :2]])

    # =========================================================================
    # Figure layout  (GridSpec)
    # =========================================================================
    fig = plt.figure(figsize=(24, 17))
    gs  = fig.add_gridspec(
        3, 4,
        height_ratios=[1.3, 2.8, 1.6],
        hspace=0.38, wspace=0.18,
    )

    ax_cam = fig.add_subplot(gs[0, :])   # camera row (full width)
    ax1    = fig.add_subplot(gs[1, 0])   # BEV 1: full vocab
    ax2    = fig.add_subplot(gs[1, 1])   # BEV 2: after GRC
    ax3    = fig.add_subplot(gs[1, 2])   # BEV 3: after DRC
    ax4    = fig.add_subplot(gs[1, 3])   # BEV 4: final selection
    ax_sc  = fig.add_subplot(gs[2, :])   # score bars (full width)

    # --- Camera row ----------------------------------------------------------
    ax_cam.imshow(stitched)
    ax_cam.set_title(f'Token: {token}', fontsize=11)
    ax_cam.axis('off')

    all_idx = np.arange(total)

    vis_drc = drc_idx

    # GRC: show all GRC-selected trajectories
    vis_grc = grc_idx

    # --- BEV 1: Full Vocab ---------------------------------------------------
    add_configured_bev_on_ax(ax1, scene.map_api, frame)
    plot_trajs(ax1, vocab_np, all_idx,
               color='#7F8C8D', alpha=0.45, lw=0.9)
    add_gt_traj(ax1, gt_pts)
    configure_bev_ax(ax1)
    add_velocity_overlay(ax1, vx, vy)
    ax1.set_title('① Full Vocab', fontsize=10, fontweight='bold')
    add_command_badge(ax1, cmd_idx, cmd_name)
    add_count_badge(ax1, total, total, "Vocab")

    # --- BEV 2: After GRC (Global Route Compliance) --------------------------
    add_configured_bev_on_ax(ax2, scene.map_api, frame)
    # grey background = discarded by GRC
    plot_trajs(ax2, vocab_np, all_idx,
               color='#BDC3C7', alpha=0.10, lw=0.5)
    # blue = GRC-selected (vis_grc always contains all vis_drc trajectories)
    plot_trajs(ax2, vocab_np, vis_grc,
               color='#2980B9', alpha=0.65, lw=1.0, zorder=10)
    add_gt_traj(ax2, gt_pts)
    configure_bev_ax(ax2)
    add_velocity_overlay(ax2, vx, vy)
    ax2.set_title('② After GRC  (Route Compliance)', fontsize=10, fontweight='bold')
    add_command_badge(ax2, cmd_idx, cmd_name)
    add_count_badge(ax2, len(grc_idx), total, "GRC")

    # --- BEV 3: After DRC (Dynamic Reachability Compliance) ------------------
    add_configured_bev_on_ax(ax3, scene.map_api, frame)
    # orange trajectory lines only — DRC is a kinematic filter, no score
    plot_trajs(ax3, vocab_np, vis_drc,
               color='#E67E22', alpha=0.65, lw=1.0, zorder=5)
    add_gt_traj(ax3, gt_pts)
    configure_bev_ax(ax3)
    add_velocity_overlay(ax3, vx, vy)
    ax3.set_title('③ After DRC  (Reachability)', fontsize=10, fontweight='bold')
    add_command_badge(ax3, cmd_idx, cmd_name)
    add_count_badge(ax3, len(drc_idx), total, "DRC")

    # --- BEV 4: Final Selection (score-argmax) --------------------------------
    add_configured_bev_on_ax(ax4, scene.map_api, frame)
    # faded DRC background
    plot_trajs(ax4, vocab_np, vis_drc,
               color='#F0B27A', alpha=0.20, lw=0.6)
    # red = final trajectory
    ax4.plot(final_pts[:, 1], final_pts[:, 0],
             color='#E74C3C', lw=3.2, marker='o', markersize=5,
             zorder=110, label='Selected')
    add_gt_traj(ax4, gt_pts)
    configure_bev_ax(ax4)
    add_velocity_overlay(ax4, vx, vy)
    ax4.set_title('④ Final Selection  (Score Argmax)', fontsize=10, fontweight='bold')
    add_command_badge(ax4, cmd_idx, cmd_name)
    badge(ax4, f"vocab idx: {final_vocab_idx}", 0.97, 0.03, 'right', 'bottom', '#E74C3C', fontsize=8)

    # --- Score bars ----------------------------------------------------------
    plot_score_bars(ax_sc, predictions, drc_final_pos,
                    gt_token_scores=gt_token_scores,
                    final_vocab_idx=final_vocab_idx)

    # --- Global legend -------------------------------------------------------
    legend_elems = [
        Line2D([0], [0], color='#7F8C8D', lw=2,
               label=f'Full Vocab  ({total})'),
        Line2D([0], [0], color='#2980B9', lw=2,
               label=f'After GRC  ({len(grc_idx)})'),
        Line2D([0], [0], color='#E67E22', lw=2,
               label=f'After DRC  ({len(drc_idx)})'),
        Line2D([0], [0], color='#E74C3C', lw=3, marker='o', markersize=5,
               label='Final Trajectory'),
        Line2D([0], [0], color='green',   lw=2, marker='o', markersize=4,
               label='GT Trajectory'),
    ]
    fig.legend(handles=legend_elems, loc='lower center', ncol=5, fontsize=9,
               bbox_to_anchor=(0.5, -0.01), framealpha=0.9)

    # --- Super title with command color --------------------------------------
    cmd_color = DRIVING_COMMAND_COLORS.get(cmd_idx, '#95A5A6')
    fig.suptitle(
        f'H-Safe Pruning Pipeline  |  Command: {cmd_name}  |  Token: {token}',
        fontsize=13, fontweight='bold', color=cmd_color, y=1.005,
    )

    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved: {output_path}")


# =========================================================
# Per-metric score map visualization
# =========================================================

def visualize_score_per_metric(
    scene,
    features:        Dict[str, torch.Tensor],
    targets:         Dict[str, torch.Tensor],
    predictions:     Dict[str, torch.Tensor],
    token:           str,
    base_output_dir: Path,
    cmd_idx:         int,
    cmd_name:        str,
    sample_idx:      int = 0,
    gt_token_scores: Optional[Dict] = None,
    vx:              float = 0.0,
    vy:              float = 0.0,
):
    """
    For each of the 8 score metrics, create a side-by-side BEV visualization:
      Left  panel : DRC trajectories colored by *predicted* score (sigmoid/softmax)
      Right panel : DRC trajectories colored by *GT* score from pkl (if available)
                    For 'imi' (no GT) the right panel shows "No GT" message.

    Saved to:
      base_output_dir / {metric_name} / {metric}_{cmd}_{i:04d}_{token[:16]}.png
    """
    vocab = predictions.get('trajectory_vocab')
    if vocab is None:
        return

    vocab_np = vocab.cpu().numpy() if isinstance(vocab, torch.Tensor) else np.asarray(vocab)
    total    = len(vocab_np)

    drc_idx         = to_numpy_indices(predictions.get('drc_selected_indices'), total)

    frame_idx = scene.scene_metadata.num_history_frames - 1
    frame     = scene.frames[frame_idx]
    stitched  = get_stitched_camera(scene, frame_idx)

    def _extract_traj(d, key='gt_trajectory'):
        t = d.get(key) if d else None
        if t is None:
            return None
        arr = t.cpu().numpy() if isinstance(t, torch.Tensor) else np.asarray(t)
        return arr[0] if arr.ndim == 3 else arr

    gt_np   = (_extract_traj(features)
                or _extract_traj(targets)
                or _extract_traj(predictions))
    gt_pts  = (np.concatenate([np.zeros((1, 2)), gt_np[:, :2]])
               if gt_np is not None else None)
    cmd_color    = DRIVING_COMMAND_COLORS.get(cmd_idx, '#95A5A6')
    score_label  = {'imi': 'Softmax prob'}

    def _draw_bev_panel(ax, bev_scores, panel_title):
        """Helper: draw one BEV panel with color-coded DRC trajectories.
        Renders trajectory lines (viridis) + endpoint scatter dots on top.
        """
        # Shared subsampling: same helper as visualize_pruning_stages
        _sel, _vis_idx = subsample_drc(drc_idx)
        _vis_scores    = bev_scores[_sel]

        add_configured_bev_on_ax(ax, scene.map_api, frame)
        # Trajectory lines — viridis, thin, semi-transparent
        sm = plot_trajs_by_score(
            ax, vocab_np, _vis_idx, _vis_scores,
            cmap_name='viridis', alpha=0.30, lw=0.8,
        )
        # Endpoint scatter — viridis colored dots (same indices as lines)
        plot_endpoint_scatter(
            ax, vocab_np, _vis_idx, _vis_scores,
            cmap_name='viridis', s=10, alpha=0.85, zorder=8,
        )
        add_gt_traj(ax, gt_pts)
        configure_bev_ax(ax)
        add_velocity_overlay(ax, vx, vy)
        add_command_badge(ax, cmd_idx, cmd_name)
        badge(ax, f"DRC: {len(drc_idx)}",
              0.97, 0.97, 'right', 'top', '#2C3E50', fontsize=8)
        ax.set_title(panel_title, fontsize=10, fontweight='bold')

        if gt_pts is not None:
            legend_elems = [
                Line2D([0], [0], color='green', lw=2, marker='o', markersize=4,
                       label='GT Traj')
            ]
            ax.legend(handles=legend_elems, fontsize=8, loc='lower right')
        return sm

    for metric in SCORE_METRICS_ALL:
        pred_arr = get_drc_score_array(predictions, metric)
        if pred_arr is None:
            continue

        metric_display = METRIC_DISPLAY_NAMES.get(metric, metric).replace('\n', ' ')
        weight_label   = METRIC_WEIGHT_LABEL.get(metric, '')
        cb_label       = score_label.get(metric, 'Sigmoid score')

        # GT score array for DRC-selected trajectories (None for 'imi')
        gt_full = (gt_token_scores or {}).get(metric)   # shape (16384,) or None
        gt_drc  = None
        if gt_full is not None:
            gt_drc = np.asarray(gt_full, dtype=np.float32)[drc_idx]  # (num_drc,)

        # ── Create output directory ──────────────────────────────────────────
        metric_dir = base_output_dir / metric
        metric_dir.mkdir(parents=True, exist_ok=True)

        # ── Figure layout: camera top, 2 BEVs bottom ─────────────────────────
        fig = plt.figure(figsize=(24, 13))
        gs  = fig.add_gridspec(2, 2, height_ratios=[1.0, 3.0], hspace=0.22, wspace=0.12)
        ax_cam = fig.add_subplot(gs[0, :])   # camera full width
        ax_pred = fig.add_subplot(gs[1, 0])  # left  : predicted
        ax_gt   = fig.add_subplot(gs[1, 1])  # right : GT

        # Camera row
        ax_cam.imshow(stitched)
        ax_cam.set_title(f'Token: {token}', fontsize=10)
        ax_cam.axis('off')

        # Left panel — Predicted
        sm_pred = _draw_bev_panel(
            ax_pred, pred_arr,
            f'Predicted  ({metric_display}, {weight_label})',
        )
        cbar_pred = plt.colorbar(sm_pred, ax=ax_pred, fraction=0.03, pad=0.02)
        cbar_pred.set_label(cb_label, fontsize=9)

        # Right panel — GT (or "No GT" message for imi)
        if gt_drc is not None:
            sm_gt = _draw_bev_panel(
                ax_gt, gt_drc,
                f'GT  ({metric_display})',
            )
            cbar_gt = plt.colorbar(sm_gt, ax=ax_gt, fraction=0.03, pad=0.02)
            cbar_gt.set_label('GT score', fontsize=9)
        else:
            add_configured_bev_on_ax(ax_gt, scene.map_api, frame)
            configure_bev_ax(ax_gt)
            ax_gt.text(0.5, 0.5, 'No GT available\n(IMI metric)',
                       transform=ax_gt.transAxes, ha='center', va='center',
                       fontsize=14, color='grey',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            ax_gt.set_title(f'GT  ({metric_display})', fontsize=10, fontweight='bold')

        fig.suptitle(
            f'Predicted vs GT — {metric_display}  |  Command: {cmd_name}  |  Token: {token}',
            fontsize=12, fontweight='bold', color=cmd_color, y=1.01,
        )

        out_name = f"{metric}_{cmd_name}_{sample_idx:04d}_{token[:16]}.png"
        out_path = metric_dir / out_name
        plt.savefig(out_path, dpi=130, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(f"    [{metric}] Saved: {out_path}")


# =========================================================
# Main (Hydra entry point)
# =========================================================

@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Pre-create per-metric score directories
    for metric in SCORE_METRICS_ALL:
        (OUTPUT_DIR / metric).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 0. Load GT scores
    # ------------------------------------------------------------------
    gt_scores = load_gt_scores(GT_SCORE_PATH)  # {token: {metric: array(16384,)}}

    # ------------------------------------------------------------------
    # 1. Instantiate DRSI agent  (drsi_vov has pruning=True by default)
    # ------------------------------------------------------------------
    print(f"Loading checkpoint: {CKPT_PATH}")

    agent = instantiate(cfg.agent)

    checkpoint  = torch.load(CKPT_PATH, map_location='cpu')
    state_dict  = checkpoint['state_dict']
    msg = agent.load_state_dict(
        {k.replace("agent.", ""): v for k, v in state_dict.items()},
        strict=False,
    )
    print(f"  Checkpoint loaded: {msg}")

    agent.eval()
    agent = agent.to(device)

    # ------------------------------------------------------------------
    # 2. Load navtest dataset
    # ------------------------------------------------------------------
    print("Loading dataset...")
    data_path            = Path(cfg.navsim_log_path)
    original_sensor_path = Path(cfg.original_sensor_path)

    scene_filter = instantiate(cfg.train_test_split.scene_filter)

    # Intersect scene_filter.log_names with val_logs (same as training script)
    val_logs = getattr(cfg, 'val_logs', None)
    if val_logs is not None:
        if scene_filter.log_names is not None:
            scene_filter.log_names = [n for n in scene_filter.log_names if n in val_logs]
        else:
            scene_filter.log_names = val_logs

    scene_loader = SceneLoader(
        original_sensor_path=original_sensor_path,
        data_path=data_path,
        scene_filter=scene_filter,
        sensor_config=agent.get_sensor_config(),
    )

    if cfg.use_cache_without_dataset:
        dataset = CacheOnlyDataset(
            cache_path=cfg.cache_path,
            feature_builders=agent.get_feature_builders(),
            target_builders=agent.get_target_builders(),
            log_names=scene_filter.log_names,
        )
    else:
        dataset = Dataset(
            scene_loader=scene_loader,
            feature_builders=agent.get_feature_builders(),
            target_builders=agent.get_target_builders(),
            cache_path=cfg.cache_path,
            force_cache_computation=cfg.force_cache_computation,
        )

    total_samples = len(dataset)
    target_cmds   = {0, 1, 2}   # LEFT, STRAIGHT, RIGHT  (skip UNKNOWN)
    print(f"Dataset loaded.  Total samples: {total_samples},  "
          f"Target: {NUM_SAMPLES_PER_BIN} scenes × {len(SPEED_BINS)} speed bins")

    # ------------------------------------------------------------------
    # 3. Inference + visualize  (speed-bin-balanced sampling)
    # ------------------------------------------------------------------
    bin_counts  = {label: 0 for _, _, label in SPEED_BINS}   # counts per speed bin
    vis_count   = 0

    with torch.no_grad():
        for i in range(total_samples):

            features, targets, token = dataset[i]

            # ── CPU-only batch unsqueeze (for cmd / velocity extraction) ──
            features_cpu = {}
            for k, v in features.items():
                if isinstance(v, torch.Tensor):
                    features_cpu[k] = v.unsqueeze(0)
                elif isinstance(v, list) and v and isinstance(v[0], torch.Tensor):
                    features_cpu[k] = [t.unsqueeze(0) if t is not None else None for t in v]
                else:
                    features_cpu[k] = v

            # Extract driving command (no GPU needed)
            cmd_idx, cmd_name = get_driving_command_info(features_cpu)

            # Skip UNKNOWN driving command
            if cmd_idx not in target_cmds:
                continue

            # Extract velocity and compute speed (still CPU)
            vx, vy = get_ego_velocity(features_cpu)
            speed = float(np.sqrt(vx ** 2 + vy ** 2))

            # Determine speed bin — skip if outside all defined bins (e.g. > 8 m/s)
            speed_bin = get_speed_bin_label(speed)
            if speed_bin is None:
                continue

            # Skip already-saturated speed bins
            if bin_counts[speed_bin] >= NUM_SAMPLES_PER_BIN:
                continue

            bin_counts[speed_bin] += 1
            vis_count += 1
            bin_summary = "  ".join(
                f"{label}:{bin_counts[label]}/{NUM_SAMPLES_PER_BIN}"
                for _, _, label in SPEED_BINS
            )
            print(f"\n[{vis_count}] {bin_summary}  |  Token: {token}  |  Cmd: {cmd_name}  |  Speed: {speed:.2f} m/s")

            # ── Move to GPU for inference ─────────────────────────────────
            features_batch = {}
            for k, v in features_cpu.items():
                if isinstance(v, torch.Tensor):
                    features_batch[k] = v.to(device)
                elif isinstance(v, list) and v and isinstance(v[0], torch.Tensor):
                    features_batch[k] = [t.to(device) if t is not None else None for t in v]
                else:
                    features_batch[k] = v

            # Forward pass  (pruning mode returns grc/drc/selected indices)
            predictions = agent.forward(features_batch)

            # Sanity check: pruning must have run
            if 'grc_selected_indices' not in predictions:
                print("  [SKIP] No grc_selected_indices — pruning may be disabled.")
                continue

            _total = len(predictions['trajectory_vocab'])
            _grc = to_numpy_indices(predictions.get('grc_selected_indices'), _total)
            _drc = to_numpy_indices(predictions.get('drc_selected_indices'), _total)
            n_grc, n_drc = len(_grc), len(_drc)
            print(f"  Vocab: {_total}  "
                  f"→ GRC: {n_grc}  → DRC: {n_drc}  → Final: 1"
                  f"  |  vx={vx:+.2f}  vy={vy:+.2f} m/s")

            # Verify DRC ⊆ GRC (all DRC indices must exist in GRC indices)
            _grc_set = set(_grc.tolist())
            _drc_outside = [idx for idx in _drc.tolist() if idx not in _grc_set]
            if _drc_outside:
                print(f"  [BUG] {len(_drc_outside)} DRC indices are NOT in GRC! "
                      f"e.g. {_drc_outside[:5]}")
            else:
                print(f"  [OK ] DRC ⊆ GRC verified  ({n_drc}/{n_grc})")

            # Load scene for BEV map
            try:
                scene = scene_loader.get_scene_from_token(token)
            except Exception as e:
                print(f"  [SKIP] Could not load scene: {e}")
                continue

            # ── (A) Main pruning-stage figure ────────────────────────────────
            out_name    = f"pruning_{speed_bin}_{cmd_name}_{bin_counts[speed_bin]:02d}_{token[:16]}.png"
            output_path = OUTPUT_DIR / out_name

            gt_token = gt_scores.get(token)   # {metric: array(16384,)} or None

            visualize_pruning_stages(
                scene=scene,
                features=features_batch,
                targets=targets,
                predictions=predictions,
                token=token,
                output_path=output_path,
                cmd_idx=cmd_idx,
                cmd_name=cmd_name,
                gt_token_scores=gt_token,
                vx=vx,
                vy=vy,
            )

            # ── (B) Per-metric score map figures ─────────────────────────────
            print(f"  Generating per-metric score maps...")
            visualize_score_per_metric(
                scene=scene,
                features=features_batch,
                targets=targets,
                predictions=predictions,
                token=token,
                base_output_dir=OUTPUT_DIR,
                cmd_idx=cmd_idx,
                cmd_name=cmd_name,
                sample_idx=vis_count,
                gt_token_scores=gt_token,
                vx=vx,
                vy=vy,
            )

            # ── Early stop when all speed bins are satisfied ──────────────────
            if all(bin_counts[label] >= NUM_SAMPLES_PER_BIN for _, _, label in SPEED_BINS):
                print(f"\n  All speed bins reached {NUM_SAMPLES_PER_BIN} samples. Done.")
                break

    print(f"\nVisualization complete.  Outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
