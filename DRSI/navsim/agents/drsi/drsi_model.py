# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
from typing import Dict

import numpy as np
import torch
import torch.nn as nn

from navsim.agents.drsi.drsi_backbone import DRSIBackbone
from navsim.agents.drsi.drsi_config import DRSIConfig
from navsim.agents.transfuser.transfuser_model import AgentHead
from navsim.agents.utils.attn import MemoryEffTransformer
from navsim.agents.utils.nerf import nerf_positional_encoding

import pickle


class DRSIModel(nn.Module):
    def __init__(self, config: DRSIConfig):
        super().__init__()

        self._query_splits = [
            config.num_bounding_boxes,
        ]

        self._config = config
        self._backbone = DRSIBackbone(config)

        img_num = 2 if config.use_back_view else 1
        self._keyval_embedding = nn.Embedding(
            config.img_vert_anchors * config.img_horz_anchors * img_num, config.tf_d_model
        )  # 8x8 feature grid + trajectory
        self._query_embedding = nn.Embedding(sum(self._query_splits), config.tf_d_model)

        # usually, the BEV features are variable in size.
        self.downscale_layer = nn.Conv2d(self._backbone.img_feat_c, config.tf_d_model, kernel_size=1)
        self._status_encoding = nn.Linear((4 + 2 + 2) * config.num_ego_status, config.tf_d_model)

        tf_decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.tf_d_model,
            nhead=config.tf_num_head,
            dim_feedforward=config.tf_d_ffn,
            dropout=config.tf_dropout,
            batch_first=True,
        )

        self._tf_decoder = nn.TransformerDecoder(tf_decoder_layer, config.tf_num_layers)
        self._agent_head = AgentHead(
            num_agents=config.num_bounding_boxes,
            d_ffn=config.tf_d_ffn,
            d_model=config.tf_d_model,
        )

        self._trajectory_head = DRSITrajHead(
            num_poses=config.trajectory_sampling.num_poses,
            d_ffn=config.tf_d_ffn,
            d_model=config.tf_d_model,
            nhead=config.vadv2_head_nhead,
            nlayers=config.vadv2_head_nlayers,
            vocab_path=config.vocab_path,
            config=config
        )


    def img_feat_blc(self, camera_feature):
        img_features = self._backbone(camera_feature)
        img_features = self.downscale_layer(img_features).flatten(-2, -1)
        img_features = img_features.permute(0, 2, 1)
        return img_features


    def forward(self, features: Dict[str, torch.Tensor],
                interpolated_traj=None) -> Dict[str, torch.Tensor]:
        status_feature: torch.Tensor = features["status_feature"][0]
        camera_feature = features["camera_feature"]

        if self._config.num_ego_status == 1 and status_feature.shape[1] == 32:
            status_encoding = self._status_encoding(status_feature[:, :8])
        else:
            status_encoding = self._status_encoding(status_feature)

        # original
        if isinstance(camera_feature, list):
            camera_feature = camera_feature[-1]
        img_features = self.img_feat_blc(camera_feature)
        if self._config.use_back_view:
            img_features_back = self.img_feat_blc(features["camera_feature_back"])
            img_features = torch.cat([img_features, img_features_back], 1)
        keyval = img_features

        keyval += self._keyval_embedding.weight[None, ...]

        output: Dict[str, torch.Tensor] = {}
        
        # Pruning mode (only at inference with batch size 1)
        if not self.training and self._config.pruning:
            assert status_feature.shape[0] == 1, "Batch size must be 1 during inference with pruning."
            driving_command = status_feature[:, :4] # one-hot encoded driving command (left, straight, right, unknown)
            current_velocity = status_feature[:, 4:6] # vx, vy
            trajectory = self._trajectory_head(keyval, status_encoding, driving_command, current_velocity, interpolated_traj)
        else:
            driving_command = None
            current_velocity = None
            trajectory = self._trajectory_head(keyval, status_encoding, driving_command, current_velocity, interpolated_traj)
        output.update(trajectory)
        return output
    
    def build_vocab_cache(self,):
        self._trajectory_head.build_vocab_cache()
        


class CrossAttentionLayer(nn.Module):
    def __init__(self, d_model, nhead, d_ffn, dropout=0.0):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ffn),
            nn.ReLU(),
            nn.Linear(d_ffn, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt, memory):
        tgt2 = self.cross_attn(tgt, memory, memory)[0]
        tgt = self.norm1(tgt + self.dropout(tgt2))
        tgt2 = self.ffn(tgt)
        tgt = self.norm2(tgt + self.dropout(tgt2))
        return tgt


class CrossAttentionBlock(nn.Module):
    def __init__(self, d_model, nhead, d_ffn, dropout=0.0, nlayers=1):
        super().__init__()
        self.layers = nn.ModuleList([
            CrossAttentionLayer(d_model, nhead, d_ffn, dropout)
            for _ in range(nlayers)
        ])

    def forward(self, tgt, memory):
        for layer in self.layers:
            tgt = layer(tgt, memory)
        return tgt


class DRSITrajHead(nn.Module):
    def __init__(self, num_poses: int, d_ffn: int, d_model: int, vocab_path: str,
                 nhead: int, nlayers: int, config: DRSIConfig = None
                 ):
        super().__init__()
        self.config = config
        self._num_poses = num_poses
        self.transformer = CrossAttentionBlock(
            d_model, nhead, d_ffn,
            dropout=0.0, nlayers=nlayers
        )
        self.vocab = nn.Parameter(
            torch.from_numpy(np.load(vocab_path)),
            requires_grad=False
        )
        
        # Make vocab_embedded as a buffer to store the embedded vocab for inference
        self.register_buffer("vocab_embedded", torch.zeros(1,self.vocab.size(0),d_model), persistent=True)
        self.vocab_cluster = pickle.load(open(config.vocab_cluster_path,"rb"))
        
        for k,v in self.vocab_cluster.items():
            self.register_buffer(f"vocab_cluster_{k}", torch.tensor(v, dtype=torch.long))
            # self.vocab_cluster[k] = torch.tensor(self.vocab_cluster[k], dtype=torch.long)
        self.register_buffer(f"vocab_cluster_unknown", torch.arange(self.vocab.size(0), dtype=torch.long))
        # self.vocab_cluster["unknown"] = torch.arange(self.vocab.size(0), dtype=torch.long)

        self.heads = nn.ModuleDict({
            'no_at_fault_collisions': nn.Sequential(
                nn.Linear(d_model, d_ffn),
                nn.ReLU(),
                nn.Linear(d_ffn, 1),
            ),
            'drivable_area_compliance':
                nn.Sequential(
                    nn.Linear(d_model, d_ffn),
                    nn.ReLU(),
                    nn.Linear(d_ffn, 1),
                ),
            'time_to_collision_within_bound': nn.Sequential(
                nn.Linear(d_model, d_ffn),
                nn.ReLU(),
                nn.Linear(d_ffn, 1),
            ),
            'ego_progress': nn.Sequential(
                nn.Linear(d_model, d_ffn),
                nn.ReLU(),
                nn.Linear(d_ffn, 1),
            ),
            'driving_direction_compliance': nn.Sequential(
                nn.Linear(d_model, d_ffn),
                nn.ReLU(),
                nn.Linear(d_ffn, 1),
            ),
            'lane_keeping': nn.Sequential(
                nn.Linear(d_model, d_ffn),
                nn.ReLU(),
                nn.Linear(d_ffn, 1),
            ),
            'traffic_light_compliance': nn.Sequential(
                nn.Linear(d_model, d_ffn),
                nn.ReLU(),
                nn.Linear(d_ffn, 1),
            ),
            'imi': nn.Sequential(
                nn.Linear(d_model, d_ffn),
                nn.ReLU(),
                nn.Linear(d_ffn, d_ffn),
                nn.ReLU(),
                nn.Linear(d_ffn, 1),
            )
        })

        self.inference_imi_weight = config.inference_imi_weight
        self.inference_da_weight = config.inference_da_weight
        self.normalize_vocab_pos = config.normalize_vocab_pos
        if self.normalize_vocab_pos:
            self.encoder = MemoryEffTransformer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=d_model * 4,
                dropout=0.0
            )
        self.use_nerf = config.use_nerf

        if self.use_nerf:
            self.pos_embed = nn.Sequential(
                nn.Linear(1040, d_ffn),
                nn.ReLU(),
                nn.Linear(d_ffn, d_model),
            )
        else:
            self.pos_embed = nn.Sequential(
                nn.Linear(num_poses * 3, d_ffn),
                nn.ReLU(),
                nn.Linear(d_ffn, d_model),
            )
            
    def forward(self, bev_feature, status_encoding, driving_command=None, current_velocity=None, interpolated_traj=None) -> Dict[str, torch.Tensor]:
        
        if not self.training and self.config.pruning:
            return self.forward_pruning(bev_feature, status_encoding, driving_command, current_velocity, interpolated_traj)
        
        return self.forward_training(bev_feature, status_encoding, interpolated_traj)

    def forward_training(self, bev_feature, status_encoding, interpolated_traj=None) -> Dict[str, torch.Tensor]:
        result = {}
        # vocab: 4096, 40, 3
        # bev_feature: B, 32, C
        # embedded_vocab: B, 4096, C
        vocab = self.vocab.data
        L, HORIZON, _ = vocab.shape
        B = bev_feature.shape[0]
        num_total = vocab.size(0)  # 16384
        if self.training and self.config.vocab_dropout:
            num_select = num_total // 2  # 8192
            indices = torch.randperm(num_total, device=vocab.device)[:num_select]
            vocab = vocab[indices]
            result['dropout_indices'] = indices
            L, HORIZON, _ = vocab.shape
        else:
            result['dropout_indices'] = torch.arange(num_total, device=vocab.device)
        result['trajectory_vocab_dropout'] = vocab
        if self.use_nerf:
            vocab = torch.cat(
                [
                    nerf_positional_encoding(vocab[..., :2]),
                    torch.cos(vocab[..., -1])[..., None],
                    torch.sin(vocab[..., -1])[..., None],
                ], dim=-1
            )

        if self.normalize_vocab_pos:
            embedded_vocab = self.pos_embed(vocab.view(L, -1))[None]
            embedded_vocab = self.encoder(embedded_vocab).repeat(B, 1, 1)
        else:
            embedded_vocab = self.pos_embed(vocab.view(L, -1))[None].repeat(B, 1, 1)
        tr_out = self.transformer(embedded_vocab, bev_feature)
        dist_status = tr_out + status_encoding.unsqueeze(1)

        # selected_indices: B,
        for k, head in self.heads.items():
            result[k] = head(dist_status).squeeze(-1)

        scores = (
                0.03 * result['imi'].softmax(-1).log() +
                0.1 * result['traffic_light_compliance'].sigmoid().log() +
                0.1 * result['no_at_fault_collisions'].sigmoid().log() +
                0.9 * result['drivable_area_compliance'].sigmoid().log() +
                0.2 * result['driving_direction_compliance'].sigmoid().log() +
                6.0 * (7.0 * result['time_to_collision_within_bound'].sigmoid() +
                       7.0 * result['ego_progress'].sigmoid() +
                       3.0 * result['lane_keeping'].sigmoid()
                       ).log()
        )


        selected_indices = scores.argmax(1)
        result["trajectory"] = self.vocab.data[selected_indices]
        result["trajectory_vocab"] = self.vocab.data
        result["selected_indices"] = selected_indices
        return result
    
    def forward_pruning(self, bev_feature, status_encoding, driving_command=None, current_velocity=None, interpolated_traj=None) -> Dict[str, torch.Tensor]:
        result = {}
        # vocab: 4096, 40, 3
        # bev_feature: B, 32, C
        # embedded_vocab: B, 4096, C
        result['trajectory_vocab_dropout'] = self.vocab.data
        
        
        # Global Route Compliance (GRC) pruning:
        # select a subset of the trajectory vocab based on the driving command
        
        # Select vocab cluster based on driving command
        driving_command_idx = torch.argmax(driving_command, dim=1)  # Assuming driving_command is one-hot encoded
        
        if driving_command_idx.item() == 0:
            cluster_indices = self.vocab_cluster_left
        elif driving_command_idx.item() == 1:
            cluster_indices = self.vocab_cluster_straight
        elif driving_command_idx.item() == 2:
            cluster_indices = self.vocab_cluster_right
        else:
            cluster_indices = self.vocab_cluster_unknown
        
        grc_pruned_vocab = self.vocab[cluster_indices,:,:]
        
        # Dynamic Reachability Compliance (DRC) pruning:
        # further prune the trajectory vocab based on the current velocity of the agent
        drc_selected_indices = self.dynamic_reachability_pruning(grc_pruned_vocab, current_velocity)
        if drc_selected_indices.numel() == 0:
            drc_selected_indices = copy.deepcopy(cluster_indices)
        else:
            drc_selected_indices = cluster_indices[drc_selected_indices]  # Map back to original vocab indices
        
        drc_pruned_vocab = self.vocab[drc_selected_indices,:,:]
        vocab_embedded = self.vocab_embedded[:, drc_selected_indices, :]
        

        tr_out = self.transformer(vocab_embedded, bev_feature)
        dist_status = tr_out + status_encoding.unsqueeze(1)

        # selected_indices: B,
        for k, head in self.heads.items():
            result[k] = head(dist_status).squeeze(-1)

        scores = (
                0.03 * result['imi'].softmax(-1).log() +
                0.1 * result['traffic_light_compliance'].sigmoid().log() +
                0.1 * result['no_at_fault_collisions'].sigmoid().log() +
                0.9 * result['drivable_area_compliance'].sigmoid().log() +
                0.2 * result['driving_direction_compliance'].sigmoid().log() +
                6.0 * (7.0 * result['time_to_collision_within_bound'].sigmoid() +
                       7.0 * result['ego_progress'].sigmoid() +
                       3.0 * result['lane_keeping'].sigmoid()
                       ).log()
        )


        selected_indices = scores.argmax(1)
        selected_indices = drc_selected_indices[selected_indices.tolist()]

        result["trajectory"] = self.vocab.data[selected_indices,:]
        result["trajectory_vocab"] = self.vocab.data
        result["selected_indices"] = selected_indices
        
        result["grc_selected_indices"] = cluster_indices
        result["drc_selected_indices"] = drc_selected_indices
        return result
    
    @torch.no_grad()
    def dynamic_reachability_pruning(self, vocab, current_velocity):
        
        def compute_curvature(traj_three_points):
            # traj_three_points: (num_traj, 3, 2)
            p1, p2, p3 = traj_three_points[:,0], traj_three_points[:,1], traj_three_points[:,2] # each is (num_traj, 2)
            a = torch.norm(p2 - p1, dim=1) # (num_traj,)
            b = torch.norm(p3 - p2, dim=1) # (num_traj,)
            c = torch.norm(p3 - p1, dim=1) # (num_traj,)
            s = (a + b + c) / 2
            area = torch.sqrt(s * (s - a) * (s - b) * (s - c))
            curvature = 4 * area / (a * b * c + 1e-6) # add small value to avoid division by zero
            return curvature
        
        vx_traj = vocab[:,10,0] # / 1.0 = vx
        vy_traj = vocab[:,10,1] # / 1.0 = vy
        ax = (vx_traj - current_velocity[0][0]) / 0.5
        
        traj_three_points = vocab[:,[0,5,10],:2] # shape: (num_traj, 3, 2)
        kappa = compute_curvature(traj_three_points) # shape: (num_traj,) 
        ay = (vy_traj - current_velocity[0][1]) / 0.5 + kappa * (vx_traj**2 + vy_traj**2) 
        
        score_ax_pos = -0.5 * torch.tanh(ax - 3.0) + 0.5
        score_ax_neg = 0.5 * torch.tanh(0.5*(ax + 5.0)) + 0.5
        
        score_ay = -0.5 * torch.tanh(0.5 * torch.abs(ay) - 3.0) + 0.5
        
        score_drc = ((score_ax_pos * score_ax_neg) + score_ay) / 2.0
                
        selected_indices = torch.where(score_drc > 0.5)[0]
        return selected_indices
        
        
        
        
    
    @torch.no_grad()
    def build_vocab_cache(self,):
        if self.normalize_vocab_pos:
            embedded_vocab = self.pos_embed(self.vocab.view(self.vocab.shape[0], -1))[None]
            embedded_vocab = self.encoder(embedded_vocab)
        else:
            embedded_vocab = self.pos_embed(self.vocab.view(self.vocab.shape[0], -1))[None]
            
        self.vocab_embedded = embedded_vocab.detach()

    
