

# DRSI: Decoupled Retrieval-Style Inference for Vocabulary-Based End-to-End Driving

**Dong-Gyu Kim · Hyun-Jun Kim · Jeong-Woo Park**
Dept. of Automotive Engineering, Hanyang University

## Submission Links

| Item | Link |
|---|---|
| Pre-recorded presentation video | [YouTube](https://www.youtube.com/watch?v=jhkgY0vVDiw) |
| Presentation slides | [FinalProject_HyunjunKim_Dongguekim_Jeongwoopark.pdf](FinalProject_HyunjunKim_Dongguekim_Jeongwoopark.pdf) |
| Report | [Report.pdf](Report.pdf) |
| Dataset | [NAVSIM](https://github.com/autonomousvision/navsim) (not included in this repo — see [Dataset](#dataset) below) |
| Demo | See [Demo](#demo) below — qualitative result figures from the report |
| Runnable notebook | [final-project.ipynb](final-project.ipynb) |

## Overview

Vocabulary-based end-to-end driving selects a future trajectory from a fixed candidate set, but existing planners repeatedly process the scene-invariant trajectory vocabulary together with the scene-dependent driving context in a single online forward pass. We recast this as a **retrieve-then-rerank** problem: the trajectory vocabulary is a static candidate collection, lightweight scene cues form a retrieval query, and the scene-conditioned evaluator acts as a reranker.

We propose **Decoupled Retrieval-Style Inference (DRSI)**, built on two modules:

- **Offline Candidate Indexing (OCI)** — caches scene-invariant trajectory embeddings for the fixed vocabulary offline, removing the need to forward the vocabulary embedding network online.
- **Scene-aware Candidate Retrieval (SCR)** — retrieves a compact, route-consistent (**Global Route Compliance**) and dynamically reachable (**Dynamic Reachability Compliance**) candidate subset online using only the driving command and ego motion state, before the scene-conditioned evaluator reranks the shortlist.

On NAVSIM v2, DRSI (Large, 16,384-candidate vocabulary) reduces inference latency from 163.4 ms to 23.9 ms (≈6.8× speedup) over a reproduced Hydra-MDP++ baseline, while improving `navhard` EPDMS from 37.6 to 40.2 and keeping `navtest` EPDMS nearly unchanged (86.9 → 86.1). Full method details, ablations, and qualitative results are in [Report.pdf](Report.pdf).

## Demo

This project does not ship a separate demo video. Instead, refer to the qualitative results in [Report.pdf](Report.pdf):

- **Fig. 3** — visualizes the Scene-aware Candidate Retrieval (SCR) pipeline in a low-speed right-turn scene: full vocabulary → after Global Route Compliance (GRC) retrieval → after Dynamic Reachability Compliance (DRC) retrieval.
- **Table I–IV** — latency / EPDMS comparison against the Hydra-MDP++ baseline, the `navhard`/`navtest` submetric breakdown, and the ablation study on each DRSI component.

[final-project.ipynb](final-project.ipynb) also reproduces the GRC/DRC retrieval behavior shown in Fig. 3 directly on the shipped trajectory vocabulary, with executed plots, so the retrieval mechanism can be inspected without needing the full NAVSIM dataset or a GPU.

## Dataset

This project trains and evaluates on **[NAVSIM](https://github.com/autonomousvision/navsim)** (built on OpenScene/nuPlan), which includes multi-camera sensor blobs and LiDAR logs and is **too large to store in this repository or on Google Drive**.

To set up the dataset and pretrained weights, follow **[DRSI/docs/install.md](DRSI/docs/install.md)**, which covers:

1. Downloading the OpenScene sensor blobs / navsim logs and nuPlan maps for the splits you need (`mini`, `trainval`, `test`, `navhard`, etc.) — see [DRSI/docs/splits.md](DRSI/docs/splits.md) for which split to use.
2. Downloading the simulated trajectory ground-truths (`traj_pdm_v2`) and the pretrained VoV backbone weights.
3. Installing the `navsim` devkit and its conda environment.

The top-level [DRSI/README.md](DRSI/README.md) additionally documents the expected `navsim_workspace` directory layout and provides direct download links for the pretrained DRSI checkpoint and driving-command cluster file used at inference time.

## Repository Structure

```
.
├── README.md                                          ← this file
├── final-project.ipynb                                ← runnable notebook (see below)
├── Report.pdf                                          ← full project report
├── FinalProject_HyunjunKim_Dongguekim_Jeongwoopark.pdf ← presentation slides
└── DRSI/                                               ← DRSI codebase (navsim-devkit + DRSI agent)
    ├── docs/                                           ← dataset / install / metrics / submission docs
    ├── navsim/agents/drsi/                             ← DRSI model, backbone, and SCR/OCI logic
    ├── scripts/drsi/                                   ← training / evaluation entry scripts
    ├── traj_final/                                     ← trajectory vocabulary files (16384.npy, 8192.npy)
    └── driving_command_cluster_generator.py            ← builds the GRC driving-command cluster file
```

## How to Run

1. **Quick look (no dataset needed):** open [final-project.ipynb](final-project.ipynb). It loads the shipped trajectory vocabulary and reproduces the GRC/DRC retrieval logic from `navsim/agents/drsi/drsi_model.py` with executed cells and plots, mirroring Fig. 3 of the report.
2. **Full training / evaluation (requires NAVSIM dataset + GPU):** set up the dataset and environment per [Dataset](#dataset) above, then follow [DRSI/README.md](DRSI/README.md) for training, EPDMS evaluation, and visualization commands.
