# DRSI

## Getting Started

Please refer to [install.md](docs/install.md) for downloading the dataset and installing the devkit.


## Workspace Structure

```
~/navsim_workspace
├── DRSI/                               ← this repository
│   └── traj_final/
│       ├── 16384.npy                   ← vocabulary file
│       └── cluster_labels_16384.pkl    ← download (see below) or generate via driving_command_cluster_generator.py
├── exp/
│   └── drsi_16384/                     ← created automatically during training
│       └── drsi_vov.ckpt               ← download (see below) or rename from epoch=XX-step=XX.ckpt after training
└── dataset/
    ├── maps/
    ├── models/
    │   └── dd3d_det_final.pth          ← VoV backbone pretrained weight
    ├── navsim_logs/
    │   ├── trainval/
    │   └── test/
    ├── sensor_blobs/
    │   ├── trainval/
    │   └── test/
    └── traj_pdm_v2/
        └── ori/
            └── navtrain_16384.pkl      ← ground-truth trajectories for training
```

## Basic setting for navsim workspace
```bash
# At navsim_workspace
touch setup_navsim.sh
```
* Copy below lines to setup_navsim.sh
```bash
#!/bin/bash

mkdir -p exp

export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="$PWD/dataset/maps"
export NAVSIM_EXP_ROOT="$PWD/exp"
export NAVSIM_DEVKIT_ROOT="$PWD/DRSI"
export OPENSCENE_DATA_ROOT="$PWD/dataset"
export NAVSIM_TRAJPDM_ROOT="$PWD/dataset/traj_pdm_v2"
```


## Download Pretrained Files

Download the pretrained checkpoint and the driving-command cluster file into the appropriate directories.

```bash
# At navsim_workspace
source setup_navsim.sh

# Pretrained DRSI checkpoint
mkdir -p exp/drsi_16384
wget -O exp/drsi_16384/drsi_vov.ckpt \
    https://huggingface.co/e2e-anon-2026/drsi/resolve/main/drsi_vov.ckpt

# Driving-command cluster labels (alternative to generating from scratch)
mkdir -p DRSI/traj_final
wget -O DRSI/traj_final/cluster_labels_16384.pkl \
    https://huggingface.co/e2e-anon-2026/drsi/resolve/main/cluster_labels_16384.pkl
```


## Making driving command_based_cluster.pkl
```bash
python driving_command_cluster_generator.py \
    --vocab_path traj_final/16384.npy \
    --log_dirs $OPENSCENE_DATA_ROOT/navsim_logs/trainval \
               $OPENSCENE_DATA_ROOT/navsim_logs/test \
    --output traj_final/cluster_labels_16384.pkl
```

## Training
```bash
# At navsim_workspace
source setup_navsim.sh
./DRSI/scripts/drsi/run_drsi_training.sh
```

## Inference - Calculating EPDMS

* Run evaluation (navtest EPDMS)
```bash
# At navsim_workspace
source setup_navsim.sh
./DRSI/scripts/drsi/run_drsi_pdm_score_evaulation.sh
```

* Run evaluation (navhard)
```bash
# At navsim_workspace
source setup_navsim.sh
./DRSI/scripts/drsi/run_drsi_pdm_score_evaulation_torchrun_navhard.sh
```

## Visualization (Single-GPU)

Before running, set `CKPT_PATH` directly in [visualization.py](visualization.py) (line 54):
```python
CKPT_PATH = "exp/drsi_16384/drsi_vov.ckpt"
```

Then run:
```bash
CUDA_VISIBLE_DEVICES=0 python visualization.py \
    agent=drsi_vov \
    train_test_split=navtrain \
    experiment_name=visualization \
    cache_path=null
```