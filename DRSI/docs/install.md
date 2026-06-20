# Download and installation


### 1. Clone the devkit

Clone the repository

```bash
git clone https://github.com/NVlabs/GTRS.git
cd GTRS
```

### 2. Download the dataset

You need to download the OpenScene logs and sensor blobs, as well as the nuPlan maps.
We provide scripts to download the nuplan maps, the mini split and the test split.
Navigate to the download directory and download the maps

**NOTE: Please check the [LICENSE file](https://motional-nuplan.s3-ap-northeast-1.amazonaws.com/LICENSE) before downloading the data.**

```bash
cd download && ./download_maps
```

Next download the data splits you want to use.
Note that the dataset splits do not exactly map to the recommended standardized training / test splits-
Please refer to [splits](splits.md) for an overview on the standardized training and test splits including their size and check which dataset splits you need to download in order to be able to run them.
You can download these splits with the following scripts.

```bash
./download_mini
./download_trainval
./download_test
./download_warmup_two_stage
./download_navhard_two_stage
./download_private_test_hard_two_stage
```

Also, the script `./download_navtrain` can be used to download a small portion of the  `trainval` dataset split which is needed for the `navtrain` training split.

Execute the following script to download the simulated ground-truths for different vocabularies:
```bash
cd ~/navsim_workspace/dataset;
mkdir traj_pdm_v2; cd traj_pdm_v2;
# ground-truths without data augmentations
mkdir ori; cd ori;
wget https://huggingface.co/Zzxxxxxxxx/gtrs/resolve/main/navtrain_8192.pkl
wget https://huggingface.co/Zzxxxxxxxx/gtrs/resolve/main/navtrain_16384.pkl
# ground-truths with data augmentations
cd ../; mkdir random_aug; cd random_aug;
wget https://huggingface.co/Zzxxxxxxxx/gtrs/resolve/main/rot_30-trans_0-va_0-p_0.5-ensemble.json
wget https://huggingface.co/Zzxxxxxxxx/gtrs/resolve/main/aug_traj_pdm.zip
unzip aug_traj_pdm.zip
rm aug_traj_pdm.zip
```

Execute the following script to download the pretrained vision backbones:
```bash
cd ~/navsim_workspace/dataset;
mkdir models; cd models;
wget https://huggingface.co/Zzxxxxxxxx/gtrs/resolve/main/dd3d_det_final.pth
```

### 3. Install the navsim-devkit

Finally, install navsim.
To this end, create a new environment and install the required dependencies:

```bash
conda env create --name conda_gtrs -f environment.yml
conda activate conda_gtrs
pip install --upgrade diffusers[torch]
pip install -e .
```