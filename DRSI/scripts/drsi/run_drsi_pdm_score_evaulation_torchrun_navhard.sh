export PROGRESS_MODE="eval"
split=navhard
agent=drsi_vov
dir=drsi_16384
NPROC_PER_NODE=3
metric_cache_path="${NAVSIM_EXP_ROOT}/${split}_two_stage_metric_cache"
cd ${NAVSIM_DEVKIT_ROOT}

for epoch in 19; do
    padded_epoch=$(printf "%02d" $epoch)
    experiment_name="${dir}/test-${padded_epoch}ep-${split}-random"
    ckpt=${NAVSIM_EXP_ROOT}/${dir}/drsi_vov.ckpt # this can also be the checkpoint we provided
    
    export DP_PREDS=None
    export SUBSCORE_PATH=${NAVSIM_EXP_ROOT}/${dir}/epoch${epoch}_${split}.pkl; # save path for the scores
    
    torchrun --nproc_per_node=${NPROC_PER_NODE} --master_port=29900 \
        ${NAVSIM_DEVKIT_ROOT}/navsim/planning/script/run_pdm_score_gpu_v2.py \
        agent=$agent \
        +combined_inference=false \
        dataloader.params.batch_size=1 \
        agent.checkpoint_path=${ckpt} \
        agent.config.vocab_path=${NAVSIM_DEVKIT_ROOT}/traj_final/8192.npy \
        agent.config.vocab_cluster_path=${NAVSIM_DEVKIT_ROOT}/traj_final/cluster_labels_8192.pkl \
        agent.config.pruning=True \
        trainer.params.precision=32 \
        experiment_name=${experiment_name} \
        +cache_path=null \
        metric_cache_path=${metric_cache_path} \
        train_test_split=${split}_two_stage
done