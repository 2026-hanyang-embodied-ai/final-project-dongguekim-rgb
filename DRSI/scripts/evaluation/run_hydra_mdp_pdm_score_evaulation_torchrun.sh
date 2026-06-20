export PROGRESS_MODE="eval"
NUM_NODES=1
NPROC_PER_NODE=1
split=navtest
agent=hydra_mdp_vov
experiment_name=hydra_mdp_vov
output_dir=${NAVSIM_EXP_ROOT}/${experiment_name}
metric_cache_path="${NAVSIM_EXP_ROOT}/${split}_metric_cache"
ckpt="${output_dir}/hydra_mdp_vov.ckpt"

echo ""
echo "Running PDMS Evaluation (GPU one_stage)..."
echo "=============================================="

cd ${output_dir}
for file in epoch=*-step=*.ckpt; do
    epoch=$(echo $file | sed -n 's/.*epoch=\([0-9][0-9]\).*/\1/p')
    new_filename="epoch${epoch}.ckpt"
    mv "$file" "$new_filename"
done

cd ${NAVSIM_DEVKIT_ROOT}

torchrun --nproc_per_node=${NPROC_PER_NODE} --master_port=29600 \
    ${NAVSIM_DEVKIT_ROOT}/navsim/planning/script/run_pdm_score_gpu_one_stage.py \
    agent=${agent} \
    agent.checkpoint_path=${ckpt} \
    agent.config.vocab_path=${NAVSIM_DEVKIT_ROOT}/traj_final/16384.npy \
    train_test_split=${split} \
    metric_cache_path=${metric_cache_path} \
    dataloader.params.batch_size=1 \
    dataloader.params.num_workers=1 \
    trainer.params.precision=32 \
    trainer.params.num_nodes=${NUM_NODES} \
    +trainer.params.devices=${NPROC_PER_NODE} \
    output_dir=${output_dir} \
    experiment_name=${experiment_name} \
    verbose=true

echo ""
echo "=============================================="
echo "Evaluation Complete!"
echo "Results CSV: ${output_dir}/"
echo "  → PDMS score: Check the 'score' column where token=='average_all_frames' in the results CSV"
echo "=============================================="