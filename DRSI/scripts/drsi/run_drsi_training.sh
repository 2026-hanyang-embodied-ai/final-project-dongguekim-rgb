NUM_NODES=1
MASTER_ADDR=127.0.0.1 # your master node ip, which can be set to 127.0.0.1 for single-node training
NODE_RANK=0 # 0 for the master node, 1 and 2 for other sub-nodes
config="default_training" # this config uses the entire navtrain dataset for training
experiment_name=drsi_16384 # this could also be train_hydra_mdp
agent=drsi_vov # the agent could also be hydra_mdp_vov

# training hyper-parameters
lr=0.0002
bs=8
max_epochs=20
num_workers=2

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_training_dense.py \
    --config-name ${config} \
    trainer.params.num_nodes=${NUM_NODES} \
    agent=${agent} \
    experiment_name=${experiment_name} \
    train_test_split=navtrain \
    dataloader.params.batch_size=${bs} \
    dataloader.params.num_workers=${num_workers} \
    trainer.params.precision=32 \
    trainer.params.max_epochs=${max_epochs} \
    ~trainer.params.strategy \
    agent.config.ckpt_path=${experiment_name} \
    agent.lr=${lr} \
    cache_path=null \