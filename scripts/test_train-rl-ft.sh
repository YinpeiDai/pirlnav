#!/bin/bash



export GLOG_minloglevel=2
export MAGNUM_LOG=quiet
export HABITAT_SIM_LOG=quiet

config="configs/experiments/rl_ft_objectnav.yaml"

DATA_PATH="data/datasets/objectnav_hm3d_v2"
TENSORBOARD_DIR="tb/objectnav_il_rl_ft/seed_1/"
CHECKPOINT_DIR="/scratch/chaijy_root/chaijy1/daiyp/data/pirlnav_tmp/objectnav_il_rl_ft_ckpt/seed_1/"
PRETRAINED_WEIGHTS="ckpts/objectnav_rl_ft_hd.ckpt"

mkdir -p $TENSORBOARD_DIR
mkdir -p $CHECKPOINT_DIR

echo "In ObjectNav RL DDPPO"

python -u -m torch.distributed.launch \
    --use_env \
    --nproc_per_node 2 \
    run.py \
    --exp-config $config \
    --run-type train \
    TENSORBOARD_DIR $TENSORBOARD_DIR \
    CHECKPOINT_FOLDER $CHECKPOINT_DIR \
    NUM_UPDATES 20000 \
    NUM_ENVIRONMENTS 2 \
    RL.DDPPO.pretrained_weights $PRETRAINED_WEIGHTS \
    RL.DDPPO.pretrained True \
    TASK_CONFIG.DATASET.DATA_PATH "$DATA_PATH/{split}/{split}.json.gz" \
    VERBOSE False