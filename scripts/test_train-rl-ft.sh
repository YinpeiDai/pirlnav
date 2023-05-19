#!/bin/bash
#SBATCH --job-name=pirlnav-ddppo-1gpu-test
#SBATCH --account=chaijy1
#SBATCH --partition=gpu
#SBATCH --time=02:00:00                 
#SBATCH --nodes=1                           
#SBATCH --cpus-per-task=1                     
#SBATCH --mem-per-gpu=14000m
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-gpu=3                 
#SBATCH --output=/home/%u/pirlnav/slurm_logs/%x-%j.log         
#SBATCH --mail-user=daiyp@umich.edu
#SBATCH --mail-type=BEGIN,END


export GLOG_minloglevel=2
export MAGNUM_LOG=quiet
export HABITAT_SIM_LOG=quiet

config="configs/experiments/rl_ft_objectnav.yaml"
DATA_PATH="data/datasets/objectnav_hm3d_v2"
TENSORBOARD_DIR="tb/objectnav_il_rl_ft/test/seed_1/"
CHECKPOINT_DIR="data/new_checkpoints/objectnav_il_rl_ft_ckpt/test/seed_1/"
PRETRAINED_WEIGHTS="ckpts/objectnav_rl_ft_hd.ckpt" # objectnav_rl_ft_hd

mkdir -p $TENSORBOARD_DIR
mkdir -p $CHECKPOINT_DIR

echo "In ObjectNav RL DDPPO"

srun python -u -m run \
    --exp-config $config \
    --run-type train \
    TENSORBOARD_DIR $TENSORBOARD_DIR \
    CHECKPOINT_FOLDER $CHECKPOINT_DIR \
    NUM_UPDATES 10000 \
    NUM_ENVIRONMENTS 2 \
    RL.DDPPO.pretrained_weights $PRETRAINED_WEIGHTS \
    RL.DDPPO.pretrained True \
    TASK_CONFIG.DATASET.DATA_PATH "$DATA_PATH/{split}/{split}.json.gz" \
    VERBOSE False