#!/bin/bash
#SBATCH --job-name=pirlnav-ddppo
#SBATCH --account=chaijy1
#SBATCH --partition=gpu
#SBATCH --time=00:30:00                 
#SBATCH --nodes=1                           
#SBATCH --cpus-per-task=1                     
#SBATCH --mem-per-gpu=14000m
#SBATCH --gpus-per-node=2
#SBATCH --ntasks-per-gpu=3                 
#SBATCH --output=/home/%u/pirlnav/slurm_logs/%x-%j.log         
#SBATCH --mail-user=daiyp@umich.edu
#SBATCH --mail-type=BEGIN,END

module load python3.9-anaconda/2021.11
source /home/daiyp/.bashrc
conda activate pirlnav

MAIN_ADDR=$(srun --ntasks=1 hostname 2>&1 | tail -n1)
export MAIN_ADDR
echo $MAIN_ADDR
echo $MAIN_PORT

echo $SLURM_LOCALID
echo $SLURM_PROCID
echo $SLURM_NTASKS

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
set -x
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
    
# ln -s /scratch/chaijy_root/chaijy1/daiyp/data/habitat_data/datasets  data/datasets  
# ln -s /scratch/chaijy_root/chaijy1/daiyp/data/habitat_data/scene_datasets  data/scene_datasets
