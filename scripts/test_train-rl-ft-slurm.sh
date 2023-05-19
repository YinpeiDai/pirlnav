#!/bin/bash
#SBATCH --job-name=pirlnav-ddppo
#SBATCH --account=chaijy1
#SBATCH --partition=gpu
#SBATCH --time=00:30:00                 
#SBATCH --nodes=2 
#SBATCH --gpus-per-node=v100:2                       
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=8               
#SBATCH --mem-per-gpu=9000m                  
#SBATCH --output=/home/%u/pirlnav/slurm_logs/%x-%j.log         
#SBATCH --mail-user=daiyp@umich.edu
#SBATCH --mail-type=BEGIN,END

module load python3.9-anaconda/2021.11
source /home/daiyp/.bashrc
conda activate pirlnav

echo "NODELIST="${SLURM_NODELIST}
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MAIN_ADDR=$master_addr
echo "MAIN_ADDR="$MAIN_ADDR
/bin/hostname

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
