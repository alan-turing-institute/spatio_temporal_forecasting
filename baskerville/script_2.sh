#!/bin/bash
#SBATCH --account=vjgo8416-ai-phy-sys 
#SBATCH --qos=turing
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=36
#SBATCH --time 12:30:00

#Load required modules

module purge
module load baskerville

module load bask-apps/live
module load Miniforge3/24.1.2-0

eval "$(${EBROOTMINIFORGE3}/bin/conda shell.bash hook)"
source "${EBROOTMINIFORGE3}/etc/profile.d/mamba.sh"

mamba activate  /bask/projects/v/vjgo8416-ai-phy-sys/marj/hdjd5168_conda_ae_exploratory 


python /bask/projects/v/vjgo8416-ai-phy-sys/marj/research/AR/train_ar_fno.py --experiment-name "learnable_init_1.0" --learnable-loss  --alpha-temporal 1.0 --n-epochs 500 --batch-size 16  --lr 1e-3  --viz-freq 50 --resume
python /bask/projects/v/vjgo8416-ai-phy-sys/marj/research/AR/train_ar_fno.py --experiment-name "learnable_init_0.2" --learnable-loss  --alpha-temporal 0.2 --n-epochs 500 --batch-size 16  --lr 1e-3  --viz-freq 50 --resume






