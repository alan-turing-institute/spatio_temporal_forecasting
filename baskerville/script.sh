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


python /bask/projects/v/vjgo8416-ai-phy-sys/marj/research/AR/train_ar_fno.py --experiment-name "fixed_temp_0.5" --alpha-temporal 0.5 --n-epochs 500 --batch-size 16  --lr 1e-3  --viz-freq 50 --resume
python /bask/projects/v/vjgo8416-ai-phy-sys/marj/research/AR/train_ar_fno.py --experiment-name "fixed_temp_1.0" --alpha-temporal 1.0 --n-epochs 500 --batch-size 16  --lr 1e-3  --viz-freq 50 --resume






