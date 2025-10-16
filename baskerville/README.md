# Baskerville HPC Submission Scripts

This folder contains example SLURM submission scripts for running the spatio-temporal forecasting training jobs on the Baskerville HPC cluster.

## Overview

The scripts demonstrate how to:
- Configure SLURM job parameters for GPU compute
- Set up the Python environment with conda/mamba
- Run training experiments with different hyperparameters
- Perform parameter sweeps by running multiple experiments in sequence

## Available Scripts

### `script.sh`
Basic parameter sweep example that varies the temporal consistency weight (`--alpha-temporal`):
- Experiment 1: `alpha-temporal=0.5`
- Experiment 2: `alpha-temporal=1.0`

Both experiments use fixed loss weights and run for 500 epochs.

### `script_2.sh`
Parameter sweep with learnable loss weights enabled (`--learnable-loss`):
- Experiment 1: `alpha-temporal=1.0` with learnable loss
- Experiment 2: `alpha-temporal=0.2` with learnable loss

This demonstrates how to enable adaptive loss weight learning during training.

### `tf_script.sh`
Advanced example with teacher forcing scheduling:
- Uses a custom teacher forcing schedule: `"10:0.9,30:0.7,60:0.5,120:0.3"`
- Includes `--skip-long-term` flag to skip long-term prediction evaluation
- Shows how to configure early teacher forcing decay for faster training

## SLURM Configuration

All scripts use the following SLURM parameters:
```bash
#SBATCH --account=vjgo8416-ai-phy-sys
#SBATCH --qos=turing
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=36
#SBATCH --time 12:30:00  # (5:30:00 for tf_script.sh)
```

Adjust these parameters according to your account details and resource requirements.

## Environment Setup

The scripts demonstrate the standard Baskerville environment setup:
1. Load the base Baskerville module
2. Load Miniforge3 for conda/mamba package management
3. Activate your conda environment
4. Run Python training scripts

## Command-Line Parameter Sweeping

One of the key features demonstrated in these scripts is how easy it is to perform parameter sweeps by varying command-line arguments. The training script (`train_ar_fno.py`) accepts numerous hyperparameters via the command line, making it simple to explore different configurations without modifying code.

### Key Parameters for Sweeping

**Model Architecture:**
- `--n-modes`: Number of Fourier modes (e.g., `12 12`)
- `--hidden-channels`: Hidden channels in FNO (e.g., `16`, `32`)
- `--n-layers`: Number of FNO layers (e.g., `4`, `6`)
- `--t-in`: Input timesteps (e.g., `10`, `15`)
- `--t-out`: Output timesteps (e.g., `5`, `10`)

**Training Configuration:**
- `--n-epochs`: Number of training epochs (e.g., `500`, `1000`)
- `--batch-size`: Batch size (e.g., `8`, `16`, `32`)
- `--lr`: Learning rate (e.g., `1e-3`, `5e-4`)
- `--weight-decay`: Weight decay for regularization (e.g., `1e-5`)
- `--grad-clip`: Gradient clipping value (e.g., `1.0`)

**Loss Function:**
- `--alpha-temporal`: Temporal consistency weight (e.g., `0.2`, `0.5`, `1.0`)
- `--learnable-loss`: Enable learnable loss weights (flag)

**Teacher Forcing:**
- `--tf-initial`: Initial teacher forcing ratio (e.g., `1.0`, `0.9`)
- `--tf-schedule`: Schedule as "epoch:ratio" pairs (e.g., `"10:0.9,30:0.7,60:0.5"`)

**Experiment Management:**
- `--experiment-name`: Name for the experiment (creates separate output folders)
- `--resume`: Resume training from checkpoint (flag)
- `--viz-freq`: Visualization frequency in epochs (e.g., `20`, `50`)
- `--skip-long-term`: Skip long-term prediction evaluation (flag)

### Example Parameter Sweep

To sweep over multiple hyperparameter values, simply run the training script multiple times with different arguments:

```bash
# Sweep over temporal consistency weights
python train_ar_fno.py --experiment-name "alpha_0.2" --alpha-temporal 0.2 --n-epochs 500
python train_ar_fno.py --experiment-name "alpha_0.5" --alpha-temporal 0.5 --n-epochs 500
python train_ar_fno.py --experiment-name "alpha_1.0" --alpha-temporal 1.0 --n-epochs 500

# Sweep over learning rates
python train_ar_fno.py --experiment-name "lr_1e-3" --lr 1e-3 --n-epochs 500
python train_ar_fno.py --experiment-name "lr_5e-4" --lr 5e-4 --n-epochs 500
python train_ar_fno.py --experiment-name "lr_1e-4" --lr 1e-4 --n-epochs 500

# Sweep over model sizes
python train_ar_fno.py --experiment-name "small" --hidden-channels 16 --n-layers 4
python train_ar_fno.py --experiment-name "medium" --hidden-channels 32 --n-layers 6
python train_ar_fno.py --experiment-name "large" --hidden-channels 64 --n-layers 8
```

## Usage

1. **Customize the script**: Update the account, paths, and environment to match your setup
2. **Choose parameters**: Modify the command-line arguments to suit your experiment
3. **Submit the job**: 
   ```bash
   sbatch script.sh
   ```
4. **Monitor progress**: Check job status with `squeue` and output logs in your working directory

## Output

Each experiment creates a separate directory under `results/` with the specified `--experiment-name`. The directory contains:
- Trained model checkpoints
- Visualization plots and GIFs
- Training history and logs
- Configuration details

## Tips

- Use descriptive experiment names that reflect the hyperparameters being tested
- The `--resume` flag allows continuing interrupted training runs
- Multiple experiments can be chained in a single script for automated sweeps
- Adjust SLURM time limits based on your dataset size and number of epochs
- Monitor GPU memory usage and adjust `--batch-size` if needed

## For More Information

See the main [README.md](../README.md) for comprehensive documentation on:
- Model architecture details
- Complete list of command-line parameters
- Training best practices
- Advanced usage examples
