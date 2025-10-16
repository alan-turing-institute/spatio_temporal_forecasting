# Autoregressive Models for Spatiotemporal Prediction (AR)

This module provides autoregressive wrappers for various spatial models (FNO, Diffusion) to handle temporal evolution in physics simulations, particularly for MHD and plasma physics applications.

## Overview

This repo implements a flexible framework for autoregressive temporal modeling with different spatial processing backends:

- **Fourier Neural Operator (FNO)**: Efficient spectral method for spatial processing
- **Diffusion Models**: Generative approach for spatial field prediction
- **Modular Design**: Easy to extend with new spatial models via `AutoregressiveBase`

## Architecture

```
AutoregressiveBase (Abstract)
├── AutoregressiveFNO (FNO backend)
└── AutoregressiveDiffusion (Diffusion backend)
```

### Key Components

- **`AutoregressiveBase`**: Abstract base class handling temporal logic
- **`Loss`**: Combined loss with relative Lp and temporal consistency terms
- **`MultivariableFNO`**: FNO implementation for multi-variable spatial fields
- **`SpatialDiffusionModel`**: Diffusion model for spatial field generation



## Quick Start

### Training an Autoregressive FNO

```python
import AutoregressiveFNO, MultivariableFNO, Loss
import torch

# Create base FNO model
fno = MultivariableFNO(
    n_vars=1,
    n_modes=(8, 8),
    hidden_channels=16,
    n_layers=4
)

# Wrap with autoregressive functionality
model = AutoregressiveFNO(
    fno_model=fno,
    t_in=10,          # Input timesteps
    t_out=5,          # Output timesteps to predict
    step_size=1,      # Single-step prediction
    teacher_forcing_ratio=0.5
)

# Create loss function
loss_fn = Loss(
    d=2,                    # Spatial dimension
    p=2,                    # L2 norm
    alpha_temporal=0.5,     # Temporal consistency weight
    learnable=False         # Fixed weights
)

# Training loop (simplified)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for batch in train_loader:
    x = batch["input_fields"]   # [B, t_in, n_vars, H, W]
    y = batch["output_fields"]  # [B, t_out, n_vars, H, W]
    
    pred = model(x, target=y, grid=grid)
    loss = loss_fn(pred, y)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

### Using the Command-Line Interface

```bash
# Basic training
python train_ar_fno.py \
    --data-path data/bout_ml_dataset_256.pt \
    --experiment-name my_experiment \
    --n-epochs 500 \
    --batch-size 16

# With custom hyperparameters
python train_ar_fno.py \
    --experiment-name advanced_experiment \
    --n-modes 12 12 \
    --hidden-channels 32 \
    --n-layers 6 \
    --t-in 10 \
    --t-out 5 \
    --alpha-temporal 0.3 \
    --lr 1e-3 \
    --grad-clip 1.0

# Resume training from checkpoint
python train_ar_fno.py \
    --experiment-name my_experiment \
    --resume

# With teacher forcing schedule
python train_ar_fno.py \
    --experiment-name tf_experiment \
    --tf-schedule "10:0.9,30:0.7,60:0.5,120:0.3"
```

## Configuration

### Model Parameters

- `--n-modes`: Number of Fourier modes (default: [8, 8])
- `--hidden-channels`: Hidden channels in FNO (default: 16)
- `--n-layers`: Number of FNO layers (default: 4)
- `--t-in`: Input timesteps (default: 10)
- `--t-out`: Output timesteps (default: 5)
- `--step-size`: Autoregressive step size (default: 1)

### Training Parameters

- `--n-epochs`: Number of epochs (default: 500)
- `--batch-size`: Batch size (default: 16)
- `--lr`: Learning rate (default: 1e-3)
- `--weight-decay`: Weight decay (default: 1e-5)
- `--grad-clip`: Gradient clipping (default: 1.0)

### Loss Parameters

- `--alpha-temporal`: Temporal consistency weight (default: 0.5)
- `--learnable-loss`: Use learnable loss weights (flag)

### Teacher Forcing

Teacher forcing helps stabilize training by occasionally using ground truth instead of predictions during autoregressive rollout.

- `--tf-initial`: Initial teacher forcing ratio (default: 1.0)
- `--tf-schedule`: Schedule as "epoch:ratio" pairs (default: "150:1.0,250:0.9,300:0.7,5000:0.5")

## Data Format

The module expects data in the following format:

```python
{
    'data': torch.Tensor,              # [n_trajectories, n_timesteps, H, W, n_vars]
    'constant_scalars': torch.Tensor,  # [n_trajectories, n_scalars]
    'constant_fields': torch.Tensor    # [H, W, n_fields]
}
```

### Example Data Loading

```python
from load_data import load_data, parse_args

args = parse_args()
train_loader, val_loader, full_data = load_data(args)
```

## Features

### 1. Flexible Autoregressive Framework

The `AutoregressiveBase` class provides:
- Temporal encoding via 3D convolutions
- Multi-step prediction support
- Teacher forcing during training
- Pure inference without teacher forcing

### 2. Advanced Loss Function

The `Loss` module combines:
- **Relative Lp Loss**: Normalizes by target magnitude
- **Temporal Consistency Loss**: Penalizes inconsistent temporal derivatives
- **Learnable Weights**: Optional automatic loss weight balancing

### 3. Checkpoint Management

Automatic checkpoint loading and saving:
- Saves best model based on validation loss
- Stores optimizer and scheduler states
- Preserves normalization statistics
- Tracks training history

### 4. Comprehensive Visualization

- Prediction GIFs for qualitative assessment
- Loss component tracking
- Long-term prediction rollout
- Error evolution plots

## Advanced Usage

### Custom Spatial Models

Extend `AutoregressiveBase` to use custom spatial models:

```python
import AutoregressiveBase

class AutoregressiveCustom(AutoregressiveBase):
    def __init__(self, custom_model, **kwargs):
        super().__init__(n_vars=custom_model.n_vars, **kwargs)
        self.custom_model = custom_model
    
    def _spatial_forward(self, x, grid=None):
        # Implement your spatial processing
        return self.custom_model(x, grid)
```

### Learnable Loss Weights

Enable automatic loss weight learning:

```python
loss_fn = Loss(
    d=2,
    p=2,
    alpha_temporal=0.5,
    learnable=True  # Enable learnable weights
)

# Include loss parameters in optimizer
optimizer = torch.optim.Adam(
    list(model.parameters()) + list(loss_fn.parameters()),
    lr=1e-3
)
```

### Multi-step Prediction

Configure the model for multi-step prediction:

```python
model = AutoregressiveFNO(
    fno_model=fno,
    t_in=10,
    t_out=20,      # Predict 20 steps
    step_size=5,   # Predict 5 steps at once
    teacher_forcing_ratio=0.5
)
```

## Output Structure

Training creates the following directory structure:

```
results/
└── experiment_name/
    ├── checkpoints/
    │   └── best_model.pth
    ├── visualizations/
    │   ├── predictions_epoch_*.png
    │   └── ...
    ├── gifs/
    │   ├── training_epoch_*.gif
    │   └── ...
    ├── plots/
    │   ├── loss_history.png
    │   └── loss_weights_history.png
    ├── long_term_predictions/
    │   ├── error_evolution.png
    │   └── prediction_*.gif
    └── config.txt
```

## Best Practices

1. **Start with teacher forcing**: Use high initial teacher forcing ratio (0.9-1.0)
2. **Gradually reduce**: Schedule teacher forcing reduction over training
3. **Monitor loss components**: Watch relative Lp vs temporal consistency
4. **Validate frequently**: Check validation performance every 10-20 epochs
5. **Use gradient clipping**: Prevent exploding gradients (clip at 1.0)
6. **Normalize data**: Compute and use dataset statistics
7. **Save checkpoints**: Enable automatic checkpoint saving

## Troubleshooting

### Training instability
- Reduce learning rate
- Increase gradient clipping
- Adjust teacher forcing schedule
- Check data normalization

### Poor long-term predictions
- Increase `t_in` (more context)
- Reduce `step_size` (finer predictions)
- Tune `alpha_temporal` (temporal consistency)
- Use teacher forcing longer

### Memory issues
- Reduce `batch_size`
- Reduce `hidden_channels`
- Reduce `n_layers`
- Use gradient checkpointing

