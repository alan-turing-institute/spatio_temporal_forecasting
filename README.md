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

---

## Loss Functions

<details>
<summary><b>📐 Mathematical Formulation (Click to expand)</b></summary>

The loss module combines two key components for spatiotemporal prediction accuracy and physical consistency.

### Total Loss

$$\mathcal{L}_{\text{total}} = w_1 \cdot \mathcal{L}_{\text{relative}} + w_2 \cdot \mathcal{L}_{\text{temporal}}$$

### 1. Relative $L^p$ Loss

Measures the relative error between predictions and targets in $L^p$ norm:

$$\mathcal{L}_{\text{relative}} = \frac{1}{B} \sum_{i=1}^{B} \frac{\|\text{pred}_i - \text{target}_i\|_p}{\|\text{target}_i\|_p + \epsilon}$$

**Where:**
- $\text{pred}$: Predicted tensor $[B, T, n_{\text{vars}}, H, W]$ or $[B, \ldots]$
- $\text{target}$: Ground truth tensor (same shape as pred)
- $p$: Order of $L^p$ norm (default: 2 for $L^2$/Euclidean norm)
- $\epsilon$: Small constant to prevent division by zero
- $\|\cdot\|_p$: $L^p$ norm computed over all dimensions except batch
- $B$: Batch size

**Properties:**
- Scale-invariant (relative error)
- Normalized by target magnitude
- Averaged over batch dimension

### 2. Temporal Consistency Loss

Penalizes inconsistent temporal derivatives between consecutive timesteps:

$$\mathcal{L}_{\text{temporal}} = \frac{1}{B(T-1)} \sum_{i=1}^{B} \sum_{t=1}^{T-1} \frac{\|\Delta\text{pred}_{i,t} - \Delta\text{target}_{i,t}\|_p}{\|\Delta\text{target}_{i,t}\|_p + \epsilon}$$

**Where:**
- $\Delta\text{pred}_{i,t} = \text{pred}_{i,t+1} - \text{pred}_{i,t}$ (temporal finite differences)
- $\Delta\text{target}_{i,t} = \text{target}_{i,t+1} - \text{target}_{i,t}$
- $T$: Number of timesteps
- Returns 0 if only 1 timestep present

**Properties:**
- Enforces smooth temporal evolution
- Helps maintain physical consistency over time
- Improves long-term rollout stability

### 3. Loss Weights

Two modes for weight configuration:

#### Fixed Weights (default: `learnable=False`)

$$w_1 = 1.0 \quad \text{(fixed)}$$
$$w_2 = \alpha_{\text{temporal}} \quad \text{(default: 0.5, user-specified)}$$

#### Learnable Weights (`learnable=True`)

$$w_1 = \exp(\theta_1)$$
$$w_2 = \exp(\theta_2)$$

- $\theta_1, \theta_2$: Learnable parameters initialized as $[1.0, \alpha_{\text{temporal}}]$
- Uses exponential to ensure positive weights
- Allows automatic balance between loss components during training

</details>

<details>
<summary><b>💻 Usage Examples (Click to expand)</b></summary>

### Basic Usage

```python
from loss import Loss

# Create loss function
criterion = Loss(
    d=2,                    # Spatial dimension
    p=2,                    # L2 norm
    alpha_temporal=0.5,     # Weight for temporal loss
    learnable=False         # Fixed weights
)

# During training
pred = model(input)         # [B, T, n_vars, H, W]
target = ...                # [B, T, n_vars, H, W]

loss = criterion(pred, target)
loss.backward()
```

### With Learnable Weights

```python
# Create loss with learnable weights
criterion = Loss(
    d=2,
    p=2,
    alpha_temporal=0.5,
    learnable=True
)

# Add loss parameters to optimizer
optimizer = torch.optim.Adam([
    {'params': model.parameters()},
    {'params': criterion.parameters(), 'lr': 0.01}
])
```

### Monitoring Loss Components

```python
# Get individual loss components for logging
components = criterion.get_loss_components(pred, target)

print(f"Relative Lp Loss: {components['relative_lp']:.6f}")
print(f"Temporal Loss: {components['temporal_consistency']:.6f}")
print(f"Weight (Relative): {components['weight_rel']:.6f}")
print(f"Weight (Temporal): {components['weight_temp']:.6f}")
```

</details>

<details>
<summary><b>📊 Visualization Tools (Click to expand)</b></summary>

The module includes comprehensive visualization functions for tracking training progress.

### 1. Simple Loss Plot

```python
from loss import plot_loss

plot_loss(train_losses, val_losses)
# Saves to: loss_history.png
```

### 2. Detailed Loss Plot with Components

```python
from loss import plot_loss_with_components

plot_loss_with_components(
    train_losses=train_losses,
    val_losses=val_losses,
    train_components=train_components,
    val_components=val_components,
    save_path='./results'
)
# Saves to: ./results/loss_history.png
```

**Includes 6 subplots:**
1. Total loss (train & val) with best model marker
2. Relative Lp loss component
3. Temporal consistency component
4. Ratio of temporal/relative loss
5. Log-scale total loss
6. Stacked area plot showing component contributions

### 3. Loss Weights Evolution

```python
from loss import plot_loss_weights_evolution

plot_loss_weights_evolution(
    loss_weights_history=weights_history,
    save_path='./results'
)
# Saves to: ./results/loss_weights_history.png
```

</details>

<details>
<summary><b>⚙️ Parameters & Configuration (Click to expand)</b></summary>

### Loss Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `d` | int | 2 | Spatial dimension |
| `p` | int | 2 | Order of Lp norm (2 = L2/Euclidean norm) |
| `alpha_temporal` | float | 0.5 | Weight for temporal consistency loss |
| `learnable` | bool | False | Whether loss weights are learnable parameters |

### Input Tensor Shapes

The loss functions accept flexible input shapes:

- **Spatiotemporal**: `[B, T, n_vars, H, W]` - Full spatiotemporal data
- **Spatial only**: `[B, n_vars, H, W]` - Single timestep (temporal loss = 0)
- **Generic**: `[B, ...]` - Any shape with batch dimension first

</details>

<details>
<summary><b>💡 Tips & Best Practices (Click to expand)</b></summary>

### Design Considerations

**Why Relative Loss?**
- Provides scale-invariance across variables with different magnitudes
- Essential for multi-variable predictions (e.g., temperature, pressure, velocity)
- Enables fair comparison across different physical quantities

**Why Temporal Consistency?**
- Prevents unphysical temporal oscillations
- Enforces smooth evolution (important for PDEs)
- Improves long-term rollout stability
- Maintains physical realism in predictions

### Hyperparameter Tuning

1. **Starting `alpha_temporal`**: Begin with 0.5, adjust based on validation performance
2. **Learnable weights**: Use smaller learning rate (0.01-0.001) for loss parameters
3. **Long rollouts**: Increase `alpha_temporal` for longer prediction horizons
4. **Unstable training**: If temporal loss dominates, reduce `alpha_temporal`

### Common Issues

| Issue | Possible Cause | Solution |
|-------|----------------|----------|
| Temporal loss → 0 | Weight too small | Increase `alpha_temporal` |
| Unstable predictions | Temporal loss too dominant | Decrease `alpha_temporal` |
| Oscillating predictions | Insufficient temporal regularization | Increase `alpha_temporal` |

</details>

---

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
