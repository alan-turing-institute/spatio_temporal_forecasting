"""Data loading and preprocessing utilities."""

import argparse
from typing import Tuple, List

import numpy as np
import torch
from torch.utils.data import DataLoader
from autoemulate.experimental.data.spatiotemporal_dataset import AutoEmulateDataset

from spatio_temporal_forecasting.constants import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_LEARNING_RATE,
    DEFAULT_WEIGHT_DECAY,
    DEFAULT_GRAD_CLIP,
    DEFAULT_T_IN,
    DEFAULT_T_OUT,
    DEFAULT_STEP_SIZE,
    DEFAULT_TEACHER_FORCING_INITIAL,
    DEFAULT_ALPHA_TEMPORAL,
    DEFAULT_N_MODES,
    DEFAULT_HIDDEN_CHANNELS,
    DEFAULT_N_LAYERS,
    DEFAULT_SCHEDULER_FACTOR,
    DEFAULT_SCHEDULER_PATIENCE,
    DEFAULT_VIZ_FREQ,
    DEFAULT_N_VIZ_SAMPLES,
    DEFAULT_N_LONG_TERM_SAMPLES,
    TRAIN_VAL_SPLIT,
    EPSILON,
)


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.
    
    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(description='Train Autoregressive FNO')
    
    # Data parameters
    parser.add_argument('--data-path', type=str, default='data/bout_ml_dataset_256.pt',
                        help='Path to dataset')
    parser.add_argument('--target-dir', type=str, 
                        default='/bask/projects/v/vjgo8416-ai-phy-sys/marj/research/AR',
                        help='Working directory')
    
    # Model parameters
    parser.add_argument('--n-modes', type=int, nargs=2, default=DEFAULT_N_MODES,
                        help='Number of Fourier modes')
    parser.add_argument('--hidden-channels', type=int, default=DEFAULT_HIDDEN_CHANNELS,
                        help='Hidden channels in FNO')
    parser.add_argument('--n-layers', type=int, default=DEFAULT_N_LAYERS,
                        help='Number of FNO layers')
    parser.add_argument('--t-in', type=int, default=DEFAULT_T_IN,
                        help='Input timesteps')
    parser.add_argument('--t-out', type=int, default=DEFAULT_T_OUT,
                        help='Output timesteps')
    parser.add_argument('--step-size', type=int, default=DEFAULT_STEP_SIZE,
                        help='Autoregressive step size')
    
    # Training parameters
    parser.add_argument('--n-epochs', type=int, default=500,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=DEFAULT_BATCH_SIZE,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=DEFAULT_LEARNING_RATE,
                        help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=DEFAULT_WEIGHT_DECAY,
                        help='Weight decay')
    parser.add_argument('--num-workers', type=int, default=8,
                        help='Number of data loader workers')
    
    # Loss parameters
    parser.add_argument('--alpha-temporal', type=float, default=DEFAULT_ALPHA_TEMPORAL,
                        help='Temporal consistency weight')
    parser.add_argument('--learnable-loss', action='store_true', default=False,
                        help='Use learnable loss weights')
    
    # Teacher forcing schedule
    parser.add_argument('--tf-initial', type=float, default=DEFAULT_TEACHER_FORCING_INITIAL,
                        help='Initial teacher forcing ratio')
    parser.add_argument('--tf-schedule', type=str, default='150:1.0,250:0.9,300:0.7,5000:0.5',
                        help='Teacher forcing schedule as epoch:ratio pairs (comma-separated)')
    
    # Optimization parameters
    parser.add_argument('--scheduler-factor', type=float, default=DEFAULT_SCHEDULER_FACTOR,
                        help='LR scheduler reduction factor')
    parser.add_argument('--scheduler-patience', type=int, default=DEFAULT_SCHEDULER_PATIENCE,
                        help='LR scheduler patience')
    parser.add_argument('--grad-clip', type=float, default=DEFAULT_GRAD_CLIP,
                        help='Gradient clipping max norm')
    
    # Visualization parameters
    parser.add_argument('--viz-freq', type=int, default=DEFAULT_VIZ_FREQ,
                        help='Visualization frequency (epochs)')
    parser.add_argument('--n-viz-samples', type=int, default=DEFAULT_N_VIZ_SAMPLES,
                        help='Number of samples for visualization')
    
    # Misc
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--output-dir', type=str, default='./results',
                        help='Main directory for all outputs and results')
    parser.add_argument('--experiment-name', type=str, default=None,
                        help='Name of experiment folder (auto-generated if not provided)')
    parser.add_argument('--checkpoint-name', type=str, default='best_model.pth',
                        help='Filename for model checkpoint')
    
    # Long-term prediction parameters
    parser.add_argument('--n-long-term-samples', type=int, default=DEFAULT_N_LONG_TERM_SAMPLES,
                        help='Number of samples for long-term prediction visualization')
    parser.add_argument('--skip-long-term', action='store_true',
                        help='Skip long-term predictions (faster completion)')
    parser.add_argument('--resume', action='store_true',
                        help='Resume training from last checkpoint if available')
    parser.add_argument('--checkpoint-path', type=str, default=None,
                        help='Specific checkpoint path to resume from (overrides auto-detection)')
    
    return parser.parse_args()


def set_seed(seed: int) -> None:
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def parse_tf_schedule(schedule_str: str) -> List[Tuple[int, float]]:
    """
    Parse teacher forcing schedule from string.
    
    Args:
        schedule_str: String like '150:1.0,250:0.9,300:0.7'
    
    Returns:
        Sorted list of (epoch, ratio) tuples
        
    Raises:
        ValueError: If schedule string format is invalid
    """
    try:
        pairs = schedule_str.split(',')
        schedule = []
        for pair in pairs:
            epoch_str, ratio_str = pair.split(':')
            schedule.append((int(epoch_str), float(ratio_str)))
        return sorted(schedule)
    except (ValueError, AttributeError) as e:
        raise ValueError(
            f"Invalid teacher forcing schedule format: '{schedule_str}'. "
            f"Expected format: 'epoch:ratio,epoch:ratio,...'"
        ) from e


def load_data(args: argparse.Namespace) -> Tuple[DataLoader, DataLoader, dict]:
    """
    Load and split dataset into train and validation sets.
    
    Args:
        args: Arguments namespace with data_path, t_in, t_out, batch_size, num_workers
        
    Returns:
        Tuple of (train_loader, val_loader, full_data)
    """
    print(f"Loading data from {args.data_path}...")
    data = torch.load(args.data_path)
    
    print(f"Data shape: {data['data'].shape}")
    print(f"Constant scalars shape: {data['constant_scalars'].shape}")
    
    # Split at trajectory level
    n_trajectories = data['data'].shape[0]
    train_traj_count = int(TRAIN_VAL_SPLIT * n_trajectories)
    
    print(f"Train trajectories: {train_traj_count}, Val trajectories: {n_trajectories - train_traj_count}")
    
    # Create train/val data splits
    train_data = {
        'data': data["data"][:train_traj_count],
        'constant_scalars': data["constant_scalars"][:train_traj_count],
        'constant_fields': data["constant_fields"]
    }
    
    val_data = {
        'data': data["data"][train_traj_count:],
        'constant_scalars': data["constant_scalars"][train_traj_count:],
        'constant_fields': data["constant_fields"]
    }
    
    # Create datasets
    train_dataset = AutoEmulateDataset(
        data_path=None,
        data=train_data,
        n_steps_input=args.t_in,
        n_steps_output=args.t_out,
        dtype=torch.float32
    )
    
    val_dataset = AutoEmulateDataset(
        data_path=None,
        data=val_data,
        n_steps_input=args.t_in,
        n_steps_output=args.t_out,
        dtype=torch.float32
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False
    )
    
    return train_loader, val_loader, data


def compute_normalization_stats(
    train_loader: DataLoader,
    device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute mean and std for data normalization.
    
    Args:
        train_loader: Training data loader
        device: Device to place tensors on
        
    Returns:
        Tuple of (data_mean, data_std) tensors
    """
    print("Computing normalization statistics...")
    all_x = []
    all_y = []
    
    for batch in train_loader:
        x_batch = batch["input_fields"]
        y_batch = batch["output_fields"]
        all_x.append(x_batch)
        all_y.append(y_batch)
    
    all_x = torch.cat(all_x, dim=0)
    all_y = torch.cat(all_y, dim=0)
    
    x_mean = all_x.mean()
    x_std = all_x.std()
    y_mean = all_y.mean()
    y_std = all_y.std()
    
    # Use same normalization for consistency
    data_mean = ((x_mean + y_mean) / 2).to(device)
    data_std = ((x_std + y_std) / 2).to(device)
    
    # Prevent division by zero
    data_std = torch.clamp(data_std, min=EPSILON)
    
    print(f"Normalization - Mean: {data_mean:.6f}, Std: {data_std:.6f}")
    
    return data_mean, data_std

