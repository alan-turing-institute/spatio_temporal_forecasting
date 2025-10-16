"""Checkpoint loading and management utilities."""
from typing import Dict, Any
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from constants import CHECKPOINT_EXTENSION, CHECKPOINT_EXTENSION_ALT, CHECKPOINT_KEYS


def load_last_checkpoint(
    output_paths: Dict[str, str],
    model: nn.Module,
    loss_fn: nn.Module,
    optimizer: Optimizer,
    scheduler: _LRScheduler,
    args: Any
) -> Dict[str, Any]:
    """
    Load the most recent checkpoint from the checkpoints directory.
    
    Args:
        output_paths: Dictionary with paths including 'checkpoints'
        model: Model instance to load state into
        loss_fn: Loss function instance
        optimizer: Optimizer instance
        scheduler: Learning rate scheduler instance
        args: Argument namespace with training config
        
    Returns:
        dict: Training state information
    """
    checkpoint_dir = Path(output_paths['checkpoints'])
    
    # Find all checkpoint files
    checkpoint_files = []
    if checkpoint_dir.exists():
        for f in checkpoint_dir.iterdir():
            if f.suffix in [CHECKPOINT_EXTENSION, CHECKPOINT_EXTENSION_ALT]:
                checkpoint_files.append(f)
    
    if not checkpoint_files:
        print("No checkpoints found. Starting training from scratch.")
        return _get_default_checkpoint_data()
    
    # Sort by modification time (most recent last)
    checkpoint_files.sort(key=lambda x: x.stat().st_mtime)
    last_checkpoint = checkpoint_files[-1]
    
    print(f"\n{'='*60}")
    print(f"Loading checkpoint: {last_checkpoint}")
    print(f"{'='*60}")
    
    try:
        # Load checkpoint with weights_only=False to avoid the warning
        checkpoint = torch.load(last_checkpoint, map_location='cpu', weights_only=False)
        
        # Load model state with strict=False to ignore metadata
        model.load_state_dict(checkpoint[CHECKPOINT_KEYS['model_state']], strict=False)
        print("✓ Model state loaded")
        
        # Load loss function state (if it has learnable parameters)
        loss_state_key = CHECKPOINT_KEYS['loss_state']
        if loss_state_key in checkpoint and hasattr(loss_fn, 'load_state_dict'):
            try:
                loss_fn.load_state_dict(checkpoint[loss_state_key], strict=False)
                print("✓ Loss function state loaded")
            except Exception as e:
                print(f"⚠ Warning: Could not load loss state: {e}")
        
        # Load optimizer state
        try:
            optimizer.load_state_dict(checkpoint[CHECKPOINT_KEYS['optimizer_state']])
            print("✓ Optimizer state loaded")
        except Exception as e:
            print(f"⚠ Warning: Could not load optimizer state: {e}")
            print("  (Will continue with fresh optimizer)")
        
        # Load scheduler state
        scheduler_state_key = CHECKPOINT_KEYS['scheduler_state']
        if scheduler_state_key in checkpoint:
            try:
                scheduler.load_state_dict(checkpoint[scheduler_state_key])
                print("✓ Scheduler state loaded")
            except Exception as e:
                print(f"⚠ Warning: Could not load scheduler state: {e}")
        
        # Extract training history using constants
        start_epoch = checkpoint.get(CHECKPOINT_KEYS['epoch'], 0) + 1
        best_val_loss = checkpoint.get(CHECKPOINT_KEYS['best_val_loss'], float('inf'))
        train_losses = checkpoint.get(CHECKPOINT_KEYS['train_losses'], [])
        val_losses = checkpoint.get(CHECKPOINT_KEYS['val_losses'], [])
        train_loss_components = checkpoint.get(CHECKPOINT_KEYS['train_loss_components'], [])
        val_loss_components = checkpoint.get(CHECKPOINT_KEYS['val_loss_components'], [])
        loss_weights_history = checkpoint.get(CHECKPOINT_KEYS['loss_weights_history'], [])
        
        # Get normalization stats
        data_mean = checkpoint.get(CHECKPOINT_KEYS['data_mean'], None)
        data_std = checkpoint.get(CHECKPOINT_KEYS['data_std'], None)
        
        print(f"\nCheckpoint Info:")
        print(f"  Resuming from epoch: {start_epoch}")
        print(f"  Best validation loss: {best_val_loss:.6f}")
        print(f"  Training history: {len(train_losses)} epochs")
        print(f"  Current LR: {optimizer.param_groups[0]['lr']:.2e}")
        
        if args.learnable_loss and loss_weights_history:
            final_weights = loss_weights_history[-1]
            print(f"  Loss weights: Rel={final_weights[0]:.3f}, Temp={final_weights[1]:.3f}")
        
        print(f"{'='*60}\n")
        
        return {
            'start_epoch': start_epoch,
            'best_val_loss': best_val_loss,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_loss_components': train_loss_components,
            'val_loss_components': val_loss_components,
            'loss_weights_history': loss_weights_history,
            'data_mean': data_mean,
            'data_std': data_std,
            'loaded': True
        }
        
    except Exception as e:
        print(f"\n✗ Error loading checkpoint: {e}")
        print("Starting training from scratch.\n")
        import traceback
        traceback.print_exc()
        
        return _get_default_checkpoint_data()


def _get_default_checkpoint_data() -> Dict[str, Any]:
    """
    Get default checkpoint data structure for fresh training.
    
    Returns:
        Dictionary with default checkpoint values
    """
    return {
        'start_epoch': 0,
        'best_val_loss': float('inf'),
        'train_losses': [],
        'val_losses': [],
        'train_loss_components': [],
        'val_loss_components': [],
        'loss_weights_history': [],
        'data_mean': None,
        'data_std': None,
        'loaded': False
    }


def load_specific_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    loss_fn: nn.Module,
    optimizer: Optimizer,
    scheduler: _LRScheduler
) -> Dict[str, Any]:
    """
    Load a specific checkpoint by path.
    
    Args:
        checkpoint_path: Path to checkpoint file
        model: Model instance
        loss_fn: Loss function instance
        optimizer: Optimizer instance
        scheduler: Learning rate scheduler instance
        
    Returns:
        Dictionary with training state
        
    Raises:
        FileNotFoundError: If checkpoint file doesn't exist
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    print(f"Loading specific checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    model.load_state_dict(checkpoint[CHECKPOINT_KEYS['model_state']])
    loss_fn.load_state_dict(checkpoint[CHECKPOINT_KEYS['loss_state']])
    optimizer.load_state_dict(checkpoint[CHECKPOINT_KEYS['optimizer_state']])
    scheduler.load_state_dict(checkpoint[CHECKPOINT_KEYS['scheduler_state']])
    
    return {
        'start_epoch': checkpoint[CHECKPOINT_KEYS['epoch']] + 1,
        'best_val_loss': checkpoint[CHECKPOINT_KEYS['best_val_loss']],
        'train_losses': checkpoint[CHECKPOINT_KEYS['train_losses']],
        'val_losses': checkpoint[CHECKPOINT_KEYS['val_losses']],
        'train_loss_components': checkpoint[CHECKPOINT_KEYS['train_loss_components']],
        'val_loss_components': checkpoint[CHECKPOINT_KEYS['val_loss_components']],
        'loss_weights_history': checkpoint.get(CHECKPOINT_KEYS['loss_weights_history'], []),
        'data_mean': checkpoint[CHECKPOINT_KEYS['data_mean']],
        'data_std': checkpoint[CHECKPOINT_KEYS['data_std']],
        'loaded': True
    }