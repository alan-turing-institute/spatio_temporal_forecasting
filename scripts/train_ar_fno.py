#!/usr/bin/env python3
"""
Autoregressive FNO Training Script

This script trains an autoregressive Fourier Neural Operator for spatiotemporal prediction
in physics simulations. It supports:
- Configurable FNO architecture
- Learnable and fixed loss weights
- Teacher forcing schedules
- Automatic checkpointing and resumption
- Comprehensive visualization
- Long-term prediction evaluation

Usage:
    Basic training:
        python train_ar_fno.py --experiment-name my_exp --n-epochs 500
    
    Resume training:
        python train_ar_fno.py --experiment-name my_exp --resume
    
    Custom hyperparameters:
        python train_ar_fno.py --experiment-name advanced \\
            --n-modes 12 12 --hidden-channels 32 --t-in 10 --t-out 5

For more options, run:
    python train_ar_fno.py --help
"""

import os
import argparse
from typing import Tuple, Dict

import torch
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt

from spatio_temporal_forecasting.fno_emulator import MultivariableFNO
from spatio_temporal_forecasting.AR_FNO import AutoregressiveFNO
from spatio_temporal_forecasting.loss import Loss
from spatio_temporal_forecasting.make_gif import visualize_predictions, plot_gif, create_gif_visualization, create_long_term_predictions

from spatio_temporal_forecasting.load_checkpoint import load_last_checkpoint
from spatio_temporal_forecasting.loss import plot_loss_with_components, plot_loss_weights_evolution
from spatio_temporal_forecasting.load_data import parse_args, set_seed, parse_tf_schedule, load_data, compute_normalization_stats
from spatio_temporal_forecasting.constants import DEFAULT_GRID_SIZE, PRINT_FREQ_BATCHES, CHECKPOINT_KEYS, EPSILON


def create_model(args: argparse.Namespace, device: torch.device) -> AutoregressiveFNO:
    """Create and initialize model."""
    print("Creating model...")
    
    # Create base FNO
    fno_base = MultivariableFNO(
        n_vars=1,
        n_modes=tuple(args.n_modes),
        hidden_channels=args.hidden_channels,
        n_layers=args.n_layers,
        use_skip_connections=False
    )
    
    # Wrap with autoregressive functionality
    model = AutoregressiveFNO(
        fno_model=fno_base,
        t_in=args.t_in,
        t_out=args.t_out,
        step_size=args.step_size,
        teacher_forcing_ratio=args.tf_initial
    )
    
    model = model.to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    return model


def create_spatial_grid(size: int, device: torch.device) -> torch.Tensor:
    """
    Create 2D coordinate grid for spatial processing.
    
    Args:
        size: Grid size (assumes square grid size x size)
        device: Device to place grid on
        
    Returns:
        Grid tensor of shape [2, size, size] with x and y coordinates in [0, 1]
    """
    xx, yy = torch.meshgrid(
        torch.linspace(0, 1, size, device=device),
        torch.linspace(0, 1, size, device=device),
        indexing='xy'
    )
    grid = torch.stack([xx, yy], dim=0)
    return grid


def compute_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    data_mean: torch.Tensor,
    data_std: torch.Tensor
) -> Dict[str, float]:
    """
    Compute evaluation metrics for predictions.
    
    Metrics include:
    - Relative error: Mean absolute relative error
    - Max error: Maximum absolute error
    - Conservation error: Error in total quantity conservation
    
    Args:
        pred: Predictions [B, T, n_vars, H, W]
        target: Ground truth [B, T, n_vars, H, W]
        data_mean: Normalization mean
        data_std: Normalization std
        
    Returns:
        Dictionary with metric names and values
    """
    with torch.no_grad():
        # Denormalize
        pred_denorm = pred * data_std + data_mean
        target_denorm = target * data_std + data_mean
        
        # Relative error
        rel_error = torch.mean(
            torch.abs(pred_denorm - target_denorm) / (torch.abs(target_denorm) + EPSILON)
        )
        
        # Max error
        max_error = torch.max(torch.abs(pred_denorm - target_denorm))
        
        # Conservation error
        pred_sum = torch.sum(pred_denorm, dim=(-2, -1))
        target_sum = torch.sum(target_denorm, dim=(-2, -1))
        conservation_error = torch.mean(
            torch.abs(pred_sum - target_sum) / (torch.abs(target_sum) + EPSILON)
        )
        
        return {
            'rel_error': rel_error.item(),
            'max_error': max_error.item(),
            'conservation_error': conservation_error.item()
        }


def train_epoch(
    model: AutoregressiveFNO,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: Loss,
    grid: torch.Tensor,
    data_mean: torch.Tensor,
    data_std: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device
) -> Tuple[float, Dict[str, float], Dict[str, float]]:
    """
    Train model for one epoch.
    
    Args:
        model: Model to train
        train_loader: Training data loader
        optimizer: Optimizer
        loss_fn: Loss function
        grid: Spatial grid
        data_mean: Normalization mean
        data_std: Normalization std
        args: Training arguments
        device: Device
        
    Returns:
        Tuple of (average_loss, loss_components, metrics)
    """
    model.train()
    epoch_train_loss = 0.0
    epoch_loss_components = {'relative_lp': 0.0, 'temporal_consistency': 0.0}
    train_metrics = {'rel_error': 0, 'max_error': 0, 'conservation_error': 0}
    n_train_batches = 0
    
    for batch_idx, batch in enumerate(train_loader):
        x_batch = batch["input_fields"].permute(0, 1, 4, 2, 3).to(device)
        y_batch = batch["output_fields"].permute(0, 1, 4, 2, 3).to(device)
        
        # Normalize
        x_batch = (x_batch - data_mean) / data_std
        y_batch = (y_batch - data_mean) / data_std
        
        # Create batch grid
        batch_size = x_batch.shape[0]
        batch_grid = grid.unsqueeze(0).expand(batch_size, -1, -1, -1)
        
        # Forward pass
        optimizer.zero_grad()
        pred = model(x_batch, target=y_batch, grid=batch_grid)
        
        # Compute loss
        loss = loss_fn(pred, y_batch)
        
        # Get loss components for monitoring
        loss_components = loss_fn.get_loss_components(pred, y_batch)
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
        optimizer.step()
        
        # Track metrics
        epoch_train_loss += loss.item()
        for key in epoch_loss_components:
            epoch_loss_components[key] += loss_components[key]
        
        batch_metrics = compute_metrics(pred, y_batch, data_mean, data_std)
        for key, value in batch_metrics.items():
            train_metrics[key] += value
        n_train_batches += 1
        
        if batch_idx % PRINT_FREQ_BATCHES == 0:
            print(f"  Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.6f}, "
                  f"RelErr: {batch_metrics['rel_error']:.4f}")
    
    # Average metrics
    avg_train_loss = epoch_train_loss / n_train_batches
    for key in epoch_loss_components:
        epoch_loss_components[key] /= n_train_batches
    for key in train_metrics:
        train_metrics[key] /= n_train_batches
    
    return avg_train_loss, epoch_loss_components, train_metrics


def validate(
    model: AutoregressiveFNO,
    val_loader: DataLoader,
    loss_fn: Loss,
    grid: torch.Tensor,
    data_mean: torch.Tensor,
    data_std: torch.Tensor,
    device: torch.device
) -> Tuple[float, Dict[str, float], Dict[str, float]]:
    """
    Validate model on validation set.
    
    Uses pure autoregressive prediction without teacher forcing.
    
    Args:
        model: Model to validate
        val_loader: Validation data loader
        loss_fn: Loss function
        grid: Spatial grid
        data_mean: Normalization mean
        data_std: Normalization std
        device: Device
        
    Returns:
        Tuple of (average_loss, loss_components, metrics)
    """
    model.eval()
    val_loss = 0.0
    val_loss_components = {'relative_lp': 0.0, 'temporal_consistency': 0.0}
    val_metrics = {'rel_error': 0, 'max_error': 0, 'conservation_error': 0}
    n_val_batches = 0
    
    with torch.no_grad():
        for batch in val_loader:
            x_batch = batch["input_fields"].permute(0, 1, 4, 2, 3).to(device)
            y_batch = batch["output_fields"].permute(0, 1, 4, 2, 3).to(device)
            
            # Normalize
            x_batch = (x_batch - data_mean) / data_std
            y_batch = (y_batch - data_mean) / data_std
            
            batch_size = x_batch.shape[0]
            batch_grid = grid.unsqueeze(0).expand(batch_size, -1, -1, -1)
            
            # Pure autoregressive prediction
            pred = model.predict(x_batch, n_steps=y_batch.shape[1], grid=batch_grid)
            
            # Compute loss
            loss = loss_fn(pred, y_batch)
            val_loss += loss.item()
            
            # Get loss components
            loss_components = loss_fn.get_loss_components(pred, y_batch)
            for key in val_loss_components:
                val_loss_components[key] += loss_components[key]
            
            batch_metrics = compute_metrics(pred, y_batch, data_mean, data_std)
            for key, value in batch_metrics.items():
                val_metrics[key] += value
            n_val_batches += 1
    
    # Average metrics
    avg_val_loss = val_loss / n_val_batches
    for key in val_loss_components:
        val_loss_components[key] /= n_val_batches
    for key in val_metrics:
        val_metrics[key] /= n_val_batches
    
    return avg_val_loss, val_loss_components, val_metrics


def setup_output_directories(args: argparse.Namespace) -> Dict[str, str]:
    """
    Create output directory structure for experiment.
    
    Creates the following structure:
        output_dir/
        └── experiment_name/
            ├── checkpoints/
            ├── visualizations/
            ├── gifs/
            ├── plots/
            ├── long_term_predictions/
            └── config.txt
    
    Args:
        args: Arguments containing output_dir and experiment_name
    
    Returns:
        Dictionary mapping directory names to absolute paths
    """
    from datetime import datetime
    
    # Generate experiment name if not provided
    if args.experiment_name is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.experiment_name = f'exp_{timestamp}'
    
    # Convert to absolute path to avoid issues with directory changes
    output_dir_abs = os.path.abspath(args.output_dir)
    os.makedirs(output_dir_abs, exist_ok=True)
    
    # Create main experiment directory (absolute path)
    exp_dir = os.path.join(output_dir_abs, args.experiment_name)
    os.makedirs(exp_dir, exist_ok=True)
    
    # Create subdirectories (all absolute paths)
    paths = {
        'root': exp_dir,
        'checkpoints': os.path.join(exp_dir, 'checkpoints'),
        'visualizations': os.path.join(exp_dir, 'visualizations'),
        'gifs': os.path.join(exp_dir, 'gifs'),
        'plots': os.path.join(exp_dir, 'plots'),
        'long_term': os.path.join(exp_dir, 'long_term_predictions')
    }
    
    # Ensure all subdirectories exist
    for path in paths.values():
        os.makedirs(path, exist_ok=True)
    
    # Save configuration
    config_path = os.path.join(exp_dir, 'config.txt')
    with open(config_path, 'w') as f:
        f.write("Experiment Configuration\n")
        f.write("=" * 50 + "\n\n")
        for arg, value in sorted(vars(args).items()):
            f.write(f"{arg}: {value}\n")
    
    print(f"\n{'='*60}")
    print(f"Experiment Setup")
    print(f"{'='*60}")
    print(f"Experiment directory: {exp_dir}")
    print(f"Checkpoints: {paths['checkpoints']}")
    print(f"Visualizations: {paths['visualizations']}")
    print(f"GIFs: {paths['gifs']}")
    print(f"Plots: {paths['plots']}")
    print(f"Long-term predictions: {paths['long_term']}")
    print(f"{'='*60}\n")
    
    return paths


def train(args: argparse.Namespace) -> None:
    """
    Main training function.
    
    Orchestrates the entire training process:
    1. Setup (seeds, device, directories)
    2. Data loading
    3. Model creation
    4. Training loop with validation
    5. Checkpointing
    6. Visualization
    7. Long-term prediction evaluation
    
    Args:
        args: Command-line arguments with all training configuration
    """
    # Setup
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Enable cudnn optimizations
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    
    # Change to target directory
    if args.target_dir:
        os.chdir(args.target_dir)
        print(f"Working directory: {os.getcwd()}")
    
    # Setup output directories
    output_paths = setup_output_directories(args)
    
    # Load data
    train_loader, val_loader, full_data = load_data(args)
    
    # Compute normalization
    data_mean, data_std = compute_normalization_stats(train_loader, device)
    
    # Create model
    model = create_model(args, device)
    
    # Create spatial grid (assuming 64x64 from data)
    grid = create_spatial_grid(DEFAULT_GRID_SIZE, device)
    
    # Create loss function
    loss_fn = Loss(
        d=2, 
        p=2, 
        alpha_temporal=args.alpha_temporal,
        learnable=args.learnable_loss
    ).to(device)
    
    print(f"\nLoss Configuration:")
    print(f"  Learnable weights: {args.learnable_loss}")
    print(f"  Alpha temporal: {args.alpha_temporal}")
    if args.learnable_loss:
        print(f"  Initial weights: {loss_fn.get_weights()}")
    
    # Setup optimizer (include loss parameters if learnable)
    if args.learnable_loss:
        optimizer = torch.optim.Adam(
            list(model.parameters()) + list(loss_fn.parameters()),
            lr=args.lr,
            weight_decay=args.weight_decay
        )
    else:
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=args.scheduler_factor,
        patience=args.scheduler_patience
    )

    
    # ===== LOAD CHECKPOINT IF EXISTS =====
    checkpoint_data = load_last_checkpoint(
        output_paths, model, loss_fn, optimizer, scheduler, args
    )
    
    # Parse teacher forcing schedule
    tf_schedule = parse_tf_schedule(args.tf_schedule)
    
    # Training tracking
    train_losses = []
    val_losses = []
    train_loss_components = []
    val_loss_components = []
    loss_weights_history = []
    best_val_loss = float('inf')

    start_epoch = checkpoint_data['start_epoch']
    best_val_loss = checkpoint_data['best_val_loss']
    train_losses = checkpoint_data['train_losses']
    val_losses = checkpoint_data['val_losses']
    train_loss_components = checkpoint_data['train_loss_components']
    val_loss_components = checkpoint_data['val_loss_components']
    loss_weights_history = checkpoint_data['loss_weights_history']
    
    # Handle normalization stats
    if checkpoint_data['loaded'] and checkpoint_data['data_mean'] is not None:
        # Use saved normalization
        data_mean = checkpoint_data['data_mean'].to(device)
        data_std = checkpoint_data['data_std'].to(device)
        print("Using normalization stats from checkpoint")
    else:
        # Compute normalization
        data_mean, data_std = compute_normalization_stats(train_loader, device)
    # ===== END CHECKPOINT LOADING =====

    
    # Training loop
    print(f"\nStarting training from epoch {start_epoch + 1}...")
    for epoch in range(start_epoch, args.n_epochs):
        print(f"\nEpoch {epoch+1}/{args.n_epochs}")
        
        # Update teacher forcing ratio
        for threshold, ratio in tf_schedule:
            if epoch < threshold:
                model.teacher_forcing_ratio = ratio
                break
        
        # Train
        train_loss, train_comp, train_met = train_epoch(
            model, train_loader, optimizer, loss_fn, grid,
            data_mean, data_std, args, device
        )
        
        train_losses.append(train_loss)

        train_loss_components.append(train_comp)
        
        # Validate
        val_loss, val_comp, val_met = validate(
            model, val_loader, loss_fn, grid, data_mean, data_std, device
        )
        
        val_losses.append(val_loss)
        val_loss_components.append(val_comp)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Print summary
        print(f"Epoch {epoch+1} Results:")
        print(f"  Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
        print(f"  Train RelErr: {train_met['rel_error']:.4f} | Val RelErr: {val_met['rel_error']:.4f}")
        print(f"  Conservation Error: {val_met['conservation_error']:.4f}")
        print(f"  Loss Components:")
        print(f"    Train - Rel Lp: {train_comp['relative_lp']:.6f}, Temporal: {train_comp['temporal_consistency']:.6f}")
        print(f"    Val   - Rel Lp: {val_comp['relative_lp']:.6f}, Temporal: {val_comp['temporal_consistency']:.6f}")
        print(f"  Teacher Forcing: {model.teacher_forcing_ratio:.2f}")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.2e}")
        
        # Monitor loss weights if learnable
        if args.learnable_loss:
            weights = loss_fn.get_weights()
            loss_weights_history.append(weights)
            print(f"  Loss Weights: Rel={weights[0]:.3f}, Temp={weights[1]:.3f}")
        
        # Visualizations
        if (epoch + 1) % args.viz_freq == 0:
            print(f"Creating visualizations for epoch {epoch+1}...")
            
            # Get absolute paths to subdirectories
            viz_dir = output_paths['visualizations']
            gif_dir = output_paths['gifs']
            plot_dir = output_paths['plots']
            
            # Ensure directories exist
            os.makedirs(viz_dir, exist_ok=True)
            os.makedirs(gif_dir, exist_ok=True)
            os.makedirs(plot_dir, exist_ok=True)
            
            # Save original directory
            original_dir = os.getcwd()
            
            try:
                # Create GIFs - change to gif directory
                os.chdir(gif_dir)
                create_gif_visualization(
                    model, val_loader, grid, data_mean, data_std,
                    epoch+1, device, n_samples=args.n_viz_samples
                )
                
                plot_gif(
                    model, val_loader, grid, data_mean, data_std,
                    N=5, device=device, epoch=epoch, prefix='training'
                )
                
                # Create visualizations - change to viz directory
                os.chdir(viz_dir)
                visualize_predictions(
                    model, val_loader, grid, data_mean, data_std,
                    epoch+1, device, n_samples=2
                )
                
                # Create plots - change to plot directory
                os.chdir(plot_dir)
                plot_loss_with_components(
                    train_losses, val_losses, 
                    train_loss_components, val_loss_components
                )
                
                # Plot loss weights evolution if learnable
                if args.learnable_loss and len(loss_weights_history) > 0:
                    weights_array = np.array(loss_weights_history)
                    plt.figure(figsize=(10, 6))
                    plt.plot(weights_array[:, 0], label='Relative Lp Weight', linewidth=2)
                    plt.plot(weights_array[:, 1], label='Temporal Weight', linewidth=2)
                    plt.xlabel('Epoch')
                    plt.ylabel('Weight')
                    plt.title('Loss Weight Evolution')
                    plt.legend()
                    plt.grid(True, alpha=0.3)
                    plt.tight_layout()
                    plt.savefig('loss_weights_history.png', dpi=150)
                    plt.close()
            
            except Exception as e:
                print(f"Warning: Error during visualization: {e}")
                print(f"Current directory: {os.getcwd()}")
                print(f"Expected directories exist: viz={os.path.exists(viz_dir)}, "
                      f"gif={os.path.exists(gif_dir)}, plot={os.path.exists(plot_dir)}")
            
            finally:
                # Always return to original directory
                os.chdir(original_dir)

            checkpoint = {
                CHECKPOINT_KEYS['epoch']: epoch,
                CHECKPOINT_KEYS['model_state']: model.state_dict(),
                CHECKPOINT_KEYS['loss_state']: loss_fn.state_dict(),
                CHECKPOINT_KEYS['optimizer_state']: optimizer.state_dict(),
                CHECKPOINT_KEYS['scheduler_state']: scheduler.state_dict(),
                CHECKPOINT_KEYS['data_mean']: data_mean,
                CHECKPOINT_KEYS['data_std']: data_std,
                CHECKPOINT_KEYS['train_losses']: train_losses,
                CHECKPOINT_KEYS['val_losses']: val_losses,
                CHECKPOINT_KEYS['train_loss_components']: train_loss_components,
                CHECKPOINT_KEYS['val_loss_components']: val_loss_components,
                CHECKPOINT_KEYS['best_val_loss']: best_val_loss,
                CHECKPOINT_KEYS['args']: vars(args)
            }
            checkpoint_path = os.path.join(output_paths['checkpoints'], f"model_epoch_{epoch+1}.pth")
            torch.save(checkpoint, checkpoint_path)
            print(f"Model saved at visualization frequency for epoch {epoch} to {checkpoint_path}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint = {
                CHECKPOINT_KEYS['epoch']: epoch,
                CHECKPOINT_KEYS['model_state']: model.state_dict(),
                CHECKPOINT_KEYS['loss_state']: loss_fn.state_dict(),
                CHECKPOINT_KEYS['optimizer_state']: optimizer.state_dict(),
                CHECKPOINT_KEYS['scheduler_state']: scheduler.state_dict(),
                CHECKPOINT_KEYS['data_mean']: data_mean,
                CHECKPOINT_KEYS['data_std']: data_std,
                CHECKPOINT_KEYS['train_losses']: train_losses,
                CHECKPOINT_KEYS['val_losses']: val_losses,
                CHECKPOINT_KEYS['train_loss_components']: train_loss_components,
                CHECKPOINT_KEYS['val_loss_components']: val_loss_components,
                CHECKPOINT_KEYS['best_val_loss']: best_val_loss,
                CHECKPOINT_KEYS['args']: vars(args)
            }
            checkpoint_path = os.path.join(output_paths['checkpoints'], args.checkpoint_name)
            torch.save(checkpoint, checkpoint_path)
            print(f"Best Model found and saved to {checkpoint_path}")
        
        # Early stopping check
        if len(val_losses) > 15:
            recent_losses = val_losses[-5:]
            if all(recent_losses[i] >= recent_losses[i-1] for i in range(1, 5)):
                print("  Early stopping triggered")
                # Uncomment to enable early stopping
                # break
        
        print("-" * 40)
    
    # Final visualizations
    print("\nCreating final visualizations...")
    
    # Ensure all output directories still exist
    for path in output_paths.values():
        os.makedirs(path, exist_ok=True)
    
    original_dir = os.getcwd()
    try:
        os.chdir(output_paths['visualizations'])
        visualize_predictions(
            model, val_loader, grid, data_mean, data_std,
            "FINAL", device, n_samples=3
        )
        
        os.chdir(output_paths['gifs'])
        create_gif_visualization(
            model, val_loader, grid, data_mean, data_std,
            "FINAL", device, n_samples=args.n_viz_samples
        )
        
        os.chdir(output_paths['plots'])
        plot_loss_with_components(
            train_losses, val_losses,
            train_loss_components, val_loss_components,
        )
        
        if args.learnable_loss and len(loss_weights_history) > 0:
            plot_loss_weights_evolution(loss_weights_history, save_path='.')
    
    finally:
        os.chdir(original_dir)
    
    # Long-term predictions
    if not args.skip_long_term:
        print("\n" + "="*60)
        print("Creating long-term predictions...")
        print("="*60)
        
        # Ensure long-term directory exists
        os.makedirs(output_paths['long_term'], exist_ok=True)
        
        create_long_term_predictions(
            model=model,
            full_data=full_data,
            grid=grid,
            data_mean=data_mean,
            data_std=data_std,
            device=device,
            output_dir=output_paths['long_term'],
            n_samples=args.n_long_term_samples,
            t_in=args.t_in
        )
    else:
        print("\nSkipping long-term predictions (--skip-long-term flag set)")
    
    print(f"\n{'='*60}")
    print("Training Complete!")
    print(f"{'='*60}")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"All results saved to: {output_paths['root']}")
    print(f"Model checkpoint: {os.path.join(output_paths['checkpoints'], args.checkpoint_name)}")
    if args.learnable_loss:
        final_weights = loss_fn.get_weights()
        print(f"Final loss weights: Rel={final_weights[0]:.3f}, Temp={final_weights[1]:.3f}")
    print(f"{'='*60}\n")


def main() -> None:
    """
    Entry point for training script.
    
    Parses command-line arguments and initiates training.
    """
    args = parse_args()
    train(args)


if __name__ == '__main__':
    main()