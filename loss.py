"""Loss functions for spatiotemporal prediction."""
import os
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from constants import EPSILON, DEFAULT_LP_NORM_ORDER, DEFAULT_SPATIAL_DIM


class Loss(nn.Module):
    def __init__(
        self,
        d: int = DEFAULT_SPATIAL_DIM,
        p: int = DEFAULT_LP_NORM_ORDER,
        alpha_temporal: float = 0.5,
        learnable: bool = False
    ):
        """
        Loss module for PDE problems with temporal consistency.
        
        Args:
            d: Spatial dimension (default: 2)
            p: Order of Lp norm (default: 2 for L2)
            alpha_temporal: Weight for temporal consistency loss (default: 0.5)
            learnable: If True, loss weights are learnable parameters (default: False)
        """
        super().__init__()
        self.d = d
        self.p = p
        self.learnable = learnable
        
        if self.learnable:
            # Learnable weights: [relative_lp, temporal_consistency]
            self.loss_weights = nn.Parameter(torch.tensor([1.0, alpha_temporal]))
        else:
            # Fixed weights
            self.register_buffer('loss_weights', torch.tensor([1.0, alpha_temporal]))
    
    def relative_lp_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute relative Lp loss.
        
        Args:
            pred: Predictions [B, ...]
            target: Ground truth [B, ...]
        
        Returns:
            Relative Lp loss averaged over batch
        """
        batch_size = pred.shape[0]
        
        # Flatten all dimensions except batch
        pred_flat = pred.reshape(batch_size, -1)
        target_flat = target.reshape(batch_size, -1)
        
        # Compute Lp norms per sample
        diff_norms = torch.norm(pred_flat - target_flat, p=self.p, dim=1)
        target_norms = torch.norm(target_flat, p=self.p, dim=1)
        
        # Relative error per sample, then average
        relative_errors = diff_norms / (target_norms + EPSILON)
        
        return torch.mean(relative_errors)
    
    def temporal_consistency_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute temporal consistency loss.
        Penalizes inconsistent temporal derivatives between consecutive timesteps.
        
        Args:
            pred: Predictions [B, T, ...]
            target: Ground truth [B, T, ...]
        
        Returns:
            Temporal consistency loss (0 if only 1 timestep)
        """
        # Check if we have multiple timesteps
        if pred.shape[1] <= 1:
            return torch.tensor(0.0, device=pred.device)
        
        batch_size = pred.shape[0]
        n_timesteps = pred.shape[1]
        
        # Compute temporal differences (finite differences)
        pred_diff = pred[:, 1:] - pred[:, :-1]      # [B, T-1, ...]
        target_diff = target[:, 1:] - target[:, :-1]  # [B, T-1, ...]
        
        # Reshape to [B, T-1, -1] for norm computation
        pred_diff_flat = pred_diff.reshape(batch_size, n_timesteps - 1, -1)
        target_diff_flat = target_diff.reshape(batch_size, n_timesteps - 1, -1)
        
        # Compute Lp norms over spatial dimensions (dim=2)
        diff_norms = torch.norm(pred_diff_flat - target_diff_flat, p=self.p, dim=2)
        target_norms = torch.norm(target_diff_flat, p=self.p, dim=2)
        
        # Relative error per sample and timestep, then average
        relative_errors = diff_norms / (target_norms + EPSILON)
        
        return torch.mean(relative_errors)
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute total loss.
        
        Args:
            pred: Predictions [B, T, ...] or [B, ...]
            target: Ground truth [B, T, ...] or [B, ...]
        
        Returns:
            Total loss value
        """
        # Compute individual loss terms
        rel_loss = self.relative_lp_loss(pred, target)
        temp_loss = self.temporal_consistency_loss(pred, target)
        
        if self.learnable:
            # Use softmax-normalized learnable weights
            # I have changed this to exp as the model is forcing the temporal to go to zero  
            weights = torch.exp(self.loss_weights)
            total_loss = weights[0] * rel_loss + weights[1] * temp_loss
        else:
            # Use fixed weights (loss_weights[0] is always 1.0)
            total_loss = rel_loss + self.loss_weights[1] * temp_loss
        
        return total_loss
    
    def get_loss_components(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> Dict[str, float]:
        """
        Get individual loss components for logging/monitoring.
        
        Args:
            pred: Predictions [B, T, ...] or [B, ...]
            target: Ground truth [B, T, ...] or [B, ...]
        
        Returns:
            Dictionary with individual loss values
        """
        with torch.no_grad():
            rel_loss = self.relative_lp_loss(pred, target)
            temp_loss = self.temporal_consistency_loss(pred, target)
            
            if self.learnable:
                weights = torch.exp(self.loss_weights)
                return {
                    'relative_lp': rel_loss.item(),
                    'temporal_consistency': temp_loss.item(),
                    'weight_rel': weights[0].item(),
                    'weight_temp': weights[1].item()
                }
            else:
                return {
                    'relative_lp': rel_loss.item(),
                    'temporal_consistency': temp_loss.item(),
                    'weight_rel': 1.0,
                    'weight_temp': self.loss_weights[1].item()
                }
    
    def get_weights(self) -> np.ndarray:
        """
        Get current loss weights (for monitoring during training).
        
        Returns:
            Numpy array of weights
        """
        if self.learnable:
            weights = torch.exp(self.loss_weights)
            return weights.detach().cpu().numpy()
        else:
            return self.loss_weights.cpu().numpy()


def plot_loss(
    train_losses: List[float],
    val_losses: List[float]
) -> None:
    """
    Plot training and validation losses - simple version.
    
    Args:
        train_losses: List of training losses per epoch
        val_losses: List of validation losses per epoch
    """
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss', linewidth=2)
    plt.plot(val_losses, label='Val Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Progress')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig('loss_history.png', dpi=150)
    plt.close()


def plot_loss_with_components(
    train_losses: List[float],
    val_losses: List[float],
    train_components: List[Dict[str, float]],
    val_components: List[Dict[str, float]],
    save_path: str = '.'
) -> None:
    """
    Plot comprehensive training and validation losses with loss components.
    
    Args:
        train_losses: List of total training losses per epoch
        val_losses: List of total validation losses per epoch
        train_components: List of dicts with keys ['relative_lp', 'temporal_consistency'] per epoch
        val_components: List of dicts with keys ['relative_lp', 'temporal_consistency'] per epoch
        save_path: Path to save the plot (default: '.')
    """
    epochs = range(1, len(train_losses) + 1)
    
    fig = plt.figure(figsize=(18, 10))
    
    # Main plot: Total losses
    ax1 = plt.subplot(2, 3, 1)
    ax1.plot(epochs, train_losses, 'b-', linewidth=2, label='Training Loss', alpha=0.8)
    ax1.plot(epochs, val_losses, 'r-', linewidth=2, label='Validation Loss', alpha=0.8)
    ax1.plot(epochs, train_losses, 'bo', markersize=4, alpha=0.6)
    ax1.plot(epochs, val_losses, 'ro', markersize=4, alpha=0.6)
    
    # Mark best validation loss
    if val_losses:
        best_epoch = np.argmin(val_losses) + 1
        best_val_loss = min(val_losses)
        ax1.plot(best_epoch, best_val_loss, 'g*', markersize=15, 
                label=f'Best: {best_val_loss:.6f} (Epoch {best_epoch})')
        ax1.axvline(x=best_epoch, color='green', linestyle='--', alpha=0.5)
    
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Total Loss', fontsize=12)
    ax1.set_title('Total Loss (Relative Lp + α×Temporal)', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Extract component data
    train_rel = [c['relative_lp'] for c in train_components]
    train_temp = [c['temporal_consistency'] for c in train_components]
    val_rel = [c['relative_lp'] for c in val_components]
    val_temp = [c['temporal_consistency'] for c in val_components]
    
    # Plot 2: Relative Lp loss component
    ax2 = plt.subplot(2, 3, 2)
    ax2.plot(epochs, train_rel, 'b-', linewidth=2, label='Train Relative Lp', alpha=0.8)
    ax2.plot(epochs, val_rel, 'r-', linewidth=2, label='Val Relative Lp', alpha=0.8)
    ax2.plot(epochs, train_rel, 'bo', markersize=3, alpha=0.6)
    ax2.plot(epochs, val_rel, 'ro', markersize=3, alpha=0.6)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Relative Lp Loss', fontsize=12)
    ax2.set_title('Relative L2 Loss Component', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Temporal consistency loss component
    ax3 = plt.subplot(2, 3, 3)
    ax3.plot(epochs, train_temp, 'b-', linewidth=2, label='Train Temporal', alpha=0.8)
    ax3.plot(epochs, val_temp, 'r-', linewidth=2, label='Val Temporal', alpha=0.8)
    ax3.plot(epochs, train_temp, 'bo', markersize=3, alpha=0.6)
    ax3.plot(epochs, val_temp, 'ro', markersize=3, alpha=0.6)
    ax3.set_xlabel('Epoch', fontsize=12)
    ax3.set_ylabel('Temporal Consistency Loss', fontsize=12)
    ax3.set_title('Temporal Consistency Component', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Ratio of temporal to relative loss
    ax4 = plt.subplot(2, 3, 4)
    train_ratio = [t/r if r > 0 else 0 for t, r in zip(train_temp, train_rel)]
    val_ratio = [t/r if r > 0 else 0 for t, r in zip(val_temp, val_rel)]
    ax4.plot(epochs, train_ratio, 'b-', linewidth=2, label='Train Temporal/RelLp', alpha=0.8)
    ax4.plot(epochs, val_ratio, 'r-', linewidth=2, label='Val Temporal/RelLp', alpha=0.8)
    ax4.set_xlabel('Epoch', fontsize=12)
    ax4.set_ylabel('Ratio', fontsize=12)
    ax4.set_title('Temporal / Relative Lp Ratio', fontsize=14, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)
    ax4.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Equal contribution')
    
    # Plot 5: Log scale total loss
    ax5 = plt.subplot(2, 3, 5)
    ax5.semilogy(epochs, train_losses, 'b-', linewidth=2, label='Training Loss', alpha=0.8)
    ax5.semilogy(epochs, val_losses, 'r-', linewidth=2, label='Validation Loss', alpha=0.8)
    if val_losses:
        best_epoch = np.argmin(val_losses) + 1
        best_val_loss = min(val_losses)
        ax5.plot(best_epoch, best_val_loss, 'g*', markersize=15)
    ax5.set_xlabel('Epoch', fontsize=12)
    ax5.set_ylabel('Loss (log scale)', fontsize=12)
    ax5.set_title('Total Loss (Log Scale)', fontsize=14, fontweight='bold')
    ax5.legend(fontsize=10)
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: Stacked area plot showing component contributions
    ax6 = plt.subplot(2, 3, 6)
    ax6.fill_between(epochs, 0, train_rel, alpha=0.5, label='Relative Lp', color='blue')
    ax6.fill_between(epochs, train_rel, 
                    [r + t for r, t in zip(train_rel, train_temp)], 
                    alpha=0.5, label='Temporal (α×)', color='cyan')
    ax6.plot(epochs, train_losses, 'k-', linewidth=2, label='Total', alpha=0.8)
    ax6.set_xlabel('Epoch', fontsize=12)
    ax6.set_ylabel('Loss', fontsize=12)
    ax6.set_title('Training Loss Composition', fontsize=14, fontweight='bold')
    ax6.legend(fontsize=10)
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    loss_plot_path = os.path.join(save_path, 'loss_history.png')
    plt.savefig(loss_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Loss plot saved to: {loss_plot_path}")


def plot_loss_weights_evolution(
    loss_weights_history: List[np.ndarray],
    save_path: str = '.'
) -> None:
    """
    Plot how learnable loss weights evolve during training.
    
    Args:
        loss_weights_history: List of weight arrays [rel_lp_weight, temporal_weight] over epochs
        save_path: Directory to save plot (default: '.')
    """
    if not loss_weights_history or len(loss_weights_history) == 0:
        print("No loss weights history to plot")
        return
    
    # Convert to numpy array for easier plotting
    weights_array = np.array(loss_weights_history)  # Shape: (n_epochs, 2)
    epochs = range(1, len(loss_weights_history) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Evolution of each weight over time
    ax1.plot(epochs, weights_array[:, 0], linewidth=2, 
            label='Relative Lp Weight', color='blue', alpha=0.8, marker='o', markersize=3)
    ax1.plot(epochs, weights_array[:, 1], linewidth=2, 
            label='Temporal Weight', color='red', alpha=0.8, marker='s', markersize=3)
    
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Weight Value (Softmax Normalized)', fontsize=12)
    ax1.set_title('Loss Weight Evolution\n(How much does each component matter?)', 
                 fontsize=14, fontweight='bold')
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Equal weights')
    
    # Plot 2: Ratio of weights
    weight_ratio = weights_array[:, 1] / (weights_array[:, 0] + 1e-8)
    ax2.plot(epochs, weight_ratio, linewidth=2, color='purple', alpha=0.8)
    ax2.fill_between(epochs, weight_ratio, alpha=0.3, color='purple')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Temporal Weight / Relative Lp Weight', fontsize=12)
    ax2.set_title('Relative Importance of Temporal vs Spatial Loss', 
                 fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Equal importance')
    ax2.legend(fontsize=12)
    
    plt.tight_layout()
    
    weights_plot_path = os.path.join(save_path, 'loss_weights_history.png')
    plt.savefig(weights_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Loss weights evolution plot saved to: {weights_plot_path}")


def plot_temporal_weights(
    temporal_weights_history: List[np.ndarray],
    save_path: str = 'training_visualizations'
) -> None:
    """
    Plot how temporal weights evolve during training.
    
    Args:
        temporal_weights_history: List of temporal weight arrays over epochs
        save_path: Directory to save plot (default: 'training_visualizations')
    """
    os.makedirs(save_path, exist_ok=True)
    
    if not temporal_weights_history:
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Convert to numpy array for easier plotting
    weights_array = np.array(temporal_weights_history)  # Shape: (n_epochs, t_in)
    epochs = range(1, len(temporal_weights_history) + 1)
    t_in = weights_array.shape[1]
    
    # Plot 1: Evolution of each weight over time
    colors = plt.cm.viridis(np.linspace(0, 1, t_in))
    for i in range(t_in):
        ax1.plot(epochs, weights_array[:, i], linewidth=2, 
                label=f't-{t_in-i-1}', color=colors[i], alpha=0.8)
    
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Weight Value', fontsize=12)
    ax1.set_title('Temporal Weight Evolution\n(Which historical timesteps matter?)', 
                 fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10, ncol=2)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Heatmap of weights over epochs
    im = ax2.imshow(weights_array.T, aspect='auto', cmap='viridis', 
                    interpolation='nearest')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Timestep Index', fontsize=12)
    ax2.set_title('Temporal Weights Heatmap\n(Brighter = More Important)', 
                 fontsize=14, fontweight='bold')
    ax2.set_yticks(range(t_in))
    ax2.set_yticklabels([f't-{t_in-i-1}' for i in range(t_in)])
    plt.colorbar(im, ax=ax2, label='Weight Value')
    
    plt.tight_layout()
    
    weights_plot_path = os.path.join(save_path, 'temporal_weights_evolution.png')
    plt.savefig(weights_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Temporal weights plot saved to: {weights_plot_path}")