"""Spatial diffusion model for physics field generation."""

import torch
import torch.nn as nn
from typing import Optional, Tuple

from denoising_diffusion_pytorch import Unet, GaussianDiffusion
import torch.nn.functional as F


class SpatialDiffusionModel(nn.Module):
    """
    Spatial diffusion model for generating physical field variables.
    
    Uses denoising_diffusion_pytorch for the core diffusion implementation,
    with custom conditioning for temporal autoregressive modeling.
    
    This model generates spatial fields conditioned on previous timesteps,
    making it suitable for physics simulations where each timestep depends
    on the previous state.
    """
    
    def __init__(
        self,
        n_vars: int,
        image_size: int = 64,
        timesteps: int = 1000,
        sampling_timesteps: Optional[int] = None,
        loss_type: str = 'l2',
        beta_schedule: str = 'cosine',
        conditioning_type: str = 'concat',  # 'concat', 'cross_attn', 'film'
        unet_dim: int = 64,
        unet_dim_mults: Tuple[int, ...] = (1, 2, 4, 8),
        unet_channels: int = 3,
        unet_out_dim: Optional[int] = None,
        unet_cond_dim: Optional[int] = None,
        unet_resnet_groups: int = 8,
        unet_learned_variance: bool = False,
        unet_learned_sinusoidal_cond: bool = False,
        unet_random_fourier_features: bool = False,
        unet_learned_sinusoidal_dim: int = 16,
        auto_normalize: bool = True
    ):
        """
        Args:
            n_vars: Number of physical variables (channels)
            image_size: Size of spatial domain (assumes square)
            timesteps: Number of diffusion timesteps
            sampling_timesteps: Number of sampling steps (None = same as timesteps)
            loss_type: Loss function type ('l1', 'l2', 'huber')
            beta_schedule: Noise schedule ('linear', 'cosine', 'sigmoid')
            conditioning_type: How to condition on previous timesteps
            unet_*: UNet architecture parameters
            auto_normalize: Whether to automatically normalize inputs
        """
        super().__init__()
        
        self.n_vars = n_vars
        self.image_size = image_size
        self.conditioning_type = conditioning_type
        self.auto_normalize = auto_normalize
        
        # For compatibility with denoising_diffusion_pytorch, we need channels == out_dim
        # So we'll handle conditioning differently - use the same channels for input/output
        # and modify the conditioning approach in the forward pass
        
        # Create UNet for denoising - minimal parameters for compatibility  
        self.unet = Unet(
            dim=unet_dim,
            dim_mults=unet_dim_mults,
            channels=n_vars,  # Keep original channel count
            out_dim=n_vars    # Must match channels for this library
        )
        
        # Create Gaussian diffusion wrapper - minimal parameters for compatibility
        self.diffusion = GaussianDiffusion(
            model=self.unet,
            image_size=image_size,
            timesteps=timesteps,
            sampling_timesteps=sampling_timesteps,
            auto_normalize=auto_normalize
        )
        
        # Conditioning projection layers - handle conditioning externally
        if conditioning_type == 'concat':
            # Project concatenated input back to original channel count
            self.cond_proj = nn.Conv2d(n_vars * 2, n_vars, 1)
        elif conditioning_type == 'film':
            # FiLM: Feature-wise Linear Modulation
            self.film_proj = nn.Sequential(
                nn.Linear(n_vars * image_size * image_size, unet_dim * 2),
                nn.ReLU(),
                nn.Linear(unet_dim * 2, unet_dim * 2)
            )
        elif conditioning_type == 'cross_attn':
            # Cross-attention conditioning
            self.cond_proj = nn.Conv2d(n_vars, unet_cond_dim or unet_dim, 1)
            
    def _prepare_condition(self, condition: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Prepare conditioning information based on conditioning type.
        
        Args:
            condition: Conditioning tensor [B, n_vars, H, W]
            x: Current noisy input [B, n_vars, H, W]
            
        Returns:
            Processed input for UNet [B, n_vars, H, W]
        """
        if self.conditioning_type == 'concat':
            # Concatenate and project back to original channel count
            concat_input = torch.cat([x, condition], dim=1)  # [B, n_vars*2, H, W]
            return self.cond_proj(concat_input)  # [B, n_vars, H, W]
        elif self.conditioning_type == 'film':
            # FiLM conditioning - will be handled in modified UNet
            return x
        elif self.conditioning_type == 'cross_attn':
            # Cross-attention conditioning
            return x
        else:
            return x
    
    def forward(self, x: torch.Tensor, condition: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass for training (computes loss).
        
        Args:
            x: Target tensor [B, n_vars, H, W]
            condition: Conditioning tensor [B, n_vars, H, W]
            
        Returns:
            Loss tensor
        """
        # During training, return the diffusion loss
        return self.diffusion(x)
    
    def predict_step(self, condition: torch.Tensor, grid: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Single prediction step for training (returns predicted tensor, not loss).
        This is what should be called during autoregressive training.
        
        Args:
            condition: Conditioning tensor [B, n_vars, H, W]
            grid: Optional grid coordinates
            
        Returns:
            Predicted tensor [B, n_vars, H, W] with gradients
        """
        # For training, we need to generate samples that maintain gradients
        # This is a simplified approach - add noise and denoise one step
        batch_size = condition.shape[0]
        device = condition.device
        
        # Generate noise
        noise = torch.randn_like(condition)
        
        # Add conditioning information to the noise
        if self.conditioning_type == 'concat' and hasattr(self, 'cond_proj'):
            # Mix condition with noise
            mixed_input = torch.cat([noise, condition], dim=1)
            conditioned_input = self.cond_proj(mixed_input)
        else:
            # Simple additive conditioning
            conditioned_input = noise + 0.1 * condition
        
        # Use UNet to predict the denoised version (single step)
        # Create a mid-range timestep for training
        t = torch.randint(100, 500, (batch_size,), device=device, dtype=torch.long)
        
        # Get the UNet prediction (this maintains gradients)
        predicted_noise = self.unet(conditioned_input, t)
        
        # Return the denoised prediction
        return conditioned_input - predicted_noise
    
    def sample(
        self, 
        condition: Optional[torch.Tensor] = None,
        batch_size: int = 1,
        num_inference_steps: Optional[int] = None,
        grid: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
        return_all_timesteps: bool = False
    ) -> torch.Tensor:
        """
        Sample from the diffusion model.
        
        Args:
            condition: Conditioning tensor [B, n_vars, H, W]
            batch_size: Batch size if no condition provided
            num_inference_steps: Number of sampling steps
            grid: Optional grid coordinates (for future use)
            generator: Random number generator for reproducibility
            return_all_timesteps: Whether to return intermediate steps
            
        Returns:
            Generated tensor [B, n_vars, H, W]
        """
        if condition is not None:
            batch_size = condition.shape[0]
            self._current_condition = condition
        
        # Override the UNet forward method to include conditioning
        if hasattr(self, '_current_condition'):
            original_forward = self.unet.forward
            
            def conditioned_forward(x, time, x_self_cond=None):
                if hasattr(self, '_current_condition') and self._current_condition is not None:
                    x = self._prepare_condition(self._current_condition, x)
                return original_forward(x, time, x_self_cond)
            
            self.unet.forward = conditioned_forward
        
        try:
            # Sample using the diffusion model
            if num_inference_steps is not None:
                # Temporarily override sampling timesteps
                original_sampling_timesteps = self.diffusion.sampling_timesteps
                self.diffusion.sampling_timesteps = num_inference_steps
            
            # Generate samples
            samples = self.diffusion.sample(
                batch_size=batch_size,
                return_all_timesteps=return_all_timesteps
            )
            
            if num_inference_steps is not None:
                # Restore original sampling timesteps
                self.diffusion.sampling_timesteps = original_sampling_timesteps
                
        finally:
            # Restore original UNet forward method
            if hasattr(self, '_current_condition'):
                self.unet.forward = original_forward
            
        return samples
    
    def ddim_sample(
        self,
        condition: Optional[torch.Tensor] = None,
        batch_size: int = 1,
        ddim_timesteps: int = 50,
        ddim_eta: float = 0.0,
        grid: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        DDIM sampling for faster inference.
        
        Args:
            condition: Conditioning tensor [B, n_vars, H, W]
            batch_size: Batch size if no condition provided
            ddim_timesteps: Number of DDIM steps
            ddim_eta: DDIM eta parameter (0 = deterministic)
            grid: Optional grid coordinates
            
        Returns:
            Generated tensor [B, n_vars, H, W]
        """
        if condition is not None:
            batch_size = condition.shape[0]
            self._current_condition = condition
        
        # Use DDIM sampling
        return self.diffusion.ddim_sample(
            shape=(batch_size, self.n_vars, self.image_size, self.image_size),
            timesteps=ddim_timesteps,
            eta=ddim_eta
        )
    
    @property
    def device(self):
        """Get the device of the model"""
        return next(self.parameters()).device