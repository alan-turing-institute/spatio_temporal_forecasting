"""Autoregressive wrapper for diffusion models."""

import torch
from typing import Optional

from AutoregressiveBase import AutoregressiveBase
from diffusion_emulator import SpatialDiffusionModel


class AutoregressiveDiffusion(AutoregressiveBase):
    """
    Autoregressive wrapper for diffusion models that handles temporal evolution.
    
    This class implements autoregressive rollout for diffusion models, where each
    timestep is generated through a diffusion process conditioned on previous timesteps.
    """
    
    def __init__(
        self,
        diffusion_model: SpatialDiffusionModel,
        t_in: int = 5,
        t_out: int = 10,
        step_size: int = 1,
        teacher_forcing_ratio: float = 0.5,
        num_inference_steps: int = 50,
        guidance_scale: float = 1.0,
        conditioning_strength: float = 0.1
    ):
        """
        Args:
            diffusion_model: The underlying diffusion model
            t_in: Number of input timesteps
            t_out: Number of output timesteps to predict
            step_size: Number of timesteps to predict in each forward pass
            teacher_forcing_ratio: Ratio of teacher forcing during training
            num_inference_steps: Number of denoising steps for sampling
            guidance_scale: Strength of guidance (if using classifier-free guidance)
            conditioning_strength: How strongly to condition on previous timesteps
        """
        # Initialize base class with diffusion model's n_vars
        super().__init__(
            n_vars=diffusion_model.n_vars,
            t_in=t_in,
            t_out=t_out,
            step_size=step_size,
            teacher_forcing_ratio=teacher_forcing_ratio
        )
        
        # Store diffusion-specific parameters
        self.diffusion = diffusion_model
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.conditioning_strength = conditioning_strength
        
        # For reproducible sampling during evaluation
        self.generator = torch.Generator()
        
    def _spatial_forward(self, x: torch.Tensor, grid: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Diffusion-specific spatial processing with sampling.
        
        Args:
            x: Spatial tensor to condition on [B, n_vars, H, W]
            grid: Optional grid coordinates [B, 2, H, W]
            
        Returns:
            Generated tensor [B, n_vars, H, W]
        """
        if self.training:
            # During training, use the gradient-friendly prediction method
            return self.diffusion.predict_step(condition=x, grid=grid)
        else:
            # During inference, use full sampling
            inference_steps = self.num_inference_steps
            
            # Sample from diffusion model conditioned on input
            generated = self.diffusion.sample(
                condition=x,
                num_inference_steps=inference_steps,
                grid=grid,
                generator=self.generator
            )
            
            return generated
    
    def set_seed(self, seed: int):
        """Set random seed for reproducible sampling"""
        self.generator.manual_seed(seed)