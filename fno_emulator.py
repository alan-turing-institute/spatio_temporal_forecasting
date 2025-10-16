"""Multi-variable Fourier Neural Operator for spatiotemporal fields."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

from neuralop.models import FNO


class MultivariableFNO(nn.Module):
    """
    Multi-variable Fourier Neural Operator for MHD simulations.
    This is not in AutoEmulate style so I prefer to just have a forward I will put training loop outsode of each model

    (B, C, H, W) 
    This implementation handles multiple physical variables (like density, temperature, 
    electric potential) as separate channels and performs 2D spatial convolutions.
    - Takes 2D spatial fields at one time instant
    - Processes multiple physical variables (density, temperature) as channels
    - Performs spatial Fourier transforms only
    - No temporal modeling - just spatial relationships
    so one can use this if they only want to predict the next time step and not worry about the whole time series.

    1 step - >1 step
    """
    
    def __init__(
        self,
        n_vars: int,
        n_modes: Tuple[int, int] = (16, 16),
        hidden_channels: int = 64,
        projection_channels: int = 128,
        n_layers: int = 4,
        use_skip_connections: bool = True,
        lifting_channels: int = 256,
        factorization: str = 'tucker',
        rank: float = 0.42
    ):
        """
        Args:
            n_vars: Number of physical variables (e.g., 2 for density and temperature)
            n_modes: Number of Fourier modes to keep in each spatial dimension
            hidden_channels: Hidden dimension size
            projection_channels: Projection layer dimension
            n_layers: Number of Fourier layers
            use_skip_connections: Whether to use skip connections
            lifting_channels: Lifting layer dimension
            factorization: Type of tensor factorization ('tucker', 'cp', 'tt')
            rank: Rank for tensor factorization
        """
        super().__init__()
        
        self.n_vars = n_vars
        self.n_modes = n_modes
        self.hidden_channels = hidden_channels
        self.n_layers = n_layers
        self.use_skip_connections = use_skip_connections
        
        # Use neuralop's FNO as the backbone
        # Input: [batch, n_vars, height, width]
        #  handle temporal dimension separately in autoregressive manner
        self.fno = FNO(
            n_modes=n_modes,
            hidden_channels=hidden_channels,
            in_channels=n_vars,
            out_channels=n_vars,
            lifting_channels=lifting_channels,
            projection_channels=projection_channels,
            n_layers=n_layers,
            use_mlp=True,
            mlp={'expansion': 0.5, 'dropout': 0.0},
            non_linearity=F.gelu,
            norm=None,
            preactivation=False,
            fno_skip='linear',
            mlp_skip='soft-gating',
            separable=False,
            factorization=factorization,
            rank=rank,
            joint_factorization=False,
            fixed_rank_modes=False,
            implementation='factorized',
            decomposition_kwargs={},
            domain_padding=None,
            domain_padding_mode='one-sided',
            fft_norm='forward'
        )
        
        # Additional processing layers for multi-variable interaction
        # is this making things worse? 
        if use_skip_connections:
            self.skip_connection = nn.Conv2d(n_vars, n_vars, kernel_size=1)
        
    def forward(self, x: torch.Tensor, grid: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through the FNO.
        
        Args:
            x: Input tensor of shape [batch, n_vars, height, width]
            grid: Optional grid coordinates [batch, 2, height, width]
            
        Returns:
            Output tensor of shape [batch, n_vars, height, width]
        """
        batch_size, n_vars, H, W = x.shape
        
        # Store input for skip connection
        x_skip = x
        
        # Main FNO forward pass
        out = self.fno(x, grid=grid)
        
        # Apply skip connection if enabled
        if self.use_skip_connections:
            #skip = self.skip_connection(x_skip) turining this off for testing 
            skip = x_skip
            out = out + skip
        
        return out

