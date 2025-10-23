"""Base class for autoregressive temporal modeling."""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Optional


class AutoregressiveBase(nn.Module, ABC):
    """
    Abstract base class for autoregressive temporal modeling.
    Handles all the temporal logic, subclasses implement spatial processing.
    """
    
    def __init__(
        self,
        n_vars: int,
        t_in: int = 5,
        t_out: int = 10,
        step_size: int = 1,
        teacher_forcing_ratio: float = 0.5
    ):
        """
        Args:
            n_vars: Number of physical variables
            t_in: Number of input timesteps
            t_out: Number of output timesteps to predict
            step_size: Number of timesteps to predict in each forward pass
            teacher_forcing_ratio: Ratio of teacher forcing during training
        """
        super().__init__()
        
        self.n_vars = n_vars
        self.t_in = t_in
        self.t_out = t_out
        self.step_size = step_size
        self.teacher_forcing_ratio = teacher_forcing_ratio
        
        # Setup common temporal processing layers
        self._setup_temporal_layers()
        
    def _setup_temporal_layers(self):
        """Setup temporal encoding layers - common for all models"""
        # Temporal encoding layers
        self.temporal_encoder = nn.Conv3d(
            self.n_vars, 
            self.n_vars, 
            kernel_size=(self.t_in, 1, 1),
            padding=(0, 0, 0)
        )
        
        # Output projection to handle step_size predictions
        if self.step_size > 1:
            self.output_projection = nn.Conv3d(
                self.n_vars,
                self.n_vars * self.step_size,
                kernel_size=(1, 1, 1)
            )
    
    @abstractmethod
    def _spatial_forward(self, x: torch.Tensor, grid: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Abstract method for spatial processing - each subclass implements their specific model.
        
        Args:
            x: Spatial tensor [B, n_vars, H, W]
            grid: Optional grid coordinates
            
        Returns:
            Processed tensor [B, n_vars, H, W]
        """
        pass
    
    def forward(self, x, target=None, grid=None):
        """
        Core autoregressive forward pass - same logic for all spatial models.
        
        Args:
            x: Input sequence [B, t_in, n_vars, H, W]
            target: Target sequence for teacher forcing [B, t_out, n_vars, H, W]
            grid: Optional grid coordinates
            
        Returns:
            Predicted sequence [B, t_out, n_vars, H, W]
        """
        batch_size, t_in, n_vars, H, W = x.shape
        device = x.device

        predictions = []
        current_input = x

        n_steps = (self.t_out + self.step_size - 1) // self.step_size

        # Decide once per sequence whether to use teacher forcing
        use_teacher_forcing = False
        if self.training and target is not None:
            use_teacher_forcing = torch.rand(1).item() < self.teacher_forcing_ratio

        for step in range(n_steps):
            # Temporal encoding
            temp_input = current_input.permute(0, 2, 1, 3, 4)
            temp_encoded = self.temporal_encoder(temp_input)  # [B, n_vars, 1, H, W]
            temp_encoded = temp_encoded.squeeze(2)            # [B, n_vars, H, W]

            # Spatial processing - delegated to subclass
            spatial_output = self._spatial_forward(temp_encoded, grid)  # [B, n_vars, H, W]

            # Handle multi-step prediction
            if self.step_size > 1:
                spatial_output = spatial_output.unsqueeze(2)          # [B, n_vars, 1, H, W]
                multi_step = self.output_projection(spatial_output)   # [B, n_vars*step_size, 1, H, W]
                multi_step = multi_step.squeeze(2)                    # [B, n_vars*step_size, H, W]
                multi_step = multi_step.view(batch_size, n_vars, self.step_size, H, W)
                multi_step = multi_step.permute(0, 2, 1, 3, 4)       # [B, step_size, n_vars, H, W]
            else:
                multi_step = spatial_output.unsqueeze(1)              # [B, 1, n_vars, H, W]

            predictions.append(multi_step)

            # Prepare next input for autoregressive rollout
            start_idx = step * self.step_size
            end_idx = min(start_idx + self.step_size, self.t_out)
            actual_steps = end_idx - start_idx

            if step < n_steps - 1:
                if use_teacher_forcing:
                    next_input = target[:, start_idx:start_idx + actual_steps]
                else:
                    next_input = multi_step[:, :actual_steps]

                current_input = torch.cat(
                    [current_input[:, actual_steps:], next_input], dim=1
                )

        full_predictions = torch.cat(predictions, dim=1)
        return full_predictions[:, :self.t_out]
    
    def predict(self, x: torch.Tensor, n_steps: int, grid: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Pure inference without teacher forcing.

        Args:
            x: Input sequence [B, t_in, n_vars, H, W]
            n_steps: Number of steps to predict
            grid: Optional grid coordinates

        Returns:
            Predicted sequence [B, n_steps, n_vars, H, W]
        """
        assert x.ndim == 5, f"Expected [B, t_in, n_vars, H, W], got {x.shape}"
        assert x.shape[1] == self.t_in, f"Input t_in={x.shape[1]} does not match model.t_in={self.t_in}"

        self.eval()
        original_t_out = self.t_out
        self.t_out = n_steps

        with torch.no_grad():
            predictions = self.forward(x, target=None, grid=grid)

        self.t_out = original_t_out
        assert predictions.ndim == 5, f"Predictions shape wrong: {predictions.shape}"
        return predictions

