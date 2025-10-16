"""Autoregressive wrapper for FNO models."""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Optional

from fno_emulator import MultivariableFNO
from AutoregressiveBase import AutoregressiveBase


class AutoregressiveFNO(AutoregressiveBase):
    """
    Autoregressive wrapper for FNO that handles temporal evolution.
    
    This class implements the autoregressive rollout for FNO models,
    where the model predicts future timesteps and feeds them back as input.
    """
    
    def __init__(
        self,
        fno_model: MultivariableFNO,
        t_in: int = 5,
        t_out: int = 10,
        step_size: int = 1,
        teacher_forcing_ratio: float = 0.5
    ):
        """
        Args:
            fno_model: The underlying FNO model
            t_in: Number of input timesteps
            t_out: Number of output timesteps to predict
            step_size: Number of timesteps to predict in each forward pass
            teacher_forcing_ratio: Ratio of teacher forcing during training
        """
        # Initialize base class with FNO's n_vars
        super().__init__(
            n_vars=fno_model.n_vars,
            t_in=t_in,
            t_out=t_out,
            step_size=step_size,
            teacher_forcing_ratio=teacher_forcing_ratio
        )
        
        # Store the FNO model
        self.fno = fno_model
    
    def _spatial_forward(self, x: torch.Tensor, grid: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        FNO-specific spatial processing.
        
        Args:
            x: Spatial tensor [B, n_vars, H, W]
            grid: Optional grid coordinates
            
        Returns:
            Processed tensor [B, n_vars, H, W]
        """
        return self.fno(x, grid)

