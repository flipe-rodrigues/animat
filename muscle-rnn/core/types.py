"""
Core type definitions for structured data passing.

These types avoid implicit array ordering by using named fields.
"""

from dataclasses import dataclass
from typing import Dict, Optional
import numpy as np
import torch


@dataclass
class Proprioception:
    """
    Structured proprioceptive observations.
    
    Keeps muscle length, velocity, and force as separate named arrays
    instead of concatenating into implicit positions.
    """
    lengths: np.ndarray    # [num_muscles] - muscle lengths
    velocities: np.ndarray # [num_muscles] - muscle velocities  
    forces: np.ndarray     # [num_muscles] - muscle forces
    
    @property
    def num_muscles(self) -> int:
        return len(self.lengths)
    
    def to_flat(self) -> np.ndarray:
        """Concatenate to flat array (only when required by API)."""
        return np.concatenate([self.lengths, self.velocities, self.forces])
    
    @classmethod
    def from_flat(cls, flat: np.ndarray, num_muscles: int) -> 'Proprioception':
        """Parse from flat array (for legacy compatibility)."""
        return cls(
            lengths=flat[:num_muscles],
            velocities=flat[num_muscles:2*num_muscles],
            forces=flat[2*num_muscles:3*num_muscles],
        )
    
    def to_tensor(self, device: torch.device = None) -> 'ProprioceptionTensor':
        """Convert to tensor version."""
        return ProprioceptionTensor(
            lengths=torch.tensor(self.lengths, dtype=torch.float32, device=device),
            velocities=torch.tensor(self.velocities, dtype=torch.float32, device=device),
            forces=torch.tensor(self.forces, dtype=torch.float32, device=device),
        )


@dataclass
class ProprioceptionTensor:
    """
    Tensor version of proprioceptive observations.
    
    Used within neural network forward passes.
    """
    lengths: torch.Tensor    # [batch, num_muscles]
    velocities: torch.Tensor # [batch, num_muscles]  
    forces: torch.Tensor     # [batch, num_muscles]
    
    @property
    def batch_size(self) -> int:
        return self.lengths.shape[0]
    
    @property
    def num_muscles(self) -> int:
        return self.lengths.shape[-1]
    
    @property
    def device(self) -> torch.device:
        return self.lengths.device
    
    def to_flat(self) -> torch.Tensor:
        """Concatenate to flat tensor."""
        return torch.cat([self.lengths, self.velocities, self.forces], dim=-1)
    
    @classmethod
    def from_flat(cls, flat: torch.Tensor, num_muscles: int) -> 'ProprioceptionTensor':
        """Parse from flat tensor."""
        return cls(
            lengths=flat[..., :num_muscles],
            velocities=flat[..., num_muscles:2*num_muscles],
            forces=flat[..., 2*num_muscles:3*num_muscles],
        )


@dataclass 
class Observation:
    """
    Complete structured observation from environment.
    
    Contains all observation modalities as named fields.
    """
    proprio: Proprioception  # Proprioceptive sensors
    target: np.ndarray       # Target position [3] (XYZ)
    
    def to_flat(self) -> np.ndarray:
        """Concatenate to flat array (for gym API compatibility)."""
        return np.concatenate([self.proprio.to_flat(), self.target])
    
    @classmethod
    def from_flat(cls, flat: np.ndarray, num_muscles: int) -> 'Observation':
        """Parse from flat array."""
        proprio_dim = num_muscles * 3
        return cls(
            proprio=Proprioception.from_flat(flat[:proprio_dim], num_muscles),
            target=flat[proprio_dim:proprio_dim + 3],
        )
    
    def to_tensor(self, device: torch.device = None) -> 'ObservationTensor':
        """Convert to tensor version."""
        return ObservationTensor(
            proprio=self.proprio.to_tensor(device),
            target=torch.tensor(self.target, dtype=torch.float32, device=device),
        )


@dataclass
class ObservationTensor:
    """
    Tensor version of structured observation.
    """
    proprio: ProprioceptionTensor  # [batch, ...]
    target: torch.Tensor           # [batch, 3]
    
    @property
    def batch_size(self) -> int:
        return self.proprio.batch_size
    
    @property 
    def device(self) -> torch.device:
        return self.proprio.device
    
    def to_flat(self) -> torch.Tensor:
        """Concatenate to flat tensor."""
        return torch.cat([self.proprio.to_flat(), self.target], dim=-1)
    
    @classmethod
    def from_flat(cls, flat: torch.Tensor, num_muscles: int) -> 'ObservationTensor':
        """Parse from flat tensor."""
        proprio_dim = num_muscles * 3
        return cls(
            proprio=ProprioceptionTensor.from_flat(flat[..., :proprio_dim], num_muscles),
            target=flat[..., proprio_dim:proprio_dim + 3],
        )
    
    @classmethod
    def from_numpy(cls, obs: Observation, device: torch.device = None) -> 'ObservationTensor':
        """Create from numpy Observation with batch dimension."""
        return cls(
            proprio=ProprioceptionTensor(
                lengths=torch.tensor(obs.proprio.lengths, dtype=torch.float32, device=device).unsqueeze(0),
                velocities=torch.tensor(obs.proprio.velocities, dtype=torch.float32, device=device).unsqueeze(0),
                forces=torch.tensor(obs.proprio.forces, dtype=torch.float32, device=device).unsqueeze(0),
            ),
            target=torch.tensor(obs.target, dtype=torch.float32, device=device).unsqueeze(0),
        )
