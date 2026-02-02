"""Core type definitions for structured data passing."""

from dataclasses import dataclass
from typing import Optional
import numpy as np
import torch


@dataclass
class Proprioception:
    """Proprioceptive observations (numpy arrays)."""
    lengths: np.ndarray     # [num_muscles]
    velocities: np.ndarray  # [num_muscles]
    forces: np.ndarray      # [num_muscles]

    @property
    def num_muscles(self) -> int:
        return len(self.lengths)

    def to_flat(self) -> np.ndarray:
        return np.concatenate([self.lengths, self.velocities, self.forces])

    @classmethod
    def from_flat(cls, flat: np.ndarray, num_muscles: int) -> "Proprioception":
        return cls(
            lengths=flat[:num_muscles],
            velocities=flat[num_muscles : 2 * num_muscles],
            forces=flat[2 * num_muscles : 3 * num_muscles],
        )

    def to_tensor(self, device: Optional[torch.device] = None) -> "ProprioceptionTensor":
        return ProprioceptionTensor(
            lengths=torch.as_tensor(self.lengths, dtype=torch.float32, device=device),
            velocities=torch.as_tensor(self.velocities, dtype=torch.float32, device=device),
            forces=torch.as_tensor(self.forces, dtype=torch.float32, device=device),
        )


@dataclass
class ProprioceptionTensor:
    """Proprioceptive observations (torch tensors) for neural network forward passes."""
    lengths: torch.Tensor     # [batch, num_muscles]
    velocities: torch.Tensor  # [batch, num_muscles]
    forces: torch.Tensor      # [batch, num_muscles]

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
        return torch.cat([self.lengths, self.velocities, self.forces], dim=-1)

    @classmethod
    def from_flat(cls, flat: torch.Tensor, num_muscles: int) -> "ProprioceptionTensor":
        return cls(
            lengths=flat[..., :num_muscles],
            velocities=flat[..., num_muscles : 2 * num_muscles],
            forces=flat[..., 2 * num_muscles : 3 * num_muscles],
        )


@dataclass
class Observation:
    """Complete observation from environment (numpy)."""
    proprio: Proprioception
    target: np.ndarray

    def to_flat(self) -> np.ndarray:
        return np.concatenate([self.proprio.to_flat(), self.target])

    @classmethod
    def from_flat(cls, flat: np.ndarray, num_muscles: int) -> "Observation":
        proprio_dim = num_muscles * 3
        return cls(
            proprio=Proprioception.from_flat(flat[:proprio_dim], num_muscles),
            target=flat[proprio_dim : proprio_dim + 3],
        )

    def to_tensor(self, device: Optional[torch.device] = None) -> "ObservationTensor":
        return ObservationTensor(
            proprio=self.proprio.to_tensor(device),
            target=torch.as_tensor(self.target, dtype=torch.float32, device=device),
        )


@dataclass
class ObservationTensor:
    """Complete observation (torch tensors)."""
    proprio: ProprioceptionTensor
    target: torch.Tensor  # [batch, 3]

    @property
    def batch_size(self) -> int:
        return self.proprio.batch_size

    @property
    def device(self) -> torch.device:
        return self.proprio.device

    def to_flat(self) -> torch.Tensor:
        return torch.cat([self.proprio.to_flat(), self.target], dim=-1)

    @classmethod
    def from_flat(cls, flat: torch.Tensor, num_muscles: int) -> "ObservationTensor":
        proprio_dim = num_muscles * 3
        return cls(
            proprio=ProprioceptionTensor.from_flat(flat[..., :proprio_dim], num_muscles),
            target=flat[..., proprio_dim : proprio_dim + 3],
        )

    @classmethod
    def from_numpy(cls, obs: Observation, device: Optional[torch.device] = None) -> "ObservationTensor":
        """Create batched tensor from numpy Observation."""
        return cls(
            proprio=ProprioceptionTensor(
                lengths=torch.as_tensor(obs.proprio.lengths, dtype=torch.float32, device=device).unsqueeze(0),
                velocities=torch.as_tensor(obs.proprio.velocities, dtype=torch.float32, device=device).unsqueeze(0),
                forces=torch.as_tensor(obs.proprio.forces, dtype=torch.float32, device=device).unsqueeze(0),
            ),
            target=torch.as_tensor(obs.target, dtype=torch.float32, device=device).unsqueeze(0),
        )
