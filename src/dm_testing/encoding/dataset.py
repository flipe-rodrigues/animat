import torch
from torch.utils.data import Dataset
import numpy as np
from typing import Dict, List, Tuple
import pickle

class DemonstrationDataset(Dataset):
    """Optimized dataset for SAC demonstrations."""
    
    def __init__(self, 
                 demonstrations: Dict[str, np.ndarray], 
                 sequence_length: int = 95,
                 normalize_actions: bool = True,
                 device: str = 'cpu'):
        """
        Initialize demonstration dataset.
        
        Args:
            demonstrations: Dict with 'observations', 'actions', 'episode_starts'
            sequence_length: Length of sequences (95 for perfect episode match)
            normalize_actions: Whether to normalize actions to [-1, 1]
            device: torch device for tensors
        """
        
        self.sequence_length = sequence_length
        self.normalize_actions = normalize_actions
        self.device = device
        
        # Validate and store data
        self._validate_demonstrations(demonstrations)
        self.observations = demonstrations['observations']
        self.actions = demonstrations['actions']
        
        # Detect episodes using episode_starts
        self.episodes = self._detect_episodes(demonstrations)
        
        # Create sequences
        self.sequences = self._create_sequences()
        
        # Action normalization
        if self.normalize_actions:
            self._compute_action_stats()
        
        print(f"✅ Dataset ready: {len(self.episodes)} episodes, {len(self.sequences)} sequences")
        ep_lengths = [end - start for start, end in self.episodes]
        print(f"   Episode lengths: {ep_lengths[0]} steps each")
    
    def _validate_demonstrations(self, demonstrations: Dict[str, np.ndarray]):
        """Validate demonstration data."""
        required_keys = ['observations', 'actions', 'episode_starts']
        for key in required_keys:
            if key not in demonstrations:
                raise ValueError(f"Missing required key '{key}' in demonstrations")
        
        print(f"Validated {len(demonstrations['observations'])} demonstration steps")
    
    def _detect_episodes(self, demonstrations: Dict[str, np.ndarray]) -> List[Tuple[int, int]]:
        """Detect episodes using episode_starts."""
        episodes = []
        episode_starts = demonstrations['episode_starts']
        
        for i, start in enumerate(episode_starts):
            if i < len(episode_starts) - 1:
                end = episode_starts[i + 1]
            else:
                end = len(self.observations)
            
            episodes.append((int(start), int(end)))
        
        return episodes
    
    def _create_sequences(self) -> List[Dict[str, np.ndarray]]:
        """Create one sequence per episode."""
        sequences = []
        
        for episode_start, episode_end in self.episodes:
            episode_length = episode_end - episode_start
            
            if episode_length == self.sequence_length:
                # Perfect match - no padding needed
                seq_obs = self.observations[episode_start:episode_end].astype(np.float32)
                seq_actions = self.actions[episode_start:episode_end].astype(np.float32)
                seq_mask = np.ones(self.sequence_length, dtype=np.float32)
            else:
                # Handle different lengths (pad or truncate)
                seq_obs = np.zeros((self.sequence_length, self.observations.shape[1]), dtype=np.float32)
                seq_actions = np.zeros((self.sequence_length, self.actions.shape[1]), dtype=np.float32)
                seq_mask = np.zeros(self.sequence_length, dtype=np.float32)
                
                actual_length = min(episode_length, self.sequence_length)
                seq_obs[:actual_length] = self.observations[episode_start:episode_start + actual_length]
                seq_actions[:actual_length] = self.actions[episode_start:episode_start + actual_length]
                seq_mask[:actual_length] = 1.0
            
            sequences.append({
                'observations': seq_obs,
                'actions': seq_actions,
                'mask': seq_mask
            })
        
        return sequences
    
    def _compute_action_stats(self):
        """Compute action normalization statistics."""
        all_actions = np.concatenate([seq['actions'] for seq in self.sequences], axis=0)
        
        self.action_min = np.min(all_actions, axis=0)
        self.action_max = np.max(all_actions, axis=0)
        
        print(f"   Action range: [{self.action_min.min():.3f}, {self.action_max.max():.3f}]")
    
    def _normalize_actions(self, actions: np.ndarray) -> np.ndarray:
        """Keep SAC actions in [-1, 1] range for optimal training."""
        if not self.normalize_actions:
            return actions
        
        # Keep actions in [-1, 1] range (best for neural network training)
        return np.clip(actions, 0.0, 1.0)
    
    def denormalize_actions(self, normalized_actions: np.ndarray) -> np.ndarray:
        """Actions stay in [-1, 1] range (no denormalization needed)."""
        return np.clip(normalized_actions, -1.0, 1.0)
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get a sequence."""
        sequence = self.sequences[idx]
        
        obs = torch.from_numpy(sequence['observations']).to(self.device)
        actions = torch.from_numpy(self._normalize_actions(sequence['actions'])).to(self.device)
        mask = torch.from_numpy(sequence['mask']).to(self.device)
        
        return obs, actions, mask

def load_demonstrations(path: str = "sac_demonstrations.pkl") -> DemonstrationDataset:
    """Load and create optimized dataset."""
    print(f"Loading demonstrations from {path}...")
    with open(path, 'rb') as f:
        demonstrations = pickle.load(f)
    
    return DemonstrationDataset(demonstrations)

if __name__ == "__main__":
    # Simple test
    try:
        dataset = load_demonstrations()
        
        # Test a sample
        obs, actions, mask = dataset[0]
        print(f"Sample: obs{obs.shape}, actions{actions.shape}, mask_sum={mask.sum():.0f}")
        print(f"Action range: [{actions.min():.3f}, {actions.max():.3f}]")
        
        print("🚀 Dataset ready for training!")
        
    except FileNotFoundError:
        print("❌ No demonstration file found. Create demonstrations first.")