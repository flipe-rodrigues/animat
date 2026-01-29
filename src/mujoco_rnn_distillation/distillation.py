"""
Behavioral Cloning / Distillation for RNN.

This module implements behavioral cloning to distill knowledge from
a trained MLP teacher policy into an RNN student policy.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Optional, Tuple
import logging
from collections import deque

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RNNPolicyWrapper(nn.Module):
    """
    PyTorch wrapper around your custom RNN implementations.
    
    This allows your numpy-based RNNs to be trained with PyTorch
    optimizers and automatic differentiation.
    """
    
    def __init__(self, rnn_numpy):
        """
        Wrap a numpy-based RNN for PyTorch training.
        
        Args:
            rnn_numpy: Your RNN instance (AlphaOnlyRNN or FullRNN)
        """
        super().__init__()
        
        self.rnn_numpy = rnn_numpy
        self.obs_dim = (
            rnn_numpy.target_size +
            rnn_numpy.length_size +
            rnn_numpy.velocity_size +
            rnn_numpy.force_size
        )
        self.action_dim = rnn_numpy.output_size
        
        # Convert RNN parameters to PyTorch parameters
        self._create_torch_parameters()
        
        # Hidden state (not a parameter, just state)
        self.register_buffer('h_hidden', torch.zeros(rnn_numpy.hidden_size) if hasattr(rnn_numpy, 'hidden_size') else None)
        self.register_buffer('a_hidden', torch.zeros(rnn_numpy.output_size))
        if hasattr(rnn_numpy, 'gs'):
            self.register_buffer('gs_hidden', torch.zeros(rnn_numpy.output_size))
            self.register_buffer('gd_hidden', torch.zeros(rnn_numpy.output_size))
    
    def _create_torch_parameters(self):
        """Convert numpy parameters to PyTorch parameters"""
        # Store parameter structure
        self.param_shapes = {}
        self.param_slices = {}
        
        idx = 0
        for name, weight in self.rnn_numpy.weights.items():
            size = weight.size
            self.param_shapes[f"weight_{name}"] = weight.shape
            self.param_slices[f"weight_{name}"] = (idx, idx + size)
            idx += size
        
        for name, bias in self.rnn_numpy.biases.items():
            size = bias.size
            self.param_shapes[f"bias_{name}"] = bias.shape
            self.param_slices[f"bias_{name}"] = (idx, idx + size)
            idx += size
        
        # Create single parameter vector
        initial_params = torch.FloatTensor(self.rnn_numpy.get_params())
        self.params = nn.Parameter(initial_params)
    
    def get_weight_tensor(self, name: str) -> torch.Tensor:
        """Extract weight matrix from parameter vector"""
        key = f"weight_{name}"
        start, end = self.param_slices[key]
        shape = self.param_shapes[key]
        return self.params[start:end].view(shape)
    
    def get_bias_tensor(self, name: str) -> torch.Tensor:
        """Extract bias vector from parameter vector"""
        key = f"bias_{name}"
        start, end = self.param_slices[key]
        shape = self.param_shapes[key]
        return self.params[start:end].view(shape)
    
    def reset_state(self):
        """Reset hidden state"""
        if self.h_hidden is not None:
            self.h_hidden.zero_()
        self.a_hidden.zero_()
        if hasattr(self, 'gs_hidden') and self.gs_hidden is not None:
            self.gs_hidden.zero_()
            self.gd_hidden.zero_()
    
    def forward(self, obs: torch.Tensor, reset_state: bool = False) -> torch.Tensor:
        """
        Forward pass through RNN.
        
        Args:
            obs: Observation [batch_size, obs_dim] or [obs_dim]
            reset_state: Whether to reset hidden state before forward pass
        
        Returns:
            action: Action output [batch_size, action_dim] or [action_dim]
        """
        # Handle batch dimension
        single_input = (obs.dim() == 1)
        if single_input:
            obs = obs.unsqueeze(0)
        
        batch_size = obs.shape[0]
        
        if reset_state:
            self.reset_state()
        
        # Split observation
        idx = 0
        tgt_obs = obs[:, idx:idx + self.rnn_numpy.target_size]
        idx += self.rnn_numpy.target_size
        len_obs = obs[:, idx:idx + self.rnn_numpy.length_size]
        idx += self.rnn_numpy.length_size
        vel_obs = obs[:, idx:idx + self.rnn_numpy.velocity_size]
        idx += self.rnn_numpy.velocity_size
        frc_obs = obs[:, idx:idx + self.rnn_numpy.force_size]
        
        # Implement RNN forward pass in PyTorch
        # This is a reimplementation of your RNN's step() method
        actions = []
        
        for i in range(batch_size):
            action = self._step_single(
                tgt_obs[i],
                len_obs[i],
                vel_obs[i],
                frc_obs[i],
            )
            actions.append(action)
        
        actions = torch.stack(actions)
        
        if single_input:
            actions = actions.squeeze(0)
        
        return actions
    
    def _step_single(
        self,
        tgt_obs: torch.Tensor,
        len_obs: torch.Tensor,
        vel_obs: torch.Tensor,
        frc_obs: torch.Tensor,
    ) -> torch.Tensor:
        """Single RNN step - implement based on your RNN type"""
        
        # Check if this is AlphaOnlyRNN or FullRNN
        if hasattr(self.rnn_numpy, 'hidden_size'):
            return self._step_full_rnn(tgt_obs, len_obs, vel_obs, frc_obs)
        else:
            return self._step_alpha_only(tgt_obs, len_obs, vel_obs, frc_obs)
    
    def _step_alpha_only(
        self,
        tgt_obs: torch.Tensor,
        len_obs: torch.Tensor,
        vel_obs: torch.Tensor,
        frc_obs: torch.Tensor,
    ) -> torch.Tensor:
        """AlphaOnlyRNN forward pass"""
        tgt2a = self.get_weight_tensor('tgt2a').abs()  # constraint
        
        a_input = tgt2a @ tgt_obs
        
        if self.rnn_numpy.use_bias:
            a_input = a_input + self.get_bias_tensor('a')
        
        # Update alpha with smoothing
        alpha = self.rnn_numpy.smoothing_factor
        self.a_hidden = (
            (1 - alpha) * self.a_hidden +
            alpha * torch.sigmoid(a_input)
        )
        
        return self.a_hidden
    
    def _step_full_rnn(
        self,
        tgt_obs: torch.Tensor,
        len_obs: torch.Tensor,
        vel_obs: torch.Tensor,
        frc_obs: torch.Tensor,
    ) -> torch.Tensor:
        """FullRNN forward pass"""
        # Get weights with constraints
        tgt2h = self.get_weight_tensor('tgt2h').abs()
        len2h = self.get_weight_tensor('len2h').abs()
        vel2h = self.get_weight_tensor('vel2h').abs()
        frc2h = self.get_weight_tensor('frc2h').abs()
        h2h = self.get_weight_tensor('h2h')
        a2h = self.get_weight_tensor('a2h').abs()
        h2gs = self.get_weight_tensor('h2gs')
        h2gd = self.get_weight_tensor('h2gd')
        len2a = self.get_weight_tensor('len2a')
        vel2a = self.get_weight_tensor('vel2a')
        h2a = self.get_weight_tensor('h2a')
        
        # Modulated inputs
        len_modulated = len_obs + self.gs_hidden
        vel_modulated = vel_obs * self.gd_hidden
        
        # Hidden layer input
        h_input = (
            tgt2h @ tgt_obs +
            len2h @ len_modulated +
            vel2h @ vel_modulated +
            frc2h @ frc_obs +
            h2h @ self.h_hidden +
            a2h @ self.a_hidden
        )
        
        if self.rnn_numpy.use_bias:
            h_input = h_input + self.get_bias_tensor('h')
        
        # Gamma inputs
        gs_input = h2gs @ self.h_hidden
        gd_input = h2gd @ self.h_hidden
        
        if self.rnn_numpy.use_bias:
            gs_input = gs_input + self.get_bias_tensor('gs')
            gd_input = gd_input + self.get_bias_tensor('gd')
        
        # Alpha input
        a_input = (
            h2a @ self.h_hidden +
            len2a @ len_modulated +
            vel2a @ vel_modulated
        )
        
        if self.rnn_numpy.use_bias:
            a_input = a_input + self.get_bias_tensor('a')
        
        # Update states
        alpha = self.rnn_numpy.smoothing_factor
        
        # Choose activation
        if self.rnn_numpy.activation.__name__ == 'relu':
            h_activation = torch.relu(h_input)
        elif self.rnn_numpy.activation.__name__ == 'tanh':
            h_activation = torch.tanh(h_input)
        else:
            h_activation = torch.sigmoid(h_input)
        
        self.h_hidden = (1 - alpha) * self.h_hidden + alpha * h_activation
        self.gs_hidden = (1 - alpha) * self.gs_hidden + alpha * torch.sigmoid(gs_input)
        self.gd_hidden = (1 - alpha) * self.gd_hidden + alpha * torch.sigmoid(gd_input)
        self.a_hidden = (1 - alpha) * self.a_hidden + alpha * torch.sigmoid(a_input)
        
        return self.a_hidden


class DistillationTrainer:
    """
    Train RNN student to imitate MLP teacher using behavioral cloning.
    """
    
    def __init__(
        self,
        teacher_policy,
        student_rnn,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    ):
        """
        Initialize distillation trainer.
        
        Args:
            teacher_policy: Trained MLPPolicy
            student_rnn: Your RNN instance to train
            learning_rate: Learning rate for student
            weight_decay: L2 regularization
            device: Device to train on
        """
        self.device = torch.device(device)
        
        # Teacher (frozen)
        self.teacher = teacher_policy.to(self.device)
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False
        
        # Student (trainable)
        self.student = RNNPolicyWrapper(student_rnn).to(self.device)
        self.optimizer = optim.Adam(
            self.student.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        
        # Training stats
        self.losses = deque(maxlen=100)
        self.update_count = 0
        
        logger.info(f"Initialized distillation trainer on {self.device}")
        logger.info(f"Student RNN: {sum(p.numel() for p in self.student.parameters())} parameters")
    
    def collect_teacher_demonstrations(
        self,
        env,
        n_episodes: int = 100,
        deterministic: bool = True,
    ) -> Dict[str, np.ndarray]:
        """
        Collect demonstrations from teacher policy.
        
        Args:
            env: StepBasedReachingEnv instance
            n_episodes: Number of episodes to collect
            deterministic: Whether teacher acts deterministically
        
        Returns:
            Dictionary with 'observations' and 'actions' arrays
        """
        logger.info(f"Collecting {n_episodes} teacher demonstrations...")
        
        all_observations = []
        all_actions = []
        
        for episode in range(n_episodes):
            obs = env.reset(seed=episode)
            done = False
            
            while not done:
                # Get teacher action
                with torch.no_grad():
                    obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                    action, _, _ = self.teacher.get_action(obs_tensor, deterministic=deterministic)
                    action = action.cpu().numpy().squeeze()
                
                # Store transition
                all_observations.append(obs)
                all_actions.append(action)
                
                # Step environment
                obs, reward, done, info = env.step(action)
            
            if (episode + 1) % 10 == 0:
                logger.info(f"Collected {episode + 1}/{n_episodes} episodes")
        
        dataset = {
            'observations': np.array(all_observations),
            'actions': np.array(all_actions),
        }
        
        logger.info(f"Dataset size: {len(dataset['observations'])} transitions")
        
        return dataset
    
    def train_on_demonstrations(
        self,
        dataset: Dict[str, np.ndarray],
        n_epochs: int = 100,
        batch_size: int = 64,
        validation_split: float = 0.1,
    ) -> Dict[str, List[float]]:
        """
        Train student RNN on teacher demonstrations.
        
        Args:
            dataset: Dictionary with 'observations' and 'actions'
            n_epochs: Number of training epochs
            batch_size: Batch size
            validation_split: Fraction of data for validation
        
        Returns:
            Dictionary of training history
        """
        logger.info("Starting behavioral cloning training...")
        
        # Split data
        n_samples = len(dataset['observations'])
        n_val = int(n_samples * validation_split)
        indices = np.random.permutation(n_samples)
        
        train_obs = dataset['observations'][indices[n_val:]]
        train_actions = dataset['actions'][indices[n_val:]]
        val_obs = dataset['observations'][indices[:n_val]]
        val_actions = dataset['actions'][indices[:n_val]]
        
        # Convert to tensors
        train_obs_tensor = torch.FloatTensor(train_obs).to(self.device)
        train_actions_tensor = torch.FloatTensor(train_actions).to(self.device)
        val_obs_tensor = torch.FloatTensor(val_obs).to(self.device)
        val_actions_tensor = torch.FloatTensor(val_actions).to(self.device)
        
        # Training history
        history = {
            'train_loss': [],
            'val_loss': [],
        }
        
        best_val_loss = float('inf')
        
        for epoch in range(n_epochs):
            # Training
            self.student.train()
            train_losses = []
            
            indices = np.random.permutation(len(train_obs_tensor))
            for start in range(0, len(train_obs_tensor), batch_size):
                end = start + batch_size
                batch_indices = indices[start:end]
                
                batch_obs = train_obs_tensor[batch_indices]
                batch_actions = train_actions_tensor[batch_indices]
                
                # Reset student state for each batch
                # Note: This treats each transition independently
                # For sequential training, you'd need to organize data into episodes
                self.optimizer.zero_grad()
                
                # Forward pass (with state reset for each batch)
                pred_actions = []
                self.student.reset_state()
                for obs in batch_obs:
                    pred_action = self.student(obs, reset_state=False)
                    pred_actions.append(pred_action)
                pred_actions = torch.stack(pred_actions)
                
                # MSE loss
                loss = nn.functional.mse_loss(pred_actions, batch_actions)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                train_losses.append(loss.item())
            
            # Validation
            self.student.eval()
            with torch.no_grad():
                self.student.reset_state()
                val_pred = []
                for obs in val_obs_tensor:
                    pred = self.student(obs, reset_state=False)
                    val_pred.append(pred)
                val_pred = torch.stack(val_pred)
                val_loss = nn.functional.mse_loss(val_pred, val_actions_tensor).item()
            
            # Record history
            train_loss = np.mean(train_losses)
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            
            # Logging
            if (epoch + 1) % 10 == 0:
                logger.info(
                    f"Epoch {epoch + 1}/{n_epochs} - "
                    f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}"
                )
            
            # Track best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.best_params = self.student.params.data.clone()
            
            self.update_count += 1
        
        logger.info(f"Training complete. Best val loss: {best_val_loss:.6f}")
        
        return history
    
    def export_to_numpy_rnn(self):
        """
        Export trained student parameters back to numpy RNN.
        
        Returns:
            Updated numpy RNN instance
        """
        # Get parameters from PyTorch
        params_numpy = self.student.params.detach().cpu().numpy()
        
        # Update numpy RNN
        rnn_numpy = self.student.rnn_numpy.copy()
        rnn_numpy.set_params(params_numpy)
        
        return rnn_numpy
    
    def save(self, path: str):
        """Save student checkpoint"""
        torch.save({
            'student_state_dict': self.student.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'update_count': self.update_count,
            'best_params': getattr(self, 'best_params', None),
        }, path)
        logger.info(f"Saved distillation checkpoint to {path}")
    
    def load(self, path: str):
        """Load student checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.student.load_state_dict(checkpoint['student_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.update_count = checkpoint['update_count']
        if 'best_params' in checkpoint and checkpoint['best_params'] is not None:
            self.best_params = checkpoint['best_params']
        logger.info(f"Loaded distillation checkpoint from {path}")
