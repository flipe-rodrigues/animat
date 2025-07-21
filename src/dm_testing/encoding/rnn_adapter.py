import torch
import torch.nn as nn
import os
import numpy as np
from policy import NumpyStyleRNNPolicy
from shimmy_wrapper import ModalitySpecificEncoder
import matplotlib.pyplot as plt

# ============================
# CONFIGURATION SECTION
# ============================
class Config:
    """Centralized configuration for all experiments"""
    
    # Model paths and parameters
    MODEL_NAME = "rnn_simple_noise_test"
    PYTORCH_MODEL_PATH = f"./{MODEL_NAME}/best_rnn_full.pth"
    SELECTIVE_VECNORM_PATH = f"./{MODEL_NAME}/selective_vec_normalize.pkl"
    
    # Model architecture - FIXED FOR 4×4 GRID
    HIDDEN_SIZE = 64
    ALPHA = 1
    OBS_DIM = 29  # 12 muscle + 16 grid (4×4) + 1 z = 29D
    ACTION_DIM = 4
    
    # Analysis parameters
    STIMULATION_UNITS = [6, 44]  # Units to analyze in detail
    TOP_N_UNITS = 6  # Number of top convergent units to plot
    
    # Output directory for plots
    OUTPUT_DIR = f"./analysis_results_{MODEL_NAME}"
    
    @classmethod
    def setup_output_dir(cls):
        """Create output directory if it doesn't exist"""
        os.makedirs(cls.OUTPUT_DIR, exist_ok=True)
        print(f"📁 Output directory: {cls.OUTPUT_DIR}")
        return cls.OUTPUT_DIR
    
    @classmethod
    def save_plot(cls, filename, dpi=150):
        """Save current matplotlib figure"""
        output_dir = cls.setup_output_dir()
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=dpi, bbox_inches='tight')
        print(f"💾 Saved plot: {filepath}")

# Initialize configuration
config = Config()

# Add this to your stimulation_bizzi.py
class RNNAdapter:
    """Adapter for NumpyStyleRNNPolicy with SelectiveVecNormalize"""
    
    def __init__(self, pytorch_policy_path=None, selective_vecnorm_path=None):
        # Use config defaults if not provided
        if pytorch_policy_path is None:
            pytorch_policy_path = config.PYTORCH_MODEL_PATH
        if selective_vecnorm_path is None:
            selective_vecnorm_path = config.SELECTIVE_VECNORM_PATH
            
        print(f"🔧 Loading model: {pytorch_policy_path}")
        print(f"🔧 Loading normalization: {selective_vecnorm_path}")
        
        # Load checkpoint with weights_only=False to fix PyTorch 2.6 issue
        try:
            checkpoint = torch.load(pytorch_policy_path, map_location='cpu', weights_only=False)
            print(f"✅ Successfully loaded checkpoint with weights_only=False")
        except Exception as e:
            print(f"❌ Error loading checkpoint: {e}")
            raise
        
        # Extract config from checkpoint
        if 'config' in checkpoint:
            model_config = checkpoint['config']
            obs_dim = model_config['obs_dim']
            action_dim = model_config['action_dim']
            hidden_size = model_config['hidden_size']
            print(f"📋 Model config: obs_dim={obs_dim}, action_dim={action_dim}, hidden_size={hidden_size}")
        else:
            # Use default config values
            obs_dim = config.OBS_DIM
            action_dim = config.ACTION_DIM
            hidden_size = config.HIDDEN_SIZE
            print(f"⚠️  Using default config: obs_dim={obs_dim}, action_dim={action_dim}, hidden_size={hidden_size}")
        
        # Create PyTorch model with NumpyStyleRNNPolicy
        self.policy = NumpyStyleRNNPolicy(
            obs_dim=obs_dim, 
            action_dim=action_dim, 
            hidden_size=hidden_size,
            activation=nn.Sigmoid(),
            alpha=config.ALPHA
        )
        
        # Load model state dict
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model_state_dict = checkpoint['model_state_dict']
            print(f"Loading from model_state_dict")
        else:
            model_state_dict = checkpoint
            print(f"Loading from direct checkpoint")
        
        try:
            self.policy.load_state_dict(model_state_dict)
            self.policy.eval()
            print(f"✅ Successfully loaded model weights")
        except Exception as e:
            print(f"❌ Error loading model state dict: {e}")
            raise
        
        # Print model info if available
        if 'training_type' in checkpoint:
            print(f"🏷️  Training type: {checkpoint['training_type']}")
        if 'success_rate' in checkpoint:
            print(f"📊 Success rate: {checkpoint['success_rate']:.1%}")
        if 'sparsity_stats' in checkpoint:
            stats = checkpoint['sparsity_stats']
            w_out_sparsity = stats.get('W_out', {}).get('sparsity_ratio', 0)
            w_h_sparsity = stats.get('W_h', {}).get('sparsity_ratio', 0)
            print(f"🔍 Sparsity: W_out={w_out_sparsity:.1%}, W_h={w_h_sparsity:.1%}")
        
        # Encoder - FIXED FOR 4×4
        self.encoder = ModalitySpecificEncoder(grid_size=4)  # 4×4 = 16D
        
        # Load SelectiveVecNormalize
        self.selective_norm_stats = None
        if selective_vecnorm_path and os.path.exists(selective_vecnorm_path):
            try:
                from shimmy_wrapper import SelectiveVecNormalize, create_env
                from stable_baselines3.common.vec_env import DummyVecEnv
                
                dummy_env = create_env(random_seed=42)
                dummy_vec = DummyVecEnv([lambda: dummy_env])
                
                self.selective_norm_stats = SelectiveVecNormalize.load(selective_vecnorm_path, dummy_vec)
                dummy_vec.close()
                
                print(f"✅ Loaded SelectiveVecNormalize from {selective_vecnorm_path}")
                print(f"   Normalization dims: {self.selective_norm_stats.norm_dims}")
                print(f"   obs_rms mean shape: {self.selective_norm_stats.obs_rms.mean.shape}")
                
            except Exception as e:
                print(f"❌ Could not load SelectiveVecNormalize: {e}")
                self.selective_norm_stats = None
        else:
            print(f"⚠️  WARNING: SelectiveVecNormalize not found at {selective_vecnorm_path}")
        
        # Compatibility attributes
        self.hidden_size = hidden_size
        self.input_size = 15
        self.output_size = action_dim
        
        # Hidden state management for NumpyStyleRNNPolicy
        self.h = np.zeros(self.hidden_size)
        self.o = np.zeros(self.output_size)  # Add output state
        self._pytorch_hidden = None
        
        print(f"🎯 RNNAdapter initialized successfully!")
        
    def init_state(self):
        """Reset hidden state - now properly handles both h and o states"""
        self.h = np.zeros(self.hidden_size)
        self.o = np.zeros(self.output_size)  # Reset output state too
        self._pytorch_hidden = None
        
        # Also reset the PyTorch model's internal state if needed
        if hasattr(self.policy, 'init_hidden'):
            device = next(self.policy.parameters()).device
            self._pytorch_hidden = self.policy.init_hidden(batch_size=1, device=device)
    
    def activation(self, x):
        """Simple activation function"""
        return np.tanh(x)
        
    def step(self, obs):
        # obs should be 15 raw dims
        assert obs.shape == (15,), f"got {obs.shape}, expected (15,)"
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        
        with torch.no_grad():
            # Encode
            encoded_obs = self.encoder(obs_tensor)
            
            
            assert encoded_obs.shape[-1] == 29, f"encoded→{encoded_obs.shape}, expected 29"
            
            # Apply SelectiveVecNormalize (only first 12 dimensions)
            if self.selective_norm_stats is not None:
                try:
                    assert self.selective_norm_stats.norm_dims == 12
                    assert self.selective_norm_stats.obs_rms.mean.shape == (12,)
                    assert self.selective_norm_stats.obs_rms.var.shape  == (12,)
                    
                    muscle_part = encoded_obs[:, :12]  # First 12 dimensions
                    rest_part = encoded_obs[:, 12:]    # Remaining dimensions (12:93)
                    
                    # Get normalization stats (only for first 12 dims)
                    obs_mean = torch.tensor(self.selective_norm_stats.obs_rms.mean, dtype=torch.float32)
                    obs_var = torch.tensor(self.selective_norm_stats.obs_rms.var, dtype=torch.float32)
                    
                    # Normalize only muscle sensors
                    normalized_muscle = (muscle_part - obs_mean) / torch.sqrt(obs_var + self.selective_norm_stats.epsilon)
                    normalized_muscle = torch.clamp(normalized_muscle, -self.selective_norm_stats.clip_obs, self.selective_norm_stats.clip_obs)
                    
                    # Recombine: normalized muscle + unchanged rest
                    encoded_obs = torch.cat([normalized_muscle, rest_part], dim=1)
                    
                except Exception as e:
                    print(f"Error applying SelectiveVecNormalize: {e}")
            
            # Handle NumpyStyleRNNPolicy hidden state format
            if self._pytorch_hidden is None:
                # Initialize with correct format for NumpyStyleRNNPolicy
                self._pytorch_hidden = self.policy.init_hidden(batch_size=1)
            
            # Update hidden state from numpy arrays
            # NumpyStyleRNNPolicy expects (h, o) tuple
            h_tensor = torch.tensor(self.h, dtype=torch.float32).unsqueeze(0)  # [1, 64]
            o_tensor = torch.tensor(self.o, dtype=torch.float32).unsqueeze(0)  # [1, 4]
            
            self._pytorch_hidden = (h_tensor, o_tensor)
            
            # Forward pass with sequence dimension
            encoded_obs_seq = encoded_obs.unsqueeze(1)  # Add sequence dimension: [1, 1, 93]
            action, self._pytorch_hidden = self.policy(encoded_obs_seq, self._pytorch_hidden)
            
            # Remove sequence dimension from action
            action = action.squeeze(1)  # [1, 1, 4] -> [1, 4]
            
            # Extract both h and o states back to numpy
            if isinstance(self._pytorch_hidden, tuple) and len(self._pytorch_hidden) == 2:
                h_state, o_state = self._pytorch_hidden
                self.h = h_state.squeeze().detach().numpy()
                self.o = o_state.squeeze().detach().numpy()
            else:
                # Fallback for unexpected format
                self.h = self._pytorch_hidden.squeeze().detach().numpy()

        return action.squeeze(0).numpy()
    
    # Weight properties for NumpyStyleRNNPolicy
    @property 
    def W_in(self):
        # NumpyStyleRNNPolicy uses W_in parameter directly
        return self.policy.W_in.data.numpy()
    
    @property
    def W_h(self):
        # NumpyStyleRNNPolicy uses W_h parameter directly
        return self.policy.W_h.data.numpy()
    
    @property
    def W_out(self):
        # NumpyStyleRNNPolicy uses W_out parameter directly
        return self.policy.W_out.data.numpy()
        
    @property
    def b_h(self):
        # NumpyStyleRNNPolicy uses b_h parameter directly
        return self.policy.b_h.data.numpy()
    
    @property 
    def b_out(self):
        # NumpyStyleRNNPolicy uses b_out parameter directly
        return self.policy.b_out.data.numpy()
        
    def get_params(self):
        params = []
        for param in self.policy.parameters():
            params.append(param.data.flatten())
        return torch.cat(params).numpy()