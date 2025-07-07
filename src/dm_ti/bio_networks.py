import torch
import torch.nn as nn
import numpy as np



class StandardRNNLayer(nn.Module):
    """Standard RNN layer with essential stability measures."""
    
    def __init__(self, size, tau_mem=20.0, tau_dist=None, 
                 tau_range=None, tau_adapt=None, adapt_scale=None, seed=None, device=None,
                 use_layer_norm=False):  # Default to False for better memory
        super().__init__()
        
        # Convert string device to torch.device if needed
        if isinstance(device, str):
            device = torch.device(device)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Set random seed if provided
        if seed is not None:
            torch.manual_seed(seed)
        
        self.size = size
        self.alpha = 1.0 / tau_mem
        self.use_layer_norm = use_layer_norm
        
        # Create weight matrices with simple initialization
        self.weight_ih = nn.Parameter(torch.empty(size, size, device=self.device))
        self.weight_hh = nn.Parameter(torch.empty(size, size, device=self.device))
        self.bias = nn.Parameter(torch.zeros(size, device=self.device))
        
        # Apply standard initializations
        nn.init.xavier_uniform_(self.weight_ih, gain=0.5)
        nn.init.orthogonal_(self.weight_hh, gain=0.5)
        
        # One-time initialization spectral radius control (only at init, not during forward)
        with torch.no_grad():
            u, s, v = torch.linalg.svd(self.weight_hh)
            if s[0] > 0.85:  # Slightly lower than 0.9 for better stability
                self.weight_hh.data = self.weight_hh.data * (0.85 / s[0])
        
        # Only create layer norm if we're using it
        if self.use_layer_norm:
            self.layer_norm = nn.LayerNorm(size)
        
        # Create dummy buffer for adaptation (for API compatibility)
        self.register_buffer('_dummy_adapt', torch.zeros(1, device=self.device))
    
    def forward(self, inputs, adaptation=None, membrane=None):
        # Input handling
        if not isinstance(inputs, torch.Tensor):
            inputs = torch.as_tensor(inputs, dtype=torch.float32, device=self.device)
        elif inputs.device != self.device:
            inputs = inputs.to(self.device)
        
        # Get batch size
        batch_size = inputs.shape[0]
        
        # Initialize or handle membrane (hidden state)
        if membrane is None:
            membrane = torch.zeros(batch_size, self.size, device=self.device)
        elif membrane.device != self.device:
            membrane = membrane.to(self.device)
        
        # Basic NaN checks
        if torch.isnan(inputs).any():
            inputs = torch.nan_to_num(inputs, nan=0.0)
        
        if torch.isnan(membrane).any():
            membrane = torch.nan_to_num(membrane, nan=0.0)
        
        # Compute inner activation
        inner_activation = inputs @ self.weight_ih.t() + membrane @ self.weight_hh.t() + self.bias
        
        # Simplified RNN update with single non-linearity
        h_raw = (1 - self.alpha) * membrane + self.alpha * torch.tanh(inner_activation)
        
        # Apply layer norm if configured, then use tanh for smooth bounding
        if self.use_layer_norm:
            h_norm = self.layer_norm(h_raw)
            h_new = torch.tanh(h_norm)  # Smooth bounded output in [-1,1]
        else:
            h_new = torch.tanh(h_raw)   # Smooth bounded output in [-1,1]
    
        
        # Simplified return - just the new hidden state
        return h_new

    def reset_state(self, batch_size=None):
        """Reset the hidden state."""
        if batch_size is None:
            return None
        return torch.zeros(batch_size, self.size, device=self.device)


class PopulationEncoder(nn.Module):
    def __init__(self, input_dim, population_size=8, min_val=-1.0, max_val=1.0, sigma_factor=0.5, device=None):
        super().__init__()
        self.input_dim = input_dim
        self.population_size = population_size
        
        # Convert string device to torch.device if needed
        if isinstance(device, str):
            device = torch.device(device)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Handle scalar or vector min_val/max_val
        if not isinstance(min_val, (list, tuple, np.ndarray, torch.Tensor)):
            min_val = [min_val] * input_dim
        if not isinstance(max_val, (list, tuple, np.ndarray, torch.Tensor)):
            max_val = [max_val] * input_dim
            
        # Convert to tensor
        min_val = torch.tensor(min_val, dtype=torch.float32, device=self.device)
        max_val = torch.tensor(max_val, dtype=torch.float32, device=self.device)
        
        # Pre-compute preferred values for each dimension
        preferred_values = []
        for dim in range(input_dim):
            dim_pref = torch.linspace(min_val[dim], max_val[dim], population_size, device=self.device)
            preferred_values.append(dim_pref)
        
        # Stack into a matrix of shape [input_dim, population_size]
        self.register_buffer('preferred_values', torch.stack(preferred_values))
        
        # Width of tuning curve - now per dimension
        self.sigma = (max_val - min_val) / (population_size * sigma_factor)
        
        # Cache this value for repeated use in forward pass
        self.register_buffer('two_sigma_sq', 2.0 * self.sigma ** 2)
    
    def forward(self, x):
        # Efficient tensor conversion handling
        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(x, dtype=torch.float32, device=self.device)
        elif x.device != self.device:
            x = x.to(self.device)
        
        # Rest of processing remains the same
        batch_size = x.shape[0]
        x_expanded = x.unsqueeze(-1)
        preferred = self.preferred_values.unsqueeze(0)
        
        # Calculate Gaussian response with numerical stability
        squared_diff = (x_expanded - preferred) ** 2
        
        # Add this line to reshape for proper broadcasting
        two_sigma_sq_expanded = self.two_sigma_sq.view(1, self.input_dim, 1)
        
        # Use the reshaped tensor
        normalized_diff = torch.clamp(squared_diff / two_sigma_sq_expanded, min=0.0, max=10.0)
        response = torch.exp(-normalized_diff)
        
        # Output shape: [batch_size, input_dim * population_size] 
        return response.view(batch_size, -1)


class ModalitySpecificEncoder(nn.Module):
    def __init__(self, target_size=12, device=None):
        super().__init__()
        
        # Convert string device to torch.device if needed
        if isinstance(device, str):
            device = torch.device(device)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Only keep population encoding for target position (higher brain regions)
        self.target_encoder = PopulationEncoder(
            input_dim=2,             # Only encode X and Y (drop Z dimension)
            population_size=target_size,  # Number of neurons in the population
            min_val=[-0.65, -0.90],  # Only X and Y ranges
            max_val=[0.90, 0.35],     
            sigma_factor=1.0,      
            device=self.device
        )
        
        # Store output size for easier shape inference
        # Direct sensory input (4 length + 4 velocity + 4 force) + encoded target
        self.output_size = 12 + 1 + (2 * target_size)  # 2 dimensions × 20 neurons
    
    def forward(self, x):
        # Move input to the correct device
        x = x.to(self.device)
        
        # x shape: [batch_size, 15]
        
        # Split into modalities - keep direct sensory inputs
        muscle_data = x[:, :12]    # All muscle data (indices 0-11)
        target_pos = x[:, 12:15]   # Target position (indices 12, 13, 14)
        
        # Only encode X and Y coordinates (skip Z)
        target_pos_xy = target_pos[:, :2]  # Only take first two dimensions
        
        # Only encode target position with population code
        encoded_target = self.target_encoder(target_pos_xy)
        
        # Concatenate direct muscle data with encoded target position
        return torch.cat([muscle_data,  encoded_target, target_pos[:, 2:3]], dim=1)


