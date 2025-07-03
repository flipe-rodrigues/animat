import torch
import torch.nn as nn
import torch.nn.utils as utils
import numpy as np


class LeakyRNNCell(nn.Module):
    """
    Custom RNN cell with leaky integration (alpha parameter derived from tau).
    Compatible with PyTorch's RNN interfaces.
    """
    def __init__(self, input_size, hidden_size, tau_mem=20.0, use_layer_norm=False, spectral_radius=0.9):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.alpha = 1.0 / tau_mem
        self.use_layer_norm = use_layer_norm
        self.spectral_radius = spectral_radius
        
        # Create weight matrices as leaf tensors
        self.weight_ih = nn.Parameter(torch.empty(hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        
        # More conservative initialization
        nn.init.xavier_uniform_(self.weight_ih, gain=0.5)  # Further reduced
        nn.init.orthogonal_(self.weight_hh, gain=0.8)     # Further reduced
        
        # Manual spectral normalization buffers (avoid torch.nn.utils.spectral_norm)
        self.register_buffer('u', torch.randn(hidden_size))
        self.register_buffer('v', torch.randn(hidden_size))
        
        # Initialize spectral vectors
        with torch.no_grad():
            u, s, v = torch.linalg.svd(self.weight_hh.data)
            self.u.copy_(u[:, 0])
            self.v.copy_(v[0, :])
            
            # Apply initial spectral radius constraint
            if s[0] > self.spectral_radius:
                self.weight_hh.data = self.weight_hh.data * (self.spectral_radius / s[0])
        
        # Layer norm if specified
        if self.use_layer_norm:
            self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-8)
        
        # More conservative stability parameters
        self.activation_clip = 8.0  # Reduced from 8.0
        self.hidden_clip = 3.0      # Reduced from 3.0
    
    def spectral_normalize_weight_hh(self):
        """Apply spectral normalization using power iteration method."""
        with torch.no_grad():
            # Power iteration to find largest singular value
            v = self.v
            u = self.u
            
            # One power iteration step
            v_new = torch.mv(self.weight_hh.t(), u)
            v_new = v_new / (torch.norm(v_new) + 1e-12)
            
            u_new = torch.mv(self.weight_hh, v_new)
            u_new = u_new / (torch.norm(u_new) + 1e-12)
            
            # Update buffers
            self.u.copy_(u_new)
            self.v.copy_(v_new)
            
            # Calculate spectral norm and normalize if needed
            sigma = torch.dot(u_new, torch.mv(self.weight_hh, v_new))
            
            if sigma > self.spectral_radius:
                self.weight_hh.data.div_(sigma / self.spectral_radius)
    
    def forward(self, input, hx=None):
        """Forward pass with enhanced stability measures."""
        if hx is None:
            hx = torch.zeros(input.shape[0], self.hidden_size, device=input.device)
        
        # Handle 3D hidden states
        if hx.dim() == 3:
            hx_2d = hx[0]
        else:
            hx_2d = hx
        
        # Essential NaN protection (keep this)
        if torch.isnan(input).any() or torch.isnan(hx_2d).any():
            input = torch.nan_to_num(input, nan=0.0, posinf=1.0, neginf=-1.0)
            hx_2d = torch.nan_to_num(hx_2d, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Apply spectral normalization (keep this)
        self.spectral_normalize_weight_hh()
        
        # Forward computation with LESS aggressive clipping
        input_component = torch.mm(input, self.weight_ih.t())
        recurrent_component = torch.mm(hx_2d, self.weight_hh.t())
        
        inner_activation = input_component + recurrent_component + self.bias
        
        # Only clip if really necessary
        if torch.isnan(inner_activation).any():
            inner_activation = torch.nan_to_num(inner_activation, nan=0.0)
    
        # Apply layer norm if enabled
        if self.use_layer_norm:
            inner_activation = self.layer_norm(inner_activation)
    
        # Bounded activation
        tanh_activation = torch.tanh(inner_activation)
        
        # Leaky integration with more flexible alpha
        alpha_clamped = torch.clamp(torch.tensor(self.alpha), min=0.01, max=0.7)
        h_new = (1 - alpha_clamped) * hx_2d + alpha_clamped * tanh_activation
        
        # Only final safety clipping
        h_new = torch.clamp(h_new, min=-self.hidden_clip, max=self.hidden_clip)
        h_new = torch.nan_to_num(h_new, nan=0.0)
        
        return h_new


class LeakyRNN(nn.Module):
    """
    Multi-layer leaky RNN using custom RNN cells.
    Compatible with PyTorch's RNN interfaces and Tianshou.
    """
    def __init__(self, input_size, hidden_size, num_layers=1, 
                 batch_first=True, tau_mem=20.0, use_layer_norm=True,
                 debug=False, spectral_radius=0.9, max_grad_norm=1.0):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.debug = debug
        self.max_grad_norm = max_grad_norm
        
        # Create RNN cells for each layer
        self.cells = nn.ModuleList()
        for i in range(num_layers):
            cell = LeakyRNNCell(
                input_size if i == 0 else hidden_size, 
                hidden_size, 
                tau_mem=tau_mem,
                use_layer_norm=use_layer_norm,
                spectral_radius=spectral_radius
            )
            if debug:
                cell.debug = True
            self.cells.append(cell)
    
    def forward(self, input, hx=None):
        """Forward pass for the RNN with enhanced NaN protection."""
        # Handle batch_first format
        if self.batch_first:
            input = input.transpose(0, 1)
            
        seq_len, batch_size, _ = input.shape
        device = input.device
        
        # Check for NaN/Inf in input
        if torch.isnan(input).any() or torch.isinf(input).any():
            input = torch.nan_to_num(input, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Initialize hidden states if not provided
        if hx is None:
            hx = self.init_hidden(batch_size, device)
        
        # Check for NaN/Inf in initial hidden state
        if torch.isnan(hx).any() or torch.isinf(hx).any():
            hx = torch.nan_to_num(hx, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Prepare output tensor
        outputs = torch.zeros(seq_len, batch_size, self.hidden_size, device=device)
        
        # Process each time step
        for t in range(seq_len):
            x_t = input[t]
            
            # Additional input validation
            if torch.isnan(x_t).any() or torch.isinf(x_t).any():
                x_t = torch.nan_to_num(x_t, nan=0.0, posinf=1.0, neginf=-1.0)
            
            # Process through each layer
            layer_input = x_t
            h_next = []
            
            for layer_idx, cell in enumerate(self.cells):
                h_t = hx[layer_idx]
                
                # Additional hidden state validation
                if torch.isnan(h_t).any() or torch.isinf(h_t).any():
                    h_t = torch.nan_to_num(h_t, nan=0.0, posinf=1.0, neginf=-1.0)
                
                h_t = cell(layer_input, h_t)
                h_next.append(h_t)
                layer_input = h_t
            
            # Update hidden states for next timestep
            hx = torch.stack(h_next)
            
            # Check for NaN/Inf in hidden states
            if torch.isnan(hx).any() or torch.isinf(hx).any():
                hx = torch.nan_to_num(hx, nan=0.0, posinf=1.0, neginf=-1.0)
            
            # Store output from last layer
            outputs[t] = h_next[-1]
        
        # Convert back to batch_first if needed
        if self.batch_first:
            outputs = outputs.transpose(0, 1)
        
        # Final NaN/Inf check
        if torch.isnan(outputs).any() or torch.isinf(outputs).any():
            outputs = torch.nan_to_num(outputs, nan=0.0, posinf=1.0, neginf=-1.0)
            
        return outputs, hx
    
    def init_hidden(self, batch_size, device=None):
        """Initialize hidden state with proper dimensions."""
        if device is None and next(self.parameters(), None) is not None:
            device = next(self.parameters()).device
        return torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
    
    def reset_state(self, batch_size=None, done_env_ids=None, state=None):
        """Reset hidden states for Tianshou compatibility."""
        if batch_size is None and state is None:
            return None
            
        if state is None:
            return self.init_hidden(batch_size)
        
        if done_env_ids is not None and len(done_env_ids) > 0:
            if isinstance(state, torch.Tensor) and state.dim() == 3:
                for env_id in done_env_ids:
                    if env_id < state.size(1):
                        state[:, env_id].zero_()
            elif isinstance(state, list):
                for s in state:
                    if isinstance(s, torch.Tensor):
                        for env_id in done_env_ids:
                            if env_id < s.size(1):
                                s[:, env_id].zero_()
        
        return state


class PopulationEncoder(nn.Module):
    def __init__(self, input_dim, population_size=8, min_val=-1.0, max_val=1.0, device=None):
        super().__init__()
        self.input_dim = input_dim
        self.population_size = population_size
        
        # Handle scalar or vector min_val/max_val
        if not isinstance(min_val, (list, tuple, np.ndarray, torch.Tensor)):
            min_val = [min_val] * input_dim
        if not isinstance(max_val, (list, tuple, np.ndarray, torch.Tensor)):
            max_val = [max_val] * input_dim
            
        # Convert to tensor - register as buffers for proper device movement
        self.register_buffer('min_val', torch.tensor(min_val, dtype=torch.float32))
        self.register_buffer('max_val', torch.tensor(max_val, dtype=torch.float32))
        
        # Get device (use provided or default)
        device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize buffers immediately
        preferred_values = []
        for dim in range(self.input_dim):
            dim_pref = torch.linspace(
                self.min_val[dim], self.max_val[dim], 
                self.population_size, device=device
            )
            preferred_values.append(dim_pref)
        
        # Register all buffers during initialization
        self.register_buffer('preferred_values', torch.stack(preferred_values))
        
        # More robust sigma calculation to prevent division by zero
        sigma = (self.max_val - self.min_val) / (self.population_size * 0.5)
        sigma = torch.clamp(sigma, min=1e-6)  # Prevent extremely small sigma
        
        self.register_buffer('sigma', sigma)
        self.register_buffer('two_sigma_sq', 2.0 * sigma ** 2)
    
    def forward(self, x):
        batch_size = x.shape[0]
        
        # Check for NaN inputs
        if torch.isnan(x).any():
            print("Warning: NaN detected in PopulationEncoder input")
            x = torch.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)
        
        x_expanded = x.unsqueeze(-1)
        preferred = self.preferred_values.unsqueeze(0)
        
        # Calculate Gaussian response with numerical stability
        squared_diff = (x_expanded - preferred) ** 2
        
        # Reshape for proper broadcasting
        two_sigma_sq_expanded = self.two_sigma_sq.view(1, self.input_dim, 1)
        
        # Use the reshaped tensor with more conservative clamping
        normalized_diff = torch.clamp(squared_diff / two_sigma_sq_expanded, min=0.0, max=8.0)  # Reduced from 10.0
        response = torch.exp(-normalized_diff)
        
        # Final NaN check
        response = torch.nan_to_num(response, nan=0.0)
        
        # Output shape: [batch_size, input_dim * population_size] 
        return response.view(batch_size, -1)


class ModalitySpecificEncoder(nn.Module):
    def __init__(self, target_size=12):
        super().__init__()
        
        # Only keep population encoding for target position (higher brain regions)
        self.target_encoder = PopulationEncoder(
            input_dim=2,             # Only encode X and Y (drop Z dimension)
            population_size=target_size,  # Number of neurons in the population
            min_val=[-0.65, -0.90],  # Only X and Y ranges
            max_val=[0.90, 0.35]     
        )
        
        # Store output size for easier shape inference
        # Direct sensory input (12) + encoded target (2*target_size) + Z coordinate (1)
        self.output_size = 12 + (2 * target_size) + 1
    
    def forward(self, x):
        # Check for NaN inputs
        if torch.isnan(x).any():
            print("Warning: NaN detected in ModalitySpecificEncoder input")
            x = torch.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # Split into modalities - keep direct sensory inputs
        muscle_data = x[:, :12]      # All muscle data (indices 0-11)
        target_pos = x[:, 12:15]     # Target position (indices 12, 13, 14)
        
        # Only encode X and Y coordinates (skip Z)
        target_pos_xy = target_pos[:, :2]  # Only take first two dimensions
        
        # Only encode target position with population code
        encoded_target = self.target_encoder(target_pos_xy)
        
        # Concatenate direct muscle data with encoded target position and Z coordinate
        result = torch.cat([muscle_data, encoded_target, target_pos[:, 2:3]], dim=1)
        
        # Final NaN check
        if torch.isnan(result).any():
            print("Warning: NaN detected in ModalitySpecificEncoder output")
            result = torch.nan_to_num(result, nan=0.0)
        
        return result