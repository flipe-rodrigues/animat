import torch
import torch.nn as nn
import numpy as np

class NumpyStyleRNNPolicy(nn.Module):
    """
    Exactly the same update as the NumPy RNN:
      h ← (1−α)·h + α·activation(W_in·x + W_h·h + b_h)
      o ← (1−α)·o + α·sigmoid(W_out·h + b_out)
    """
    def __init__(self,
                 obs_dim,
                 action_dim,
                 hidden_size,
                 activation: nn.Module = nn.Sigmoid(),
                 alpha: float = 0.1):
        super().__init__()
        self.obs_dim = obs_dim
        self.hidden_size = hidden_size
        self.action_dim = action_dim
        self.activation = activation
        self.alpha = alpha

        # exactly W_in, W_h, W_out + biases
        self.W_in = nn.Parameter(torch.empty(hidden_size, obs_dim))
        self.W_h  = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.b_h  = nn.Parameter(torch.zeros(hidden_size))
        self.W_out= nn.Parameter(torch.empty(action_dim, hidden_size))
        self.b_out= nn.Parameter(torch.zeros(action_dim))

        self._init_weights()

    def _init_weights(self):
        # match xavier_init / he_init logic from utils
        nn.init.xavier_uniform_(self.W_in)
        nn.init.xavier_uniform_(self.W_h)
        nn.init.xavier_uniform_(self.W_out)
        nn.init.constant_(self.b_h,  0.)
        nn.init.constant_(self.b_out,0.)

    def init_hidden(self, batch_size, device=None):
        if device is None:
            device = next(self.parameters()).device
        # Return BOTH h and o states as a tuple
        h = torch.zeros(1, batch_size, self.hidden_size, device=device)
        o = torch.zeros(1, batch_size, self.action_dim, device=device)
        return (h, o)
    
    def forward(self, obs, hidden_state=None):
        single = False
        if obs.dim() == 2:
            obs = obs.unsqueeze(1)
            single = True

        B, T, _ = obs.shape
        device = obs.device

        if hidden_state is None:
            h = torch.zeros(B, self.hidden_size, device=device)
            o = torch.zeros(B, self.action_dim, device=device)
        else:
            # Extract BOTH h and o from hidden state
            if isinstance(hidden_state, tuple) and len(hidden_state) == 2:
                h_state, o_state = hidden_state
                h = h_state.squeeze(0) if h_state.dim() == 3 else h_state
                o = o_state.squeeze(0) if o_state.dim() == 3 else o_state
            else:
                # Fallback for old checkpoints
                h = hidden_state.squeeze(0) if hidden_state.dim() == 3 else hidden_state
                o = torch.zeros(B, self.action_dim, device=device)

        outs = []
        for t in range(T):
            x_t = obs[:, t, :]
            pre_h = x_t @ self.W_in.t() + h @ self.W_h.t() + self.b_h
            h = (1-self.alpha)*h + self.alpha*self.activation(pre_h)

            pre_o = h @ self.W_out.t() + self.b_out
            o = (1-self.alpha)*o + self.alpha*torch.sigmoid(pre_o)

            outs.append(o)

        out = torch.stack(outs, dim=1)
        if single:
            out = out.squeeze(1)

        # Return BOTH h and o as the new hidden state
        new_hidden = (h.unsqueeze(0), o.unsqueeze(0))
        return out, new_hidden