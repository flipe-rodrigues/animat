"""
RNN Core Module - Recurrent neural network for temporal integration

The RNN is the only module with recurrent connections (hidden state).
It integrates sensory information, target encoding, and motor feedback over time.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from core.constants import DEFAULT_RNN_HIDDEN_SIZE


class RNNCore(nn.Module):
    """
    Recurrent neural network core for temporal integration.
    
    This is the only module with a recurrent hidden layer.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = DEFAULT_RNN_HIDDEN_SIZE,
        rnn_type: str = 'rnn',
        num_layers: int = 1,
        input_projection_bias: bool = True,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.rnn_type = rnn_type
        self.num_layers = num_layers

        # Input projection to RNN dimension
        self.input_projection = nn.Linear(
            input_size, hidden_size, bias=input_projection_bias
        )

        # RNN - the only recurrent hidden layer
        if rnn_type == "rnn":
            self.rnn = nn.RNN(
                input_size=hidden_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                nonlinearity="relu",
            )
        elif rnn_type == "gru":
            self.rnn = nn.GRU(
                input_size=hidden_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
            )
        elif rnn_type == "lstm":
            self.rnn = nn.LSTM(
                input_size=hidden_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
            )
        else:
            raise ValueError(f"Unknown RNN type: {rnn_type}")

    def init_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Initialize hidden state for new episodes."""
        if self.rnn_type == "lstm":
            return (
                torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device),
                torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device),
            )
        else:
            return torch.zeros(
                self.num_layers, batch_size, self.hidden_size, device=device
            )

    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for single timestep.

        Args:
            x: Input features [batch, input_size]
            hidden: Optional hidden state

        Returns:
            output: RNN output [batch, hidden_size]
            new_hidden: Updated hidden state
        """
        batch_size = x.shape[0]
        device = x.device

        if hidden is None:
            hidden = self.init_hidden(batch_size, device)

        # Project to RNN dimension
        projected = F.relu(self.input_projection(x))
        projected = projected.unsqueeze(1)  # Add time dimension [batch, 1, hidden]

        # RNN forward
        output, new_hidden = self.rnn(projected, hidden)
        output = output.squeeze(1)  # Remove time dimension [batch, hidden]

        return output, new_hidden

    def forward_sequence(
        self,
        x_sequence: torch.Tensor,
        hidden: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for sequence of inputs.

        Args:
            x_sequence: Input sequence [batch, seq_len, input_size]
            hidden: Optional initial hidden state

        Returns:
            outputs: RNN outputs [batch, seq_len, hidden_size]
            final_hidden: Final hidden state
        """
        batch_size = x_sequence.shape[0]
        device = x_sequence.device

        if hidden is None:
            hidden = self.init_hidden(batch_size, device)

        # Project all timesteps
        projected = F.relu(self.input_projection(x_sequence))  # [batch, seq, hidden]

        # RNN forward on full sequence
        outputs, final_hidden = self.rnn(projected, hidden)

        return outputs, final_hidden
