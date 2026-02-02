"""RNN Core Module - Recurrent neural network for temporal integration."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union

from core.constants import DEFAULT_RNN_HIDDEN_SIZE


class RNNCore(nn.Module):
    """Recurrent neural network core for temporal integration."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = DEFAULT_RNN_HIDDEN_SIZE,
        rnn_type: str = "rnn",
        num_layers: int = 1,
        input_projection_bias: bool = True,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.rnn_type = rnn_type
        self.num_layers = num_layers

        self.input_projection = nn.Linear(input_size, hidden_size, bias=input_projection_bias)

        rnn_cls = {"rnn": nn.RNN, "gru": nn.GRU, "lstm": nn.LSTM}.get(rnn_type)
        if rnn_cls is None:
            raise ValueError(f"Unknown RNN type: {rnn_type}")

        rnn_kwargs = {"input_size": hidden_size, "hidden_size": hidden_size, "num_layers": num_layers, "batch_first": True}
        if rnn_type == "rnn":
            rnn_kwargs["nonlinearity"] = "relu"
        self.rnn = rnn_cls(**rnn_kwargs)

    def init_hidden(self, batch_size: int, device: torch.device) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Initialize hidden state for new episodes."""
        zeros = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        return (zeros, zeros.clone()) if self.rnn_type == "lstm" else zeros

    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]] = None,
    ) -> Tuple[torch.Tensor, Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass for single timestep.

        Args:
            x: [batch, input_size]
            hidden: Optional hidden state

        Returns:
            output: [batch, hidden_size]
            new_hidden: Updated hidden state
        """
        if hidden is None:
            hidden = self.init_hidden(x.shape[0], x.device)

        projected = F.relu(self.input_projection(x)).unsqueeze(1)
        output, new_hidden = self.rnn(projected, hidden)
        output = output.squeeze(1)
        
        # Clamp to prevent exploding activations
        output = torch.clamp(output, -10.0, 10.0)
        
        return output, new_hidden

    def forward_sequence(
        self,
        x_sequence: torch.Tensor,
        hidden: Optional[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]] = None,
    ) -> Tuple[torch.Tensor, Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass for sequence.

        Args:
            x_sequence: [batch, seq_len, input_size]
            hidden: Optional initial hidden state

        Returns:
            outputs: [batch, seq_len, hidden_size]
            final_hidden: Final hidden state
        """
        if hidden is None:
            hidden = self.init_hidden(x_sequence.shape[0], x_sequence.device)

        projected = F.relu(self.input_projection(x_sequence))
        return self.rnn(projected, hidden)
