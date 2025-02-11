import torch
import torch.nn as nn

class RNNController(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNNController, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden):
        out, hidden = self.rnn(x, hidden)
        out = self.fc(out[:, -1, :])
        return out, hidden

    def init_hidden(self, batch_size, hidden_size):
        return torch.zeros(1, batch_size, hidden_size)