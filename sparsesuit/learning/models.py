import torch
from torch import nn


class BiRNN(nn.Module):
    """
    Implementation of a flexible Bi-directional RNN as used in Deep Inertial Poser (DIP):
    Input: B x D_i x 1 array, where D_i depends on the number of sensors N used as inputs D_i = N x (9 + 3)
    Output: two B x D_o x 1 arrays, where D_o depends on the number of SMPL target joints D_o = 15 for DIP and 19 for SSP/MVN
    Layers: - dropout with probability 0.2
            - FC (D x 512)
            - 2 x ReLu
            - 2 layers of bi-directional LSTM cells (512 x 512)
            - 2 x FC (1024 x D_o)
            - SoftPlus activation for the second FC layer to enforce non-negativity of stds
    """

    def __init__(
        self,
        input_dim=60,
        target_dim=150,
        hidden_layers=2,
        hidden_dim=512,
        bidirectional=True,
    ):
        super().__init__()

        self.hidden_layers = hidden_layers
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional
        self.num_directions = 2 if self.bidirectional else 1

        self.dropout = nn.Dropout(
            p=0.2
        )  # module is more convenient than functional, turns off automatically in eval()
        self.fc1 = nn.Linear(in_features=input_dim, out_features=hidden_dim)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=hidden_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )
        self.fc2a = nn.Linear(
            in_features=self.num_directions * hidden_dim, out_features=target_dim
        )
        self.fc2b = nn.Linear(
            in_features=self.num_directions * hidden_dim, out_features=target_dim
        )

    def forward(self, x, h_last=None, c_last=None):
        out = self.dropout(x)
        out = self.fc1(out)
        out = self.relu(out)
        # states are init as zero if not provided
        if h_last is None:
            h_last, c_last = self.init_hidden(x.shape[0])
        out, (h, c) = self.lstm(out, (h_last.to(x.device), c_last.to(x.device)))
        out_mu = self.fc2a(out)
        out_sigma = self.fc2b(out)
        out_sigma = nn.functional.softplus(out_sigma)
        return out_mu, out_sigma, h, c

    def init_hidden(self, batch_size):
        hidden = torch.zeros(
            self.num_directions * self.hidden_layers, batch_size, self.hidden_dim
        )
        cell = torch.zeros(
            self.num_directions * self.hidden_layers, batch_size, self.hidden_dim
        )
        return hidden, cell
