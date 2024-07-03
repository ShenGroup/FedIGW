import torch
import torch.nn as nn

# Define actor and critic networks
class MLP(nn.Module):
    def __init__(self, contex_dim, output_dim):
        super(MLP, self).__init__()
        self.layer_1 = nn.Linear(contex_dim, 256)
        self.layer_2 = nn.Linear(256, 256)
        self.layer_3 = nn.Linear(256, output_dim)

    def forward(self, x):
        x = self.layer_1(x)
        x = torch.relu(self.layer_2(x))
        x = torch.tanh(self.layer_3(x))
        return x