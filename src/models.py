import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, TopKPooling, global_mean_pool
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp

EMBEDDING_SIZE = 64


class GCN(torch.nn.Module):

    def __init__(self, arch, num_features, embedding_size=EMBEDDING_SIZE):
        # Init parent
        super(GCN, self).__init__()
        torch.manual_seed(42)

        # GCN layers
        self.initial_conv = arch(num_features, embedding_size)
        self.conv1 = arch(embedding_size, embedding_size)
        self.conv2 = arch(embedding_size, embedding_size)
        self.conv3 = arch(embedding_size, embedding_size)

        # Output layer
        self.out = Linear(embedding_size * 2, 1)

    def forward(self, x, edge_index, batch_index):
        # First Conv layer
        hidden = self.initial_conv(x, edge_index)
        hidden = F.tanh(hidden)

        # Other Conv layers
        hidden = self.conv1(hidden, edge_index)
        hidden = F.tanh(hidden)
        hidden = self.conv2(hidden, edge_index)
        hidden = F.tanh(hidden)
        hidden = self.conv3(hidden, edge_index)
        hidden = F.tanh(hidden)

        # Global Pooling (stack different aggregations)
        hidden = torch.cat(
            [gmp(hidden, batch_index),
             gap(hidden, batch_index)], dim=1)

        # Apply a final (linear) classifier.
        out = self.out(hidden)

        return out, hidden
