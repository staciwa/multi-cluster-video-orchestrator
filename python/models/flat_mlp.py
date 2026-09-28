import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import Linear
from torch_geometric.utils import softmax

class FlatMLP(nn.Module):
    """
    Flat MLP Architecture for Imitation Learning (Ablation Study).
    Learns to predict continuous assigned traffic on PoP -> Cluster edges
    without any Message Passing (HeteroConv) steps. It relies solely on 
    the raw initial features of the source and destination nodes and edge attributes.
    """
    def __init__(self, hidden_channels: int = 128, edge_dim: int = 3):
        super().__init__()
        
        # 1. Feature Encoders
        self.encoder_cluster = Linear(5, hidden_channels)
        self.encoder_pop = Linear(3, hidden_channels)

        # No message passing layers (conv1, conv2, conv3) in this ablation study!

        # 2. Edge Prediction Head
        # Increased depth and width to compensate for lack of convolutions and match GNN parameter count
        self.edge_pred_head = nn.Sequential(
            nn.Linear(hidden_channels * 2 + edge_dim, hidden_channels * 2),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_channels * 2, hidden_channels * 2),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.LeakyReLU(),
            nn.Linear(hidden_channels, 1)
        )
        
        # 3. Unserved Prediction Head (per PoP)
        self.unserved_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.LeakyReLU(),
            nn.Linear(hidden_channels, 1)
        )

    def forward(self, x_dict, edge_index_dict, edge_attr_dict):
        # 1. Encode
        x_dict_encoded = {
            'cluster': F.leaky_relu(self.encoder_cluster(x_dict['cluster'])),
            'pop': F.leaky_relu(self.encoder_pop(x_dict['pop']))
        }

        # 2. Skip Message Passing entirely
        x_dict_out = x_dict_encoded

        # 3. Output (Edge Regression)
        edge_index = edge_index_dict[('pop', 'reaches', 'cluster')]
        src_idx, dst_idx = edge_index[0], edge_index[1]
        
        src_features = x_dict_out['pop'][src_idx]
        dst_features = x_dict_out['cluster'][dst_idx]
        edge_attr = edge_attr_dict[('pop', 'reaches', 'cluster')]
        
        edge_features = torch.cat([src_features, dst_features, edge_attr], dim=-1)
        raw_edge_logits = self.edge_pred_head(edge_features).squeeze(-1)
        
        invalid_mask = edge_attr[:, 2] <= 0.001
        raw_edge_logits = raw_edge_logits.masked_fill(invalid_mask, -1e9)
        
        unserved_logits = self.unserved_head(x_dict_out['pop']).squeeze(-1)
        
        num_pops = x_dict_out['pop'].size(0)
        pop_indices = torch.arange(num_pops, device=raw_edge_logits.device)
        
        concat_logits = torch.cat([raw_edge_logits, unserved_logits], dim=0)
        concat_index = torch.cat([src_idx, pop_indices], dim=0)
        
        concat_probs = softmax(concat_logits, concat_index)
        
        edge_probs = concat_probs[:raw_edge_logits.size(0)]
        
        pop_demands = x_dict['pop'][:, 2]
        assigned_demands_pred = edge_probs * pop_demands[src_idx]

        return assigned_demands_pred.unsqueeze(-1)
