import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, HeteroConv, Linear
from torch_geometric.utils import softmax

class BaselineGNN(nn.Module):
    """
    Baseline GNN Architecture for Imitation Learning (Macro-Scaled Edge Regression).
    Learns to predict continuous assigned traffic on PoP -> Cluster edges.
    
    Node types: 'cluster', 'pop'
    Edge types: ('pop', 'reaches', 'cluster')
    
    Output:
        - assigned_demands_pred: continuous value per edge representing assigned traffic.
    """
    def __init__(self, hidden_channels: int = 128, edge_dim: int = 3):
        super().__init__()
        
        # 1. Feature Encoders (Projection to hidden dimension)
        self.encoder_cluster = Linear(5, hidden_channels)
        self.encoder_pop = Linear(3, hidden_channels)

        # 2. Heterogeneous Message Passing Layers
        self.conv1 = HeteroConv({
            ('pop', 'reaches', 'cluster'): SAGEConv(hidden_channels, hidden_channels),
            ('cluster', 'rev_reaches', 'pop'): SAGEConv(hidden_channels, hidden_channels),
        }, aggr='sum')

        self.conv2 = HeteroConv({
            ('pop', 'reaches', 'cluster'): SAGEConv(hidden_channels, hidden_channels),
            ('cluster', 'rev_reaches', 'pop'): SAGEConv(hidden_channels, hidden_channels),
        }, aggr='sum')

        self.conv3 = HeteroConv({
            ('pop', 'reaches', 'cluster'): SAGEConv(hidden_channels, hidden_channels),
            ('cluster', 'rev_reaches', 'pop'): SAGEConv(hidden_channels, hidden_channels),
        }, aggr='sum')

        # 3. Edge Prediction Heads
        self.edge_pred_head = nn.Sequential(
            nn.Linear(hidden_channels * 2 + edge_dim, hidden_channels),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_channels, 1) # Predicts 1 continuous logit per edge
        )
        
        # 4. Unserved Prediction Head (per PoP)
        self.unserved_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.LeakyReLU(),
            nn.Linear(hidden_channels, 1) # Predicts 1 continuous logit per PoP for unserved demands
        )

    def forward(self, x_dict, edge_index_dict, edge_attr_dict):
        # 1. Encode
        x_dict_encoded = {
            'cluster': F.leaky_relu(self.encoder_cluster(x_dict['cluster'])),
            'pop': F.leaky_relu(self.encoder_pop(x_dict['pop']))
        }

        # 2. Message Passing
        x_dict_1 = self.conv1(x_dict_encoded, edge_index_dict)
        x_dict_1 = {key: F.leaky_relu(x) for key, x in x_dict_1.items()}
        
        x_dict_2 = self.conv2(x_dict_1, edge_index_dict)
        x_dict_2 = {key: F.leaky_relu(x) for key, x in x_dict_2.items()}

        x_dict_3 = self.conv3(x_dict_2, edge_index_dict)
        x_dict_3 = {key: F.leaky_relu(x) for key, x in x_dict_3.items()}

        # 3. Output (Edge Regression with Balance Guarantee)
        edge_index = edge_index_dict[('pop', 'reaches', 'cluster')]
        src_idx, dst_idx = edge_index[0], edge_index[1]
        
        src_features = x_dict_3['pop'][src_idx]
        dst_features = x_dict_3['cluster'][dst_idx]
        edge_attr = edge_attr_dict[('pop', 'reaches', 'cluster')]
        
        # Concat embeddings of the two nodes forming the edge + edge attributes (distance, QoS)
        edge_features = torch.cat([src_features, dst_features, edge_attr], dim=-1)
        raw_edge_logits = self.edge_pred_head(edge_features).squeeze(-1) # [num_edges]
        
        # Symmetry breaking (Masking): If QoS is 0 (distance > MaxLatency), cut off this cluster
        invalid_mask = edge_attr[:, 2] <= 0.001
        raw_edge_logits = raw_edge_logits.masked_fill(invalid_mask, -1e9)
        
        # Predict unserved logits for each PoP
        unserved_logits = self.unserved_head(x_dict_3['pop']).squeeze(-1) # [num_pops]
        
        num_pops = x_dict_3['pop'].size(0)
        pop_indices = torch.arange(num_pops, device=raw_edge_logits.device)
        
        # Concatenate edge logits and unserved logits to do a unified softmax per PoP
        concat_logits = torch.cat([raw_edge_logits, unserved_logits], dim=0)
        concat_index = torch.cat([src_idx, pop_indices], dim=0)
        
        concat_probs = softmax(concat_logits, concat_index)
        
        # Extract edge probabilities
        edge_probs = concat_probs[:raw_edge_logits.size(0)]
        
        # Multiply by total demands for each PoP to guarantee exact balance
        pop_demands = x_dict['pop'][:, 2] # n_demands is at feature index 2
        assigned_demands_pred = edge_probs * pop_demands[src_idx]

        return assigned_demands_pred.unsqueeze(-1)
