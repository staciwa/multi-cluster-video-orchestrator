import torch
import torch.nn as nn

class BaselineMLP(nn.Module):
    """
    Classic multilayer perceptron (MLP) on flattened features.
    Used for demonstration purposes in the Ablation Study (MLP vs GNN).
    Expects a fixed graph size (default 30 PoPs, 20 Clusters).
    """
    def __init__(self, num_pops=30, num_clusters=20, hidden_dim=256):
        super().__init__()
        
        self.num_pops = num_pops
        self.num_clusters = num_clusters
        
        # Features: 3 (PoP) + 5 (Cluster) + 3 (Edge)
        input_dim = (num_pops * 3) + (num_clusters * 5) + (num_pops * num_clusters * 3)
        output_dim = num_pops * num_clusters
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x_dict, edge_index_dict, edge_attr_dict):
        pop_x = x_dict['pop']
        cluster_x = x_dict['cluster']
        edge_attr = edge_attr_dict[('pop', 'reaches', 'cluster')]
        
        # Reconstruct batch_size from PyG (which concatenates along dim=0)
        batch_size = pop_x.size(0) // self.num_pops
        
        pop_flat = pop_x.view(batch_size, -1)
        cluster_flat = cluster_x.view(batch_size, -1)
        edge_flat = edge_attr.view(batch_size, -1)
        
        x = torch.cat([pop_flat, cluster_flat, edge_flat], dim=1)
        out = self.net(x)
        
        # Format output to [batch_edges, 1] (consistent with GNN)
        out = out.view(batch_size * self.num_pops * self.num_clusters, 1)
        
        # QoS Masking (in GNN -1e9 is before softmax, giving 0.0 at output, so we also put 0.0 here)
        qos_values = edge_attr[:, 2:3]
        out = torch.where(qos_values > 0, out, torch.tensor(0.0, device=out.device))
        
        return out
