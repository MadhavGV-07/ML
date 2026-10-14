"""
Graph Neural Network Architectures for Anti-Money Laundering Detection:
- GCN (Graph Convolutional Network)
- GAT (Graph Attention Network with Attention Extraction for XAI)
- GraphSAGE (Sample & Aggregate with Inductive Neighborhood Pooling)
- Baseline MLP (Tabular non-graph classifier benchmark)
- Focal Loss & Cost-Sensitive Cross Entropy for severe class imbalance
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# 1. Focal Loss for Extreme AML Imbalance
# ==========================================
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        weights = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        focal_loss = weights * ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


# ==========================================
# 2. Graph Convolutional Network (GCN)
# ==========================================
class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.2):
        super(GCNLayer, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.bias = nn.Parameter(torch.FloatTensor(out_features))
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, adj_norm):
        x = self.dropout(x)
        support = torch.mm(x, self.weight)
        output = torch.spmm(adj_norm, support) if adj_norm.is_sparse else torch.mm(adj_norm, support)
        return output + self.bias


class GCNClassifier(nn.Module):
    def __init__(self, in_features, hidden_dim=64, num_classes=2, dropout=0.3):
        super(GCNClassifier, self).__init__()
        self.conv1 = GCNLayer(in_features, hidden_dim, dropout)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.conv2 = GCNLayer(hidden_dim, hidden_dim // 2, dropout)
        self.bn2 = nn.BatchNorm1d(hidden_dim // 2)
        self.fc = nn.Linear(hidden_dim // 2, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adj_norm):
        h = F.relu(self.bn1(self.conv1(x, adj_norm)))
        h = self.dropout(h)
        h = F.relu(self.bn2(self.conv2(h, adj_norm)))
        h = self.dropout(h)
        logits = self.fc(h)
        return logits, h


# ==========================================
# 3. Graph Attention Network (GAT)
# ==========================================
class GATLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.2, alpha=0.2, concat=True):
        super(GATLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.last_attention = None  # Cache for explainability / visualization

    def forward(self, h, adj):
        # h: (N, in_features), adj: (N, N)
        Wh = torch.mm(h, self.W)  # (N, out_features)
        N = Wh.size()[0]

        # Self-attention mechanism
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])  # (N, 1)
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])  # (N, 1)

        # broadcast add
        e = self.leakyrelu(Wh1 + Wh2.T)  # (N, N)

        # Mask non-adjacent connections
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)

        # Save attention weights for explainability
        self.last_attention = attention.detach()

        h_prime = torch.matmul(attention, Wh)
        if self.concat:
            return F.elu(h_prime)
        return h_prime


class GATClassifier(nn.Module):
    def __init__(self, in_features, hidden_dim=64, num_classes=2, dropout=0.3):
        super(GATClassifier, self).__init__()
        self.gat1 = GATLayer(in_features, hidden_dim, dropout=dropout, concat=True)
        self.gat2 = GATLayer(hidden_dim, hidden_dim // 2, dropout=dropout, concat=False)
        self.fc = nn.Linear(hidden_dim // 2, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adj):
        h = self.gat1(x, adj)
        h = self.dropout(h)
        h = self.gat2(h, adj)
        h = self.dropout(h)
        logits = self.fc(h)
        return logits, h

    def get_attention_matrix(self):
        """Returns the learned attention matrix for XAI visualization."""
        if self.gat1.last_attention is not None:
            return self.gat1.last_attention.cpu().numpy()
        return None


# ==========================================
# 4. GraphSAGE (Sample and Aggregate)
# ==========================================
class GraphSAGELayer(nn.Module):
    def __init__(self, in_features, out_features, aggregator_type="mean", dropout=0.2):
        super(GraphSAGELayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.aggregator_type = aggregator_type
        self.weight = nn.Parameter(torch.FloatTensor(2 * in_features, out_features))
        self.bias = nn.Parameter(torch.FloatTensor(out_features))
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, adj):
        # Mean aggregation across 1-hop neighbors
        deg = torch.sum(adj, dim=1, keepdim=True).clamp(min=1.0)
        neigh_feat = torch.mm(adj, x) / deg

        combined = torch.cat([x, neigh_feat], dim=1)
        combined = self.dropout(combined)
        out = torch.mm(combined, self.weight) + self.bias
        # L2 normalize embeddings as recommended in GraphSAGE paper
        return F.normalize(F.relu(out), p=2, dim=1)


class GraphSAGEClassifier(nn.Module):
    def __init__(self, in_features, hidden_dim=64, num_classes=2, dropout=0.3):
        super(GraphSAGEClassifier, self).__init__()
        self.sage1 = GraphSAGELayer(in_features, hidden_dim, aggregator_type="mean", dropout=dropout)
        self.sage2 = GraphSAGELayer(hidden_dim, hidden_dim // 2, aggregator_type="mean", dropout=dropout)
        self.fc = nn.Linear(hidden_dim // 2, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adj):
        h = self.sage1(x, adj)
        h = self.dropout(h)
        h = self.sage2(h, adj)
        h = self.dropout(h)
        logits = self.fc(h)
        return logits, h


# ==========================================
# 5. Baseline Tabular Classifier (No Graph)
# ==========================================
class BaselineMLP(nn.Module):
    def __init__(self, in_features, hidden_dim=64, num_classes=2, dropout=0.3):
        super(BaselineMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, x, adj=None):
        logits = self.net(x)
        return logits, x
