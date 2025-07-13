import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv


class GCNLayer(nn.Module):
    """
    PyG 风格的 GCN 层封装
    ---------------------
    使用 torch_geometric.nn.GCNConv 实现

    Args:
        in_dim (int): 输入特征维度
        out_dim (int): 输出特征维度
        dropout (float): dropout 概率
    """

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1):
        super(GCNLayer, self).__init__()
        self.conv = GCNConv(in_dim, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (Tensor): [num_nodes, in_dim] 节点特征矩阵
            edge_index (LongTensor): [2, num_edges] COO 格式的稀疏边索引

        Returns:
            h (Tensor): [num_nodes, out_dim] 输出节点嵌入
        """
        x = self.dropout(x)
        h = self.conv(x, edge_index)
        return h

class GATLayer(nn.Module):
    """
    Graph Attention Network Layer (GAT 层)
    -------------------------------------
    实现来源: https://arxiv.org/abs/1710.10903

    Attributes:
        in_dim (int): Input features dimension per node.
        out_dim (int): Output features dimension per node.
        dropout (float): Dropout probability applied to attention scores.
        alpha (float): LeakyReLU negative slope for attention coefficient.
    """

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1, alpha: float = 0.2):
        super(GATLayer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        # 节点特征变换矩阵
        self.W = nn.Parameter(torch.empty(size=(in_dim, out_dim)))
        # 注意力权重参数 (用于拼接后的 [hi || hj])
        self.a = nn.Parameter(torch.empty(size=(2 * out_dim, 1)))

        self.leakyrelu = nn.LeakyReLU(alpha)
        self.dropout = nn.Dropout(dropout)

        # 参数初始化
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x (Tensor): Input node features, shape [B, N, in_dim]
            adj (Tensor): Adjacency matrix, shape [B, N, N], binary mask

        Returns:
            h_prime (Tensor): Output node embeddings, shape [B, N, out_dim]
            attention (Tensor): Attention coefficients, shape [B, N, N]
        """
        B, N, _ = x.size()

        # 1. 节点线性变换 h = xW
        h = torch.matmul(x, self.W)  # [B, N, out_dim]

        # 2. 构造所有节点对 (hi || hj)
        h_i = h.unsqueeze(2).expand(-1, -1, N, -1)  # [B, N, N, out_dim]
        h_j = h.unsqueeze(1).expand(-1, N, -1, -1)  # [B, N, N, out_dim]
        a_input = torch.cat([h_i, h_j], dim=-1)     # [B, N, N, 2*out_dim]

        # 3. 计算注意力权重 e_ij = LeakyReLU(a^T[hi || hj])
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(-1))  # [B, N, N]

        # 4. 只保留存在边的注意力权重，其余设为 -∞ 再做 softmax
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=-1)
        attention = self.dropout(attention)

        # 5. 聚合邻居特征：sum_j (alpha_ij * h_j)
        h_prime = torch.matmul(attention, h)  # [B, N, out_dim]

        return h_prime, attention