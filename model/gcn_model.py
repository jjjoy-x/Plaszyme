import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool
from torch_geometric.data import Batch
from torch_geometric.nn import GCNConv, GATConv
from typing import Optional


class DeepFRIModel(nn.Module):
    """
    PyTorch Geometric 实现的 DeepFRI 风格模型
    ----------------------------------------
    该模型结合 ESM 表达的节点特征与图神经网络（如 GCN 或 GAT），
    用于图级别或残基级别的蛋白质功能预测。

    Args:
        gnn_type (str): GNN 类型（支持 'gcn' 或 'gat'）
        gnn_dims (List[int]): 每层 GNN 的输出维度
        fc_dims (List[int]): 全连接层的每层维度
        out_dim (int): 最终输出维度（例如类别数）
        dropout (float): Dropout 概率，默认 0.3
        use_residue_level_output (bool): 是否输出残基级预测（默认 False）
        in_dim (Optional[int]): 输入特征维度，若为 None 将自动从输入推断
    """

    def __init__(
            self,
            gnn_type: str,
            gnn_dims: list[int],
            fc_dims: list[int],
            out_dim: int,
            dropout: float = 0.3,
            use_residue_level_output: bool = False,
            in_dim: Optional[int] = None,
    ):
        super(DeepFRIModel, self).__init__()
        self.in_dim = in_dim
        self.gnn_type_str = gnn_type.lower()
        self.gnn_dims = gnn_dims
        self.fc_dims = fc_dims
        self.out_dim = out_dim
        self.dropout_p = dropout
        self.use_residue_level_output = use_residue_level_output
        self._built = False

        # ✅ 如果 in_dim 已给定，则立即构建
        if self.in_dim is not None:
            self._build_layers(self.in_dim)

    def _get_gnn_layer(self, in_dim, out_dim):
        if self.gnn_type_str == 'gcn':
            return GCNConv(in_dim, out_dim)
        elif self.gnn_type_str == 'gat':
            return GATConv(in_dim, out_dim, heads=1, concat=False)  # 保持输出维度一致
        else:
            raise ValueError(f"Unsupported GNN type: {self.gnn_type_str}")

    def _build_layers(self, detected_in_dim: int):
        if self.in_dim is None:
            self.in_dim = detected_in_dim
            print(f"[INFO] Auto-detected input dimension: {self.in_dim}")
        elif self.in_dim != detected_in_dim:
            print(f"[WARNING] Specified in_dim ({self.in_dim}) != input ({detected_in_dim}), using specified.")

        self.gnn_layers = nn.ModuleList()
        prev_dim = self.in_dim
        for out_dim in self.gnn_dims:
            self.gnn_layers.append(self._get_gnn_layer(prev_dim, out_dim))
            prev_dim = out_dim

        self.readout = nn.Sequential(
            nn.Linear(sum(self.gnn_dims), self.fc_dims[0]),
            nn.ReLU(),
            nn.Dropout(self.dropout_p)
        )

        self.fc_layers = nn.ModuleList()
        for i in range(len(self.fc_dims) - 1):
            self.fc_layers.append(nn.Linear(self.fc_dims[i], self.fc_dims[i + 1]))
            self.fc_layers.append(nn.ReLU())
            self.fc_layers.append(nn.Dropout(self.dropout_p))

        self.output_layer = nn.Linear(self.fc_dims[-1], self.out_dim)
        self._built = True

    def forward(self, data: Batch) -> torch.Tensor:
        """
        Forward pass through GNN + Readout + Fully Connected layers
        前向传播：GNN + Pooling + FC 分类

        Args:
            data (Batch): PyG Batch 对象，包含 x (node features), edge_index (图结构), batch (图索引)

        Returns:
            Tensor: [B, out_dim] 图级输出 或 [N, out_dim] 残基级输出
        """
        if not self._built:
            self._build_layers(data.x.size(-1))  # 自动推断输入维度

        x, edge_index = data.x, data.edge_index
        batch = data.batch if hasattr(data, 'batch') else torch.zeros(x.size(0), dtype=torch.long)

        gnn_outputs = []
        for layer in self.gnn_layers:
            x = layer(x, edge_index)
            gnn_outputs.append(x)

        x = torch.cat(gnn_outputs, dim=-1)  # 拼接所有 GNN 层输出

        if self.use_residue_level_output:
            return self.output_layer(x)  # 返回残基级输出 [N, out_dim]

        # 图级别 readout（全局平均池化）
        x = global_mean_pool(x, batch)  # [B, hidden_dim]
        x = self.readout(x)
        for layer in self.fc_layers:
            x = layer(x)
        return self.output_layer(x)  # [B, out_dim]

    def predict(self, data: Batch) -> torch.Tensor:
        """
        Wrapper for inference
        推理接口，自动关闭 dropout 与梯度计算
        """
        self.eval()
        with torch.no_grad():
            return self.forward(data)