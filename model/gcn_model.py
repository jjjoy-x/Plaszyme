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

        # 若已给定 in_dim，则立即构建
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

        # GNN 堆叠
        self.gnn_layers = nn.ModuleList()
        prev_dim = self.in_dim
        for out_dim in self.gnn_dims:
            self.gnn_layers.append(self._get_gnn_layer(prev_dim, out_dim))
            prev_dim = out_dim

        # 读出 + 全连接
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
        """
        # 懒构建：用输入自动推断维度
        if not self._built:
            self._build_layers(data.x.size(-1))
            # 🔑 关键：新建完的层默认在 CPU，把整个模型迁移到输入所在设备
            self.to(data.x.device)

        x, edge_index = data.x, data.edge_index

        # 保证 edge_index 与 x 在同一设备（有些数据的 edge_index 仍在 CPU）
        if edge_index.device != x.device:
            edge_index = edge_index.to(x.device)

        # 没有 batch 属性时，补一个，并放到同一设备
        if hasattr(data, 'batch') and data.batch is not None:
            batch = data.batch
            if batch.device != x.device:
                batch = batch.to(x.device)
        else:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # 逐层 GNN
        gnn_outputs = []
        for layer in self.gnn_layers:
            x = layer(x, edge_index)
            gnn_outputs.append(x)

        # 拼接所有 GNN 层输出
        x = torch.cat(gnn_outputs, dim=-1)

        # 残基级输出
        if self.use_residue_level_output:
            return self.output_layer(x)  # [N, out_dim]

        # 图级 readout + FC
        x = global_mean_pool(x, batch)  # [B, hidden]
        x = self.readout(x)
        for layer in self.fc_layers:
            x = layer(x)
        return self.output_layer(x)  # [B, out_dim]

    def predict(self, data: Batch) -> torch.Tensor:
        """推理接口，自动关闭 dropout 与梯度计算"""
        self.eval()
        with torch.no_grad():
            return self.forward(data)