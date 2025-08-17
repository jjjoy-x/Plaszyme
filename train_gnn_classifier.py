import os
import json
import random
import torch
import argparse
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import Data, DataLoader
from model.gcn_model import DeepFRIModel
from torch.serialization import add_safe_globals
from torch_geometric.data import Data, DataEdgeAttr

from utils.visualization import (
    log_confusion_matrix,
    log_per_class_accuracy,
    log_curve,
    log_weights_histogram
)

# 一致性设置（设定随机种子）
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)

def load_dataset(dataset_dir):
    """
    加载全部 .pt 样本
    Load all graph samples from directory.
    使用 PyTorch 2.6+ 推荐方式，添加安全反序列化支持
    """

    # 添加 PyG 中常用的数据结构到安全白名单
    add_safe_globals([Data, DataEdgeAttr])

    all_files = [os.path.join(dataset_dir, f) for f in os.listdir(dataset_dir) if f.endswith(".pt")]
    all_graphs = [torch.load(f, weights_only=False) for f in all_files]
    return all_graphs

def split_dataset(graphs, split_ratio=(0.8, 0.2)):
    """
    随机划分训练集和验证集
    Split dataset into training and validation sets
    """
    random.shuffle(graphs)
    n = len(graphs)
    n_train = int(split_ratio[0] * n)
    return graphs[:n_train], graphs[n_train:]

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        output = model(batch)  # [B, num_classes]
        loss = criterion(output, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            output = model(batch)  # [B, num_classes]
            preds = output.argmax(dim=1)
            correct += (preds == batch.y).sum().item()
            total += batch.y.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch.y.cpu().numpy())
    acc = correct / total if total > 0 else 0
    return acc, all_preds, all_labels

def main(args):
    # === 初始化环境 ===
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    writer = SummaryWriter(log_dir=args.log_dir)

    # === 加载标签映射 ===
    with open(args.label_map, "r") as f:
        label2id = json.load(f)
    id2label = {v: k for k, v in label2id.items()}
    num_classes = len(label2id)

    # === 加载图数据 ===
    graphs = load_dataset(args.dataset_dir)
    train_graphs, val_graphs = split_dataset(graphs)

    train_loader = DataLoader(train_graphs, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=args.batch_size)

    # === 初始化模型 ===
    model = DeepFRIModel(
        gnn_type=args.gnn_type,
        gnn_dims=args.gnn_dims,
        fc_dims=args.fc_dims,
        out_dim=num_classes,
        dropout=args.dropout
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    # === 训练主循环 ===
    best_acc = 0
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_acc, val_preds, val_labels = evaluate(model, val_loader, device)

        lr = optimizer.param_groups[0]['lr']

        print(f"Epoch {epoch:02d} | Loss: {train_loss:.4f} | Val Acc: {val_acc:.4f} | LR: {lr:.6f}")

        # === TensorBoard 可视化 ===
        log_curve(writer, "Loss/train", [train_loss], [epoch])
        log_curve(writer, "Acc/val", [val_acc], [epoch])
        log_curve(writer, "LearningRate", [lr], [epoch])

        log_weights_histogram(writer, model, epoch)

        class_names = [id2label[i] for i in range(num_classes)]
        cm_path = os.path.join(args.save_dir, f"confusion_matrix_epoch{epoch:02d}.png")
        acc_path = os.path.join(args.save_dir, f"per_class_acc_epoch{epoch:02d}.png")

        log_confusion_matrix(val_labels, val_preds, class_names, cm_path)
        log_per_class_accuracy(val_labels, val_preds, class_names, acc_path)

        scheduler.step()

        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(args.save_dir, "best_model.pt"))

    print(f"\nTraining completed. Best Val Acc: {best_acc:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train GNN for Plastic Classification")
    parser.add_argument("--dataset_dir", type=str, required=True, help="Path to folder with .pt graph files")
    parser.add_argument("--label_map", type=str, required=True, help="Path to label2id.json")
    parser.add_argument("--save_dir", type=str, default="checkpoints", help="Where to save model checkpoints")
    parser.add_argument("--log_dir", type=str, default="logs", help="TensorBoard log directory")

    # Hyperparameters
    parser.add_argument("--gnn_type", type=str, default="gcn", choices=["gcn", "gat"], help="GNN type")
    parser.add_argument("--gnn_dims", nargs='+', type=int, default=[128, 128], help="Hidden dimensions of GNN layers")
    parser.add_argument("--fc_dims", nargs='+', type=int, default=[128, 64], help="Hidden dimensions of FC layers")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout rate")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")

    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    main(args)