# visualization.py

import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torch.utils.tensorboard import SummaryWriter


def log_confusion_matrix(y_true, y_pred, class_names, save_path):
    """
    绘制并保存混淆矩阵
    Plot and save confusion matrix to image file
    """
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def log_per_class_accuracy(y_true, y_pred, class_names, save_path):
    """
    统计每类准确率并绘图保存
    Plot and save per-class accuracy
    """
    acc = []
    for i, label in enumerate(class_names):
        total = sum(yt == i for yt in y_true)
        correct = sum((yt == i and yp == i) for yt, yp in zip(y_true, y_pred))
        acc.append(correct / total if total else 0)

    plt.figure(figsize=(10, 4))
    plt.bar(class_names, acc)
    plt.ylabel("Accuracy")
    plt.xticks(rotation=45, ha='right')
    plt.title("Per-Class Accuracy")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def log_curve(writer: SummaryWriter, tag: str, values: list, epochs: list):
    """
    将训练过程中的数值写入 TensorBoard 曲线
    Log training curve to TensorBoard

    Args:
        writer: TensorBoard writer
        tag: 标签名 (如 'Loss/train')
        values: 数值列表
        epochs: 对应 epoch 编号
    """
    for epoch, val in zip(epochs, values):
        writer.add_scalar(tag, val, epoch)


def log_weights_histogram(writer: SummaryWriter, model, epoch: int):
    """
    将模型各层权重分布写入 TensorBoard 直方图
    Log model weight histograms to TensorBoard

    Args:
        writer: TensorBoard writer
        model: 模型
        epoch: 当前 epoch
    """
    for name, param in model.named_parameters():
        if param.requires_grad:
            writer.add_histogram(name, param.detach().cpu().numpy(), epoch)