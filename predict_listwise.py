#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plaszyme — 预测脚本
====================================================

该脚本基于训练好的模型权重，对输入的PDB文件进行酶-塑料相互作用预测。

作者：Shuleihe (School of Science, Xi'an Jiaotong-Liverpool University)
XJTLU_AI_China — 2025 iGEM Team Plaszyme
"""

import logging
import os
import sys
import warnings
from typing import Dict, List, Tuple, Optional

import numpy as np

np.seterr(over="ignore", invalid="ignore", divide="ignore")
import torch
import torch.nn as nn
import pandas as pd

# ===============================================================================
# 配置参数 - 在此处修改您的设置
# ===============================================================================

# 模型和数据路径配置
MODEL_PATH = "/Users/shulei/PycharmProjects/Plaszyme/train_script/train_results/gnn_cos/best_cos.pt"  # 训练好的模型路径
PDB_ROOT = "/Users/shulei/PycharmProjects/Plaszyme/dataset/predicted_xid/pdb"  # PDB文件根目录
PT_OUT_ROOT = "/Users/shulei/PycharmProjects/Plaszyme/dataset/predicted_xid/pt"  # 图文件输出目录
SDF_ROOT = "/Users/shulei/PycharmProjects/Plaszyme/plastic/mols_for_unimol_10_sdf_new"  # SDF文件目录

# 预测配置
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # 计算设备
TOP_K = 10  # 返回前K个最高分数的结果
OUTPUT_DIR = "./prediction_results/1"  # 输出目录

# 要预测的塑料列表配置
USE_ALL_SDF_PLASTICS = True  # 是否使用SDF目录中的全部塑料
DEFAULT_PLASTICS = [
    "PET", "PE",
]  # 当USE_ALL_SDF_PLASTICS=False时使用的塑料列表

# 要预测的PDB文件配置
# 方式1: 单个PDB文件
SINGLE_PDB = "/Users/shulei/PycharmProjects/Plaszyme/dataset/predicted_xid/pdb/X0045.pdb"  # 例如: "/path/to/enzyme.pdb"

# 方式2: PDB文件列表
PDB_LIST = [
    # 例如:
    # "/path/to/enzyme1.pdb",
    # "/path/to/enzyme2.pdb",
]

# 方式3: 从目录批量处理
PDB_DIRECTORY = None  # 例如: "/path/to/pdb_directory/"

# 日志配置
VERBOSE = True  # 是否显示详细日志

# ===============================================================================
# 以下是核心代码，一般无需修改
# ===============================================================================

# 项目内部依赖
from src.builders.gnn_builder import GNNProteinGraphBuilder, BuilderConfig as GNNBuilderConfig
from src.builders.gvp_builder import GVPProteinGraphBuilder, BuilderConfig as GVPBuilderConfig
from src.models.gnn.backbone import GNNBackbone
from src.models.gvp.backbone import GVPBackbone
from src.models.seq_mlp.backbone import SeqBackbone
from src.models.plastic_backbone import PolymerTower
from src.plastic.descriptors_rdkit import PlasticFeaturizer
from src.models.interaction_head import InteractionHead


class TwinProjector(nn.Module):
    """蛋白/塑料各自线性投影到同一空间"""

    def __init__(self, in_e: int, in_p: int, out: int):
        super().__init__()
        self.proj_e = nn.Linear(in_e, out, bias=False)
        self.proj_p = nn.Linear(in_p, out, bias=False)
        nn.init.xavier_uniform_(self.proj_e.weight)
        nn.init.xavier_uniform_(self.proj_p.weight)

    def forward(self, e: torch.Tensor, p: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.proj_e(e), self.proj_p(p)


class PlaszymePredictor:
    """Plaszyme预测器类"""

    def __init__(self,
                 model_path: str,
                 pdb_root: str,
                 pt_out_root: str,
                 device: str = "cpu",
                 plastic_config_path: Optional[str] = None):
        """
        初始化预测器

        Args:
            model_path: 训练好的模型权重文件路径
            pdb_root: PDB文件根目录
            pt_out_root: 图文件输出目录
            device: 计算设备
            plastic_config_path: 塑料特征化配置文件路径
        """
        self.device = device
        self.model_path = model_path
        self.pdb_root = pdb_root
        self.pt_out_root = pt_out_root

        # 加载模型配置和权重
        self._load_model()

        # 初始化塑料特征化器
        self.plastic_featurizer = PlasticFeaturizer(config_path=plastic_config_path)

        # 初始化蛋白质图构建器
        self._init_protein_builder()

        logging.info(f"预测器初始化完成，使用设备: {self.device}")

    def _load_model(self):
        """加载训练好的模型，处理懒构建问题"""
        checkpoint = torch.load(self.model_path, map_location=self.device)
        self.cfg = checkpoint["cfg"]

        logging.info(f"模型配置: {self.cfg}")

        # 预先构建模型架构（延迟实际权重加载）
        self._build_model_architecture()

        # 保存权重字典，稍后在第一次前向传播后加载
        self.saved_weights = {
            "enzyme_backbone": checkpoint["enzyme_backbone"],
            "plastic_tower": checkpoint["plastic_tower"],
            "projector": checkpoint["projector"],
            "inter_head": checkpoint["inter_head"],
        }

        self._weights_loaded = False
        logging.info(f"成功加载模型检查点: {self.model_path}")

    def _build_model_architecture(self):
        """构建模型架构（不加载权重）"""
        # 处理可能的键名不一致问题
        backbone_type = self.cfg.get("backbone") or self.cfg.get("back", "GNN")
        emb_dim_enz = self.cfg["emb_dim_enz"]
        emb_dim_pl = self.cfg["emb_dim_pl"]
        proj_dim = self.cfg["proj_dim"]

        # 构建蛋白质主干网络（使用懒构建）
        if backbone_type == "GNN":
            self.enzyme_backbone = GNNBackbone(
                conv_type="gcn",
                hidden_dims=[128, 128, 128],
                out_dim=emb_dim_enz,
                dropout=0.1,
                residue_logits=False
            ).to(self.device)
        elif backbone_type == "GVP":
            self.enzyme_backbone = GVPBackbone(
                hidden_dims=[(128, 16), (128, 16), (128, 16)],
                out_dim=emb_dim_enz,
                dropout=0.1,
                residue_logits=False
            ).to(self.device)
        else:  # MLP
            self.enzyme_backbone = SeqBackbone(
                hidden_dims=[256, 256, 256],
                out_dim=emb_dim_enz,
                dropout=0.1,
                pool="mean",
                residue_logits=False,
                feature_priority=["seq_x", "x"]
            ).to(self.device)

        # 塑料相关模型稍后构建（需要知道塑料特征维度）
        self.plastic_tower = None
        self.projector = None
        self.inter_head = None

        # 保存配置用于后续构建
        self._emb_dim_enz = emb_dim_enz
        self._emb_dim_pl = emb_dim_pl
        self._proj_dim = proj_dim
        self._interaction = self.cfg["interaction"]
        self._bilinear_rank = self.cfg.get("bilinear_rank", 64)

    def _init_protein_builder(self):
        """初始化蛋白质图构建器"""
        # 处理可能的键名不一致问题
        backbone_type = self.cfg.get("backbone") or self.cfg.get("back", "GNN")

        if backbone_type == "GNN":
            builder_cfg = GNNBuilderConfig(
                pdb_dir=self.pdb_root,
                out_dir=self.pt_out_root,
                radius=10.0,
                embedder=[{"name": "esm"}]
            )
            self.protein_builder = GNNProteinGraphBuilder(builder_cfg, edge_mode="none")
        elif backbone_type == "GVP":
            builder_cfg = GVPBuilderConfig(
                pdb_dir=self.pdb_root,
                out_dir=self.pt_out_root,
                radius=10.0,
                embedder=[{"name": "esm"}]
            )
            self.protein_builder = GVPProteinGraphBuilder(builder_cfg)
        else:  # MLP
            builder_cfg = GNNBuilderConfig(
                pdb_dir=self.pdb_root,
                out_dir=self.pt_out_root,
                radius=10.0,
                embedder=[{"name": "esm"}]
            )
            self.protein_builder = GNNProteinGraphBuilder(builder_cfg, edge_mode="none")

    def _ensure_plastic_models_built(self, plastic_dim: int):
        """确保塑料相关模型已构建"""
        if self.plastic_tower is None:
            self.plastic_tower = PolymerTower(
                in_dim=plastic_dim,
                hidden_dims=[256, 128],
                out_dim=self._emb_dim_pl,
                dropout=0.1
            ).to(self.device)

            self.projector = TwinProjector(
                self._emb_dim_enz,
                self._emb_dim_pl,
                self._proj_dim
            ).to(self.device)

            self.inter_head = InteractionHead(
                self._proj_dim,
                mode=self._interaction,
                rank=self._bilinear_rank
            ).to(self.device)

    def _load_weights_after_forward(self):
        """在第一次前向传播后加载权重"""
        if not self._weights_loaded:
            # 对于懒构建的模型，需要在构建完成后才能加载权重
            if self.enzyme_backbone._built if hasattr(self.enzyme_backbone, '_built') else True:
                self.enzyme_backbone.load_state_dict(self.saved_weights["enzyme_backbone"], strict=True)
                logging.info("已加载enzyme_backbone权重")

            if self.plastic_tower is not None:
                self.plastic_tower.load_state_dict(self.saved_weights["plastic_tower"], strict=True)
                self.projector.load_state_dict(self.saved_weights["projector"], strict=True)
                self.inter_head.load_state_dict(self.saved_weights["inter_head"], strict=True)
                logging.info("已加载塑料相关模型权重")

            # 设置为评估模式
            self.enzyme_backbone.eval()
            if self.plastic_tower is not None:
                self.plastic_tower.eval()
                self.projector.eval()
                self.inter_head.eval()

            self._weights_loaded = True

    def _process_protein(self, pdb_path: str) -> torch.Tensor:
        """处理蛋白质PDB文件"""
        pdb_name = os.path.splitext(os.path.basename(pdb_path))[0]

        try:
            # 使用正确的build_one方法
            data, misc = self.protein_builder.build_one(pdb_path, name=pdb_name)

            if data is None:
                raise ValueError(f"build_one返回了None: {pdb_path}")

            logging.info(f"成功构建蛋白质图: {pdb_name}")
            return data.to(self.device)

        except Exception as e:
            logging.error(f"构建蛋白质图失败: {e}")

            # 备用方法：尝试从缓存加载
            try:
                pt_path = os.path.join(self.pt_out_root, f"{pdb_name}.pt")
                if os.path.exists(pt_path):
                    graph = torch.load(pt_path, map_location='cpu')
                    logging.info(f"从缓存加载图: {pt_path}")
                    return graph.to(self.device)
                else:
                    # 尝试其他可能的文件名模式
                    possible_names = [
                        f"{pdb_name}_graph.pt",
                        f"{pdb_name}_processed.pt",
                        f"graph_{pdb_name}.pt"
                    ]
                    for alt_name in possible_names:
                        alt_path = os.path.join(self.pt_out_root, alt_name)
                        if os.path.exists(alt_path):
                            graph = torch.load(alt_path, map_location='cpu')
                            logging.info(f"从缓存加载图: {alt_path}")
                            return graph.to(self.device)

                    raise FileNotFoundError(f"找不到缓存文件: {pt_path}")

            except Exception as e2:
                raise ValueError(f"无法处理PDB文件 {pdb_path}。构建失败: {e}，缓存加载也失败: {e2}")

    def _get_plastic_features(self, plastic_names: List[str]) -> Tuple[torch.Tensor, List[str]]:
        """获取塑料特征"""
        features = []
        valid_names = []

        for name in plastic_names:
            try:
                # 尝试从SDF文件获取特征
                sdf_path = os.path.join(SDF_ROOT, f"{name}.sdf")
                mol_path = os.path.join(SDF_ROOT, f"{name}.mol")

                feat = None
                if os.path.exists(sdf_path):
                    feat = self.plastic_featurizer.featurize_file(sdf_path)
                elif os.path.exists(mol_path):
                    feat = self.plastic_featurizer.featurize_file(mol_path)
                else:
                    logging.warning(f"找不到塑料 {name} 的分子文件 (.sdf/.mol)")
                    continue

                if feat is not None:
                    features.append(feat)
                    valid_names.append(name)
                    logging.debug(f"成功获取塑料 {name} 的特征，维度: {feat.shape}")
                else:
                    logging.warning(f"无法从文件提取塑料 {name} 的特征")

            except Exception as e:
                logging.warning(f"处理塑料 {name} 时出错: {e}")

        if not features:
            raise ValueError("没有有效的塑料特征")

        # 确保所有特征维度一致
        max_dim = max(f.shape[0] for f in features)
        padded_features = []
        for feat in features:
            if feat.shape[0] < max_dim:
                padded = torch.zeros(max_dim, dtype=torch.float32)
                padded[:feat.shape[0]] = feat
                padded_features.append(padded)
            else:
                padded_features.append(feat)

        logging.info(f"成功获取 {len(valid_names)} 种塑料的特征，特征维度: {max_dim}")
        return torch.stack(padded_features), valid_names

    @torch.no_grad()
    def predict(self,
                pdb_path: str,
                plastic_names: Optional[List[str]] = None,
                top_k: int = 10) -> pd.DataFrame:
        """
        预测单个PDB文件与塑料的相互作用

        Args:
            pdb_path: PDB文件路径
            plastic_names: 塑料名称列表，如果为None则使用默认列表
            top_k: 返回前k个最高分数的结果

        Returns:
            包含预测结果的DataFrame
        """
        if not os.path.exists(pdb_path):
            raise FileNotFoundError(f"PDB文件不存在: {pdb_path}")

        if plastic_names is None:
            plastic_names = get_plastic_list()  # 使用配置的塑料列表

        # 处理蛋白质
        protein_graph = self._process_protein(pdb_path)

        # 第一次前向传播（触发懒构建）
        enzyme_vec = self.enzyme_backbone(protein_graph)

        # 获取塑料特征
        plastic_features, valid_plastic_names = self._get_plastic_features(plastic_names)
        plastic_dim = plastic_features.shape[1]

        # 确保塑料模型已构建
        self._ensure_plastic_models_built(plastic_dim)

        # 在模型完全构建后加载权重
        self._load_weights_after_forward()

        # 重新进行前向传播（使用加载的权重）
        enzyme_vec = self.enzyme_backbone(protein_graph)

        # 移动到设备
        plastic_features = plastic_features.to(self.device)

        # 前向传播
        plastic_vec = self.plastic_tower(plastic_features)
        z_e, z_p = self.projector(enzyme_vec, plastic_vec)

        # 计算相互作用分数
        scores = self.inter_head.score(z_e, z_p)
        scores = scores.cpu().numpy()

        # 创建结果DataFrame
        results = pd.DataFrame({
            'plastic_name': valid_plastic_names,
            'interaction_score': scores
        })

        # 按分数降序排列
        results = results.sort_values('interaction_score', ascending=False)
        results = results.reset_index(drop=True)

        # 添加排名
        results['rank'] = range(1, len(results) + 1)

        # 返回前k个结果
        if top_k > 0:
            results = results.head(top_k)

        return results

    def predict_batch(self,
                      pdb_paths: List[str],
                      plastic_names: Optional[List[str]] = None,
                      top_k: int = 10) -> Dict[str, pd.DataFrame]:
        """批量预测多个PDB文件"""
        results = {}

        for pdb_path in pdb_paths:
            pdb_name = os.path.splitext(os.path.basename(pdb_path))[0]
            try:
                result = self.predict(pdb_path, plastic_names, top_k)
                results[pdb_name] = result
                logging.info(f"成功预测 {pdb_name}")
            except Exception as e:
                logging.error(f"预测 {pdb_name} 时出错: {e}")
                results[pdb_name] = None

        return results


def setup_logging(verbose: bool = False):
    """设置日志"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='[%(asctime)s] %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def get_all_sdf_plastics(sdf_root: str) -> List[str]:
    """从SDF目录获取所有可用的塑料名称"""
    plastic_names = []

    if not os.path.isdir(sdf_root):
        logging.warning(f"SDF目录不存在: {sdf_root}")
        return plastic_names

    try:
        for filename in os.listdir(sdf_root):
            if filename.lower().endswith(('.sdf', '.mol')):
                # 移除文件扩展名作为塑料名称
                plastic_name = os.path.splitext(filename)[0]
                plastic_names.append(plastic_name)

        plastic_names.sort()  # 按字母顺序排序
        logging.info(f"从SDF目录找到 {len(plastic_names)} 种塑料: {plastic_names}")
        return plastic_names

    except Exception as e:
        logging.error(f"读取SDF目录时出错: {e}")
        return []


def get_plastic_list() -> List[str]:
    """根据配置获取要预测的塑料列表"""
    if USE_ALL_SDF_PLASTICS:
        plastics = get_all_sdf_plastics(SDF_ROOT)
        if not plastics:
            logging.warning("无法从SDF目录获取塑料列表，使用默认列表")
            return DEFAULT_PLASTICS
        return plastics
    else:
        return DEFAULT_PLASTICS


def get_pdb_files():
    """根据配置获取要处理的PDB文件列表"""
    pdb_files = []

    if SINGLE_PDB:
        if os.path.exists(SINGLE_PDB):
            pdb_files.append(SINGLE_PDB)
        else:
            logging.error(f"指定的PDB文件不存在: {SINGLE_PDB}")

    if PDB_LIST:
        for pdb_path in PDB_LIST:
            if os.path.exists(pdb_path):
                pdb_files.append(pdb_path)
            else:
                logging.warning(f"PDB文件不存在，跳过: {pdb_path}")

    if PDB_DIRECTORY:
        if os.path.isdir(PDB_DIRECTORY):
            for filename in os.listdir(PDB_DIRECTORY):
                if filename.lower().endswith(('.pdb', '.PDB')):
                    pdb_files.append(os.path.join(PDB_DIRECTORY, filename))
        else:
            logging.error(f"指定的PDB目录不存在: {PDB_DIRECTORY}")

    return pdb_files


def main():
    """主函数"""
    setup_logging(VERBOSE)
    warnings.filterwarnings("ignore", category=UserWarning)

    # 检查模型文件
    if not os.path.exists(MODEL_PATH):
        logging.error(f"模型文件不存在: {MODEL_PATH}")
        sys.exit(1)

    # 获取要处理的PDB文件
    pdb_files = get_pdb_files()

    if not pdb_files:
        logging.error("没有找到要处理的PDB文件，请检查配置")
        sys.exit(1)

    logging.info(f"找到 {len(pdb_files)} 个PDB文件待处理")

    try:
        # 初始化预测器
        predictor = PlaszymePredictor(
            model_path=MODEL_PATH,
            pdb_root=PDB_ROOT,
            pt_out_root=PT_OUT_ROOT,
            device=DEVICE
        )

        # 创建输出目录
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        # 获取塑料列表
        plastic_list = get_plastic_list()
        logging.info(f"将预测 {len(plastic_list)} 种塑料的相互作用")

        if len(pdb_files) == 1:
            # 单个文件预测
            pdb_path = pdb_files[0]
            result = predictor.predict(
                pdb_path=pdb_path,
                plastic_names=plastic_list,
                top_k=TOP_K
            )

            pdb_name = os.path.splitext(os.path.basename(pdb_path))[0]
            print(f"\n=== {pdb_name} 预测结果 ===")
            print(result.to_string(index=False))

            # 保存结果
            output_path = os.path.join(OUTPUT_DIR, f"{pdb_name}_predictions.csv")
            result.to_csv(output_path, index=False)
            print(f"\n结果已保存到: {output_path}")

        else:
            # 批量预测
            results = predictor.predict_batch(
                pdb_paths=pdb_files,
                plastic_names=plastic_list,
                top_k=TOP_K
            )

            # 输出和保存结果
            for pdb_name, result in results.items():
                if result is not None:
                    print(f"\n=== {pdb_name} 预测结果 ===")
                    print(result.to_string(index=False))

                    output_path = os.path.join(OUTPUT_DIR, f"{pdb_name}_predictions.csv")
                    result.to_csv(output_path, index=False)

            print(f"\n批量预测完成，结果已保存到: {OUTPUT_DIR}")

    except Exception as e:
        logging.error(f"预测过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()