#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plaszyme — Enhanced Prediction Script
====================================================

This script performs enzyme-plastic interaction prediction based on trained
model weights, with comprehensive confidence assessment and detailed prediction
analysis.

Key Features
------------
- Load pre-trained models with lazy construction handling
- Support for multiple backbone architectures (GNN/GVP/MLP)
- Automatic plastic feature extraction from SDF/MOL files
- Confidence metrics and uncertainty quantification
- Batch processing capabilities
- Detailed prediction analysis and categorization

Outputs
-------
- Prediction scores with confidence metrics
- Relative strength and percentile rankings
- Embedding similarity analysis
- Prediction category classification
- Comprehensive CSV reports

Usage
-----
# Configure paths and settings at the top of the script
# Then run directly:
python predict_enhanced.py

Author
------
Shuleihe (School of Science, Xi'an Jiaotong-Liverpool University)
XJTLU_AI_China — iGEM 2025
"""

from __future__ import annotations

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
# Configuration Parameters - Modify Your Settings Here
# ===============================================================================

# Model and data path configuration
MODEL_PATH = "/Users/shulei/PycharmProjects/Plaszyme/train_script/train_results/gnn_bilinear/best_bilinear.pt"
PT_OUT_ROOT = "/Users/shulei/PycharmProjects/Plaszyme/dataset/predicted_xid/pt"
SDF_ROOT = "/Users/shulei/PycharmProjects/Plaszyme/plastic/mols_for_unimol_10_sdf_new"
PDB_ROOT = os.getcwd()

# Prediction configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TOP_K = 10  # Return top K highest scoring results
OUTPUT_DIR = "./prediction_results"
INCLUDE_CONFIDENCE = True  # Include confidence and additional metrics

# Plastic list configuration
USE_ALL_SDF_PLASTICS = True  # Use all plastics from SDF directory
DEFAULT_PLASTICS = [
    'ECOFLEX', 'Impranil', 'LDPE', 'NR', 'Nylon', 'O-PVA', 'P(3HB-co-3MP)', 'P3HP', 'P3HV',
    'P4HB', 'PA', 'PBAT', 'PBS', 'PBSA', 'PBSeT', 'PCL', 'PE', 'PEA', 'PEF', 'PEG', 'PES',
    'PET', 'PHB', 'PHBH', 'PHBV', 'PHBVH', 'PHO', 'PHPV', 'PHV', 'PLA', 'PMCL', 'PPL', 'PS', 'PU', 'PVA'
]  # Used when USE_ALL_SDF_PLASTICS=False

# 若提供，则忽略 USE_ALL_SDF_PLASTICS/SDF_ROOT/DEFAULT_PLASTICS
# 支持：
#  - "/path/to/one.sdf" 或 "/path/to/one.mol"      单个文件
#  - ["/a.sdf", "/b.mol"]                           文件列表
#  - "/path/to/sdf_folder"                          文件夹路径（遍历其中的 .sdf/.mol）
PLASTIC_INPUT: Optional[object] = None


# PDB file configuration - choose one approach
# Support：
#  - "/path/to/one.pdb"              单个文件
#  - ["/path/to/one.pdb", "/path/two.pdb"]   文件列表
#  - "/path/to/pdb_folder"           文件夹路径
PDB_INPUT = ["/Users/shulei/PycharmProjects/Plaszyme/dataset/predicted_xid/pdb/X0001.pdb"]

# Logging configuration
VERBOSE = True  # Show detailed logs

# ===============================================================================
# Core Implementation - Generally No Need to Modify Below
# ===============================================================================

# Project internal dependencies
from src.builders.gnn_builder import GNNProteinGraphBuilder, BuilderConfig as GNNBuilderConfig
from src.builders.gvp_builder import GVPProteinGraphBuilder, BuilderConfig as GVPBuilderConfig
from src.models.gnn.backbone import GNNBackbone
from src.models.gvp.backbone import GVPBackbone
from src.models.seq_mlp.backbone import SeqBackbone
from src.models.plastic_backbone import PolymerTower
from src.plastic.descriptors_rdkit import PlasticFeaturizer
from src.models.interaction_head import InteractionHead


class TwinProjector(nn.Module):
    """Dual linear projection for protein and plastic features to shared space."""

    def __init__(self, in_e: int, in_p: int, out: int):
        """Initialize projector.

        Args:
            in_e: Input dimension for enzyme features
            in_p: Input dimension for plastic features
            out: Output dimension for shared space
        """
        super().__init__()
        self.proj_e = nn.Linear(in_e, out, bias=False)
        self.proj_p = nn.Linear(in_p, out, bias=False)
        nn.init.xavier_uniform_(self.proj_e.weight)
        nn.init.xavier_uniform_(self.proj_p.weight)

    def forward(self, e: torch.Tensor, p: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Project enzyme and plastic features to shared space.

        Args:
            e: Enzyme features [B, De]
            p: Plastic features [N, Dp]

        Returns:
            Tuple of projected features (enzyme, plastic)
        """
        return self.proj_e(e), self.proj_p(p)


class PlaszymePredictor:
    """Enhanced Plaszyme predictor with confidence assessment."""

    def __init__(self,
                 model_path: str,
                 pdb_root: str,
                 pt_out_root: str,
                 device: str = "cpu",
                 plastic_config_path: Optional[str] = None):
        """Initialize predictor.

        Args:
            model_path: Path to trained model weights
            pdb_root: Root directory for PDB files
            pt_out_root: Output directory for graph files
            device: Computing device (cuda/cpu)
            plastic_config_path: Optional plastic featurizer config path
        """
        self.device = device
        self.model_path = model_path
        self.pdb_root = pdb_root
        self.pt_out_root = pt_out_root

        # Load model configuration and weights
        self._load_model()

        # Initialize plastic featurizer
        self.plastic_featurizer = PlasticFeaturizer(config_path=plastic_config_path)

        # Initialize protein graph builder
        self._init_protein_builder()

        print(f"[PREDICTOR] Initialization complete | device={self.device}")

    def _load_model(self):
        """Load trained model with lazy construction handling."""
        checkpoint = torch.load(self.model_path, map_location=self.device)
        self.cfg = checkpoint["cfg"]

        print(f"[MODEL] Config loaded | {self.cfg}")

        # Pre-build model architecture (defer actual weight loading)
        self._build_model_architecture()

        # Save weight dictionaries for later loading after lazy construction
        self.saved_weights = {
            "enzyme_backbone": checkpoint["enzyme_backbone"],
            "plastic_tower": checkpoint["plastic_tower"],
            "projector": checkpoint["projector"],
            "inter_head": checkpoint["inter_head"],
        }

        self._weights_loaded = False
        print(f"[MODEL] Checkpoint loaded successfully")

    def _build_model_architecture(self):
        """Build model architecture without loading weights."""
        # Handle potential key name inconsistencies  处理可能的键名不一致问题
        backbone_type = self.cfg.get("backbone") or self.cfg.get("back", "GNN")
        emb_dim_enz = self.cfg["emb_dim_enz"]
        emb_dim_pl = self.cfg["emb_dim_pl"]
        proj_dim = self.cfg["proj_dim"]

        # Build protein backbone network (with lazy construction)  构建蛋白质主干网络（使用懒构建）
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

        # Plastic-related models built later (need plastic feature dimensions)  塑料相关模型稍后构建
        self.plastic_tower = None
        self.projector = None
        self.inter_head = None

        # Save config for subsequent construction  保存配置用于后续构建
        self._emb_dim_enz = emb_dim_enz
        self._emb_dim_pl = emb_dim_pl
        self._proj_dim = proj_dim
        self._interaction = self.cfg["interaction"]
        self._bilinear_rank = self.cfg.get("bilinear_rank", 64)

    def _init_protein_builder(self):
        """Initialize protein graph builder."""
        # Handle potential key name inconsistencies  处理可能的键名不一致问题
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
        """Ensure plastic-related models are built."""
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
        """Load weights after first forward pass."""
        if not self._weights_loaded:
            # For lazy construction models, need to load weights after construction complete
            # 对于懒构建的模型，需要在构建完成后才能加载权重
            if self.enzyme_backbone._built if hasattr(self.enzyme_backbone, '_built') else True:
                self.enzyme_backbone.load_state_dict(self.saved_weights["enzyme_backbone"], strict=True)
                print("[WEIGHTS] Enzyme backbone weights loaded")

            if self.plastic_tower is not None:
                self.plastic_tower.load_state_dict(self.saved_weights["plastic_tower"], strict=True)
                self.projector.load_state_dict(self.saved_weights["projector"], strict=True)
                self.inter_head.load_state_dict(self.saved_weights["inter_head"], strict=True)
                print("[WEIGHTS] Plastic models weights loaded")

            # Set to evaluation mode
            self.enzyme_backbone.eval()
            if self.plastic_tower is not None:
                self.plastic_tower.eval()
                self.projector.eval()
                self.inter_head.eval()

            self._weights_loaded = True

    def _process_protein(self, pdb_path: str) -> torch.Tensor:
        """Process protein PDB file."""
        pdb_name = os.path.splitext(os.path.basename(pdb_path))[0]

        try:
            # Use correct build_one method  使用正确的build_one方法
            data, misc = self.protein_builder.build_one(pdb_path, name=pdb_name)

            if data is None:
                raise ValueError(f"build_one returned None: {pdb_path}")

            print(f"[PROTEIN] Graph construction successful | name={pdb_name}")
            return data.to(self.device)

        except Exception as e:
            print(f"[PROTEIN] Graph construction failed | name={pdb_name} | error={str(e)}")

            # Fallback: try loading from cache  备用方法：尝试从缓存加载
            try:
                pt_path = os.path.join(self.pt_out_root, f"{pdb_name}.pt")
                if os.path.exists(pt_path):
                    graph = torch.load(pt_path, map_location='cpu')
                    print(f"[PROTEIN] Loaded from cache | path={pt_path}")
                    return graph.to(self.device)
                else:
                    # Try alternative file name patterns  尝试其他可能的文件名模式
                    possible_names = [
                        f"{pdb_name}_graph.pt",
                        f"{pdb_name}_processed.pt",
                        f"graph_{pdb_name}.pt"
                    ]
                    for alt_name in possible_names:
                        alt_path = os.path.join(self.pt_out_root, alt_name)
                        if os.path.exists(alt_path):
                            graph = torch.load(alt_path, map_location='cpu')
                            print(f"[PROTEIN] Loaded from cache | path={alt_path}")
                            return graph.to(self.device)

                    raise FileNotFoundError(f"Cache file not found: {pt_path}")

            except Exception as e2:
                raise ValueError(
                    f"Failed to process PDB file {pdb_path}. Build failed: {e}, cache loading also failed: {e2}")

    def _get_plastic_features(self, plastic_items: List[str]) -> Tuple[torch.Tensor, List[str]]:
        """
        提取塑料特征：plastic_items 既可为‘塑料名’，也可为‘显式文件路径(.sdf/.mol)’。
        - 若是文件路径：直接读取该文件
        - 若是名字：从 SDF_ROOT/{name}.sdf 或 .mol 中寻找
        返回: (features_tensor [N, D], names [N])
        """
        features = []
        valid_names = []

        for item in plastic_items:
            try:
                if os.path.isfile(item) and item.lower().endswith((".sdf", ".mol")):
                    # 显式文件路径
                    feat = self.plastic_featurizer.featurize_file(item)
                    name = os.path.splitext(os.path.basename(item))[0]
                else:
                    # 当作“塑料名”在 SDF_ROOT 中查找
                    name = item
                    sdf_path = os.path.join(SDF_ROOT, f"{name}.sdf")
                    mol_path = os.path.join(SDF_ROOT, f"{name}.mol")
                    feat = None
                    if os.path.exists(sdf_path):
                        feat = self.plastic_featurizer.featurize_file(sdf_path)
                    elif os.path.exists(mol_path):
                        feat = self.plastic_featurizer.featurize_file(mol_path)
                    else:
                        print(f"[PLASTIC] Molecular file not found | name={name}")
                        continue

                if feat is not None:
                    features.append(feat)
                    valid_names.append(name)
                else:
                    print(f"[PLASTIC] Feature extraction failed | item={item}")

            except Exception as e:
                print(f"[PLASTIC] Processing error | item={item} | error={str(e)}")

        if not features:
            raise ValueError("No valid plastic features extracted")

        # 维度对齐
        max_dim = max(f.shape[0] for f in features)
        padded_features = []
        for feat in features:
            if feat.shape[0] < max_dim:
                padded = torch.zeros(max_dim, dtype=torch.float32)
                padded[:feat.shape[0]] = feat
                padded_features.append(padded)
            else:
                padded_features.append(feat)

        print(f"[PLASTIC] Features extracted successfully | count={len(valid_names)} | dim={max_dim}")
        return torch.stack(padded_features), valid_names

    @torch.no_grad()
    def predict(self,
                pdb_path: str,
                plastic_names: Optional[List[str]] = None,
                top_k: int = 10,
                include_confidence: bool = True) -> pd.DataFrame:
        """Predict enzyme-plastic interactions for single PDB file.

        Args:
            pdb_path: Path to PDB file
            plastic_names: List of plastic names, uses default if None
            top_k: Return top k highest scoring results
            include_confidence: Whether to compute confidence metrics

        Returns:
            DataFrame containing prediction results
        """
        if not os.path.exists(pdb_path):
            raise FileNotFoundError(f"PDB file does not exist: {pdb_path}")

        if plastic_names is None:
            plastic_names = get_plastic_list()

        # Process protein  处理蛋白质
        protein_graph = self._process_protein(pdb_path)

        # First forward pass (trigger lazy construction)  第一次前向传播（触发懒构建）
        enzyme_vec = self.enzyme_backbone(protein_graph)

        # Get plastic features  获取塑料特征
        plastic_features, valid_plastic_names = self._get_plastic_features(plastic_names)
        plastic_dim = plastic_features.shape[1]

        # Ensure plastic models are built  确保塑料模型已构建
        self._ensure_plastic_models_built(plastic_dim)

        # Load weights after model construction complete  在模型完全构建后加载权重
        self._load_weights_after_forward()

        # Re-run forward pass (using loaded weights)  重新进行前向传播（使用加载的权重）
        enzyme_vec = self.enzyme_backbone(protein_graph)

        # Move to device
        plastic_features = plastic_features.to(self.device)

        # Forward pass
        plastic_vec = self.plastic_tower(plastic_features)
        z_e, z_p = self.projector(enzyme_vec, plastic_vec)

        # Compute interaction scores  计算相互作用分数
        scores = self.inter_head.score(z_e, z_p)
        scores_np = scores.cpu().numpy()

        # Create base results DataFrame  创建基础结果DataFrame
        results = pd.DataFrame({
            'plastic_name': valid_plastic_names,
            'interaction_score': scores_np
        })

        # Add confidence and additional metrics  添加置信度和额外指标
        if include_confidence:
            results = self._add_confidence_metrics(results, scores, z_e, z_p)

        # Sort by descending score  按分数降序排列
        results = results.sort_values('interaction_score', ascending=False)
        results = results.reset_index(drop=True)

        # Add ranking
        results['rank'] = range(1, len(results) + 1)

        # Reorder columns
        if include_confidence:
            column_order = ['rank', 'plastic_name', 'interaction_score', 'confidence_score',
                            'relative_strength', 'score_percentile', 'embedding_similarity', 'prediction_category']
        else:
            column_order = ['rank', 'plastic_name', 'interaction_score']

        results = results[column_order]

        # Return top k results
        if top_k > 0:
            results = results.head(top_k)

        return results

    def _add_confidence_metrics(self, results: pd.DataFrame, scores: torch.Tensor,
                                z_e: torch.Tensor, z_p: torch.Tensor) -> pd.DataFrame:
        """Add confidence and additional evaluation metrics."""
        scores_np = scores.cpu().numpy()

        # 1. Confidence score (based on softmax probability)  置信度分数（基于softmax概率）
        softmax_probs = torch.softmax(scores / 0.1, dim=0).cpu().numpy()  # temperature=0.1 for sharper distribution
        results['confidence_score'] = softmax_probs

        # 2. Relative strength (relative to maximum score)  相对强度（相对于最高分的比例）
        max_score = scores_np.max()
        min_score = scores_np.min()
        if max_score != min_score:
            relative_strength = (scores_np - min_score) / (max_score - min_score)
        else:
            relative_strength = np.ones_like(scores_np)
        results['relative_strength'] = relative_strength

        # 3. Score percentiles  分数百分位数
        percentiles = []
        for score in scores_np:
            percentile = (scores_np < score).sum() / len(scores_np) * 100
            percentiles.append(percentile)
        results['score_percentile'] = percentiles

        # 4. Embedding similarity (cosine similarity)  嵌入相似度（余弦相似度）
        z_e_norm = torch.nn.functional.normalize(z_e, dim=-1)  # [1, D]
        z_p_norm = torch.nn.functional.normalize(z_p, dim=-1)  # [N, D]
        cosine_sim = torch.mm(z_e_norm, z_p_norm.t()).squeeze(0).cpu().numpy()  # [N]
        results['embedding_similarity'] = cosine_sim

        # 5. Prediction categories (based on score distribution)  预测类别（基于分数分布）
        score_std = scores_np.std()
        score_mean = scores_np.mean()

        if len(scores_np) == 1:
            only_score = scores_np[0]
            if only_score > 0:
                results['prediction_category'] = ["High Interaction"]
            else:
                results['prediction_category'] = ["No Significant Interaction"]
            return results

        def categorize_prediction(score, conf):
            if score > score_mean + score_std and conf > 0.1:
                return "High Interaction"
            elif score > score_mean and conf > 0.05:
                return "Medium Interaction"
            elif score > score_mean - score_std:
                return "Low Interaction"
            else:
                return "No Significant Interaction"

        results['prediction_category'] = [
            categorize_prediction(score, conf)
            for score, conf in zip(scores_np, softmax_probs)
        ]

        return results

    def predict_batch(self,
                      pdb_paths: List[str],
                      plastic_names: Optional[List[str]] = None,
                      top_k: int = 10,
                      include_confidence: bool = True) -> Dict[str, pd.DataFrame]:
        """Batch predict multiple PDB files."""
        results = {}

        for pdb_path in pdb_paths:
            pdb_name = os.path.splitext(os.path.basename(pdb_path))[0]
            try:
                result = self.predict(pdb_path, plastic_names, top_k, include_confidence)
                results[pdb_name] = result
                print(f"[BATCH] Prediction successful | name={pdb_name}")
            except Exception as e:
                print(f"[BATCH] Prediction failed | name={pdb_name} | error={str(e)}")
                results[pdb_name] = None

        return results


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='[%(asctime)s] %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def get_all_sdf_plastics(sdf_root: str) -> List[str]:
    """Get all available plastic names from SDF directory."""
    plastic_names = []

    if not os.path.isdir(sdf_root):
        print(f"[SDF] Directory not found | path={sdf_root}")
        return plastic_names

    try:
        for filename in os.listdir(sdf_root):
            if filename.lower().endswith(('.sdf', '.mol')):
                # Remove file extension as plastic name  移除文件扩展名作为塑料名称
                plastic_name = os.path.splitext(filename)[0]
                plastic_names.append(plastic_name)

        plastic_names.sort()  # Sort alphabetically
        print(f"[SDF] Plastics discovered | count={len(plastic_names)} | names={plastic_names}")
        return plastic_names

    except Exception as e:
        print(f"[SDF] Directory read error | error={str(e)}")
        return []


def get_plastic_list() -> List[str]:
    """Get plastic list based on configuration."""
    if USE_ALL_SDF_PLASTICS:
        plastics = get_all_sdf_plastics(SDF_ROOT)
        if not plastics:
            print("[CONFIG] Unable to get plastics from SDF directory, using default list")
            return DEFAULT_PLASTICS
        return plastics
    else:
        return DEFAULT_PLASTICS

def get_plastic_inputs() -> List[str]:
    """
    返回塑料“条目列表”：每个元素要么是 显式文件路径(.sdf/.mol)，要么是塑料名（用于 SDF_ROOT 中查找）。
    若 PLASTIC_INPUT 提供，则返回文件路径列表（或由文件夹展开得到的列表），并忽略默认库。
    若 PLASTIC_INPUT 为空，则退回 get_plastic_list()（返回的是塑料名列表）。
    """
    if PLASTIC_INPUT is None:
        # 走原逻辑：返回“塑料名”列表
        return get_plastic_list()

    items: List[str] = []
    def _add_path(p: str):
        if os.path.isfile(p) and p.lower().endswith((".sdf", ".mol")):
            items.append(os.path.abspath(p))
        elif os.path.isdir(p):
            for fn in os.listdir(p):
                if fn.lower().endswith((".sdf", ".mol")):
                    items.append(os.path.abspath(os.path.join(p, fn)))
        else:
            print(f"[PLASTIC_INPUT] Skip invalid path: {p}")

    if isinstance(PLASTIC_INPUT, str):
        _add_path(PLASTIC_INPUT)
    elif isinstance(PLASTIC_INPUT, (list, tuple)):
        for x in PLASTIC_INPUT:
            _add_path(str(x))
    else:
        print(f"[PLASTIC_INPUT] Unsupported type: {type(PLASTIC_INPUT)}")
        return []

    # 去重并排序
    items = sorted(list(dict.fromkeys(items)))
    if not items:
        print("[PLASTIC_INPUT] No valid .sdf/.mol found; fallback to SDF_ROOT/DEFAULT.")
        return get_plastic_list()
    print(f"[PLASTIC_INPUT] Using explicit plastic files | count={len(items)}")
    return items

def get_pdb_files(pdb_input) -> List[str]:
    """Resolve pdb file(s) from input.

    Args:
        pdb_input: str or List[str] (file path(s) or folder path)

    Returns:
        List of pdb file paths
    """
    pdb_files: List[str] = []

    # 字符串输入
    if isinstance(pdb_input, str):
        if os.path.isdir(pdb_input):
            # 文件夹 -> 遍历所有 .pdb
            for fn in os.listdir(pdb_input):
                if fn.lower().endswith(".pdb"):
                    pdb_files.append(os.path.join(pdb_input, fn))
        elif os.path.isfile(pdb_input) and pdb_input.lower().endswith(".pdb"):
            pdb_files.append(pdb_input)
        else:
            print(f"[CONFIG] Invalid PDB_INPUT path: {pdb_input}")
        return sorted(pdb_files)

    # 列表输入
    if isinstance(pdb_input, (list, tuple)):
        for item in pdb_input:
            if os.path.isfile(item) and item.lower().endswith(".pdb"):
                pdb_files.append(item)
            elif os.path.isdir(item):
                for fn in os.listdir(item):
                    if fn.lower().endswith(".pdb"):
                        pdb_files.append(os.path.join(item, fn))
            else:
                print(f"[CONFIG] Skip invalid item in PDB_INPUT: {item}")
        return sorted(list(dict.fromkeys(pdb_files)))  # 去重

    print(f"[CONFIG] Unsupported PDB_INPUT type: {type(pdb_input)}")
    return []


def main():
    """Main function."""
    setup_logging(VERBOSE)
    warnings.filterwarnings("ignore", category=UserWarning)

    # Check model file  检查模型文件
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Model file not found | path={MODEL_PATH}")
        sys.exit(1)

    # Get PDB files to process  获取要处理的PDB文件
    pdb_files = get_pdb_files(PDB_INPUT)

    if not pdb_files:
        print("[ERROR] No PDB files found to process, check configuration")
        sys.exit(1)

    print(f"[MAIN] PDB files discovered | count={len(pdb_files)}")

    try:
        # Initialize predictor  初始化预测器
        predictor = PlaszymePredictor(
            model_path=MODEL_PATH,
            pdb_root=PDB_ROOT,
            pt_out_root=PT_OUT_ROOT,
            device=DEVICE
        )

        # Create output directory
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        # Get plastic list  获取塑料列表
        plastic_list = get_plastic_inputs()
        print(f"[MAIN] Will predict interactions | plastic_count={len(plastic_list)}")

        if len(pdb_files) == 1:
            # Single file prediction  单个文件预测
            pdb_path = pdb_files[0]
            result = predictor.predict(
                pdb_path=pdb_path,
                plastic_names=plastic_list,
                top_k=TOP_K,
                include_confidence=INCLUDE_CONFIDENCE
            )

            pdb_name = os.path.splitext(os.path.basename(pdb_path))[0]
            print(f"\n=== {pdb_name} Prediction Results ===")

            # Adjust display format based on confidence information
            if INCLUDE_CONFIDENCE:
                print("Detailed prediction information:")
                print(result.to_string(index=False, float_format='%.4f'))

                # Output interpretive summary  输出解释性总结
                print(f"\nPrediction Summary:")
                top_result = result.iloc[0]
                print(f"• Best matching plastic: {top_result['plastic_name']}")
                print(f"• Interaction score: {top_result['interaction_score']:.4f}")
                print(f"• Confidence: {top_result['confidence_score']:.4f}")
                print(f"• Prediction category: {top_result['prediction_category']}")
                print(f"• Embedding similarity: {top_result['embedding_similarity']:.4f}")

                # Category statistics  分类统计
                categories = result['prediction_category'].value_counts()
                print(f"\nCategory Distribution:")
                for category, count in categories.items():
                    print(f"• {category}: {count} plastics")
            else:
                print(result.to_string(index=False, float_format='%.4f'))

            # Save results
            output_path = os.path.join(OUTPUT_DIR, f"{pdb_name}_predictions.csv")
            result.to_csv(output_path, index=False)
            print(f"\nResults saved to: {output_path}")

        else:
            # Batch prediction  批量预测
            results = predictor.predict_batch(
                pdb_paths=pdb_files,
                plastic_names=plastic_list,
                top_k=TOP_K,
                include_confidence=INCLUDE_CONFIDENCE
            )

            # Output and save results
            for pdb_name, result in results.items():
                if result is not None:
                    print(f"\n=== {pdb_name} Prediction Results ===")
                    if INCLUDE_CONFIDENCE:
                        print(result.to_string(index=False, float_format='%.4f'))
                        # Output brief summary
                        top_result = result.iloc[0]
                        print(
                            f"Best match: {top_result['plastic_name']} (score: {top_result['interaction_score']:.4f}, confidence: {top_result['confidence_score']:.4f})")
                    else:
                        print(result.to_string(index=False, float_format='%.4f'))

                    output_path = os.path.join(OUTPUT_DIR, f"{pdb_name}_predictions.csv")
                    result.to_csv(output_path, index=False)

            print(f"\n[BATCH] Batch prediction completed | results_saved_to={OUTPUT_DIR}")

    except Exception as e:
        print(f"[ERROR] Prediction process failed | error={str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()