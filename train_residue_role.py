#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import os
import csv
import random
from typing import Dict, List, Optional, Tuple, DefaultDict
from collections import defaultdict, Counter

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader as GeoDataLoader

from src.builders.gnn_builder import GNNProteinGraphBuilder, BuilderConfig
from src.heads.residue_role_head import ResidueRoleHead


# ---------------- 配置 ----------------
CONFIG = {
    "seed": 42,
    "device": "cuda" if torch.cuda.is_available() else "cpu",

    "pdb_dir": "/root/autodl-tmp/M-CSA/pdbs",
    "csv_path": "/tmp/pycharm_project_317/dataset/M-CSA/merged_mcsa_ec.csv",
    "cache_dir": None,

    "val_ratio": 0.15,
    "batch_size": 2,
    "num_workers": 0,

    "builder": {
        "radius": 10.0,
        "embedder": [
            {"name": "onehot"},
        ],
        "edge_mode": "rbf",
        "rbf_centers": None,
        "rbf_gamma": None,
        "add_self_loop": False,
    },

    "backbone_cfg": {
        "conv_type": "gine",
        "hidden_dims": [128, 128],
        "dropout": 0.25,
        "gcn_edge_mode": "auto",
        "gine_missing_edge_policy": "zeros",
        "residue_logits": True,
    },
    "freeze_backbone": False,
    "classifier_dims": [256],

    "role_type_priority": ["reactant", "interaction", "spectator", "cofactor", "metal", "modifier", "other"],

    "epochs": 30,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "grad_clip": 5.0,
    "log_every": 50,
    "early_stop_patience": 8,
}


# ------------- 小工具 -------------
def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def list_pdb_files(pdb_dir: str) -> List[str]:
    exts = {".pdb", ".ent", ".mmcif", ".cif"}
    files = []
    for fn in os.listdir(pdb_dir):
        p = os.path.join(pdb_dir, fn)
        if os.path.isfile(p) and os.path.splitext(fn)[1].lower() in exts:
            files.append(p)
    files.sort()
    return files


def _pick(row: dict, *candidates: str) -> Optional[str]:
    for k in candidates:
        if k in row and row[k] is not None:
            v = str(row[k]).strip()
            if v != "":
                return v
    return None


def load_annotations(csv_path: str) -> Dict[Tuple[str, str, int], str]:
    """
    返回 {(pdb_id_lower, chain, resid_int) -> role_type_lower}
    兼容列名：
      - 新：PDB ID, CHAIN ID, RESIDUE NUMBER, ROLE_TYPE
      - 旧：pdb_id, chain, resid_int, role_type
    """
    bucket: DefaultDict[Tuple[str, str, int], List[str]] = defaultdict(list)
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pdb_id = _pick(row, "pdb_id", "PDB ID", "PDB_ID", "PDB")
            chain = _pick(row, "chain", "CHAIN ID", "CHAIN", "CHAIN_ID")
            resid_str = _pick(row, "resid_int", "RESIDUE NUMBER", "RESNUM", "RESSEQ", "RESIDUE_NUMBER")
            role_type = _pick(row, "ROLE_TYPE", "role_type") or _pick(row, "ROLE", "CHEMICAL FUNCTION", "CHEMICAL_FUNCTION") or "other"
            if not (pdb_id and chain and resid_str):
                continue
            pdb_id = str(pdb_id).lower()
            chain = str(chain).strip()
            try:
                resid_int = int(resid_str)
            except ValueError:
                digits = "".join(ch for ch in str(resid_str) if ch.isdigit())
                if not digits:
                    continue
                resid_int = int(digits)
            bucket[(pdb_id, chain, resid_int)].append(str(role_type).strip().lower())

    # 多数票 & 并列优先级
    priority = CONFIG["role_type_priority"]
    priority_map = {name: i for i, name in enumerate(priority)}
    decided: Dict[Tuple[str, str, int], str] = {}
    for key, roles in bucket.items():
        cnt = Counter(roles)
        topc = max(cnt.values())
        cands = [r for r, c in cnt.items() if c == topc]
        if len(cands) == 1:
            decided[key] = cands[0]
        else:
            decided[key] = sorted(cands, key=lambda x: priority_map.get(x, len(priority_map) + 1))[0]
    return decided


def build_label_vocab(annotations: Dict[Tuple[str, str, int], str]) -> List[str]:
    role_types = sorted(set(annotations.values()))
    return ["None"] + role_types


def model0_available_chains_with_ca(pdb_path: str) -> Dict[str, int]:
    """
    统计 model 0 中每条链“标准AA+含可用 CA”的残基数。
    """
    from Bio.PDB import PDBParser, is_aa
    parser = PDBParser(QUIET=True)
    model = next(parser.get_structure("x", pdb_path).get_models())
    counts = {}
    for ch in model.get_chains():
        n = 0
        for res in ch.get_residues():
            hetflag, resseq, icode = res.id
            if hetflag != " ":
                continue
            if not is_aa(res, standard=True):
                continue
            if "CA" not in res:
                continue
            try:
                _ = res["CA"].get_coord()
                n += 1
            except Exception:
                pass
        counts[str(ch.id)] = n
    return counts


def build_one_with_chain(builder: GNNProteinGraphBuilder, pdb_path: str, chain_id: Optional[str], name: Optional[str]):
    old = builder.cfg.chain_id
    builder.cfg.chain_id = chain_id
    try:
        return builder.build_one(pdb_path, name=name)
    finally:
        builder.cfg.chain_id = old


# ----------- 数据集：预过滤非法链 -----------
class ResidueRoleDataset(Dataset):
    """
    在 __init__ 阶段：
      - 为每个 PDB 决定唯一 chain_id（标注中多数链，且必须存在于该 PDB 的可用链集合）；
      - 如果标注链在该 PDB 中不存在或该链无可用 CA → 直接跳过该 PDB；
      - 保存 (pdb_path, pdb_id_lower, picked_chain) 到 self.items。

    __getitem__：
      - 使用 picked_chain 构图；
      - 解析 data.res_ids 获 resid_int；
      - 生成 y（未标注 → None=0）。
    """

    def __init__(
        self,
        pdb_paths: List[str],
        pdb_dir: str,
        annotations: Dict[Tuple[str, str, int], str],
        label_vocab: List[str],
        builder_cfg: Dict,
        cache_dir: Optional[str] = None,
    ):
        super().__init__()
        self.pdb_dir = pdb_dir
        self.ann = annotations
        self.label_vocab = label_vocab
        self.label_to_idx = {k: i for i, k in enumerate(label_vocab)}
        self.cache_dir = cache_dir

        # Builder（链在 __getitem__ 时临时传入）
        cfg = BuilderConfig(
            pdb_dir=pdb_dir,
            out_dir=cache_dir or pdb_dir,
            embedder=builder_cfg.get("embedder", []),
            radius=builder_cfg.get("radius", 10.0),
        )
        self.builder = GNNProteinGraphBuilder(
            cfg,
            edge_mode=builder_cfg.get("edge_mode", "rbf"),
            rbf_centers=builder_cfg.get("rbf_centers", None),
            rbf_gamma=builder_cfg.get("rbf_gamma", None),
            add_self_loop=builder_cfg.get("add_self_loop", False),
        )

        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)

        # --- 预过滤非法链，构建 self.items ---
        # 1) 建立 PDB → 标注链投票
        votes_by_pdb: Dict[str, Counter] = defaultdict(Counter)
        for (pid, ch, _res), _role in self.ann.items():
            votes_by_pdb[pid].update([ch])

        # 2) 遍历 pdb_paths，选择合法链，否则跳过
        self.items: List[Tuple[str, str, str]] = []  # (pdb_path, pdb_id_lower, picked_chain)
        skipped: List[str] = []

        for pdb_path in pdb_paths:
            pdb_id = os.path.splitext(os.path.basename(pdb_path))[0].lower()
            ann_chains = list(votes_by_pdb.get(pdb_id, {}).keys())
            try:
                available = model0_available_chains_with_ca(pdb_path)  # {chain: count}
            except Exception as e:
                skipped.append(f"{pdb_id}: parse_failed({e})")
                continue

            if not available:
                skipped.append(f"{pdb_id}: no_available_chain")
                continue

            picked = None
            if ann_chains:
                # 只接受“既在标注里、也在可用集合里”的链
                inter = [ch for ch in ann_chains if ch in available and available[ch] > 0]
                if inter:
                    # 按标注票数排序取第一
                    vote = votes_by_pdb[pdb_id]
                    inter.sort(key=lambda ch: (-vote[ch], -available[ch]))
                    picked = inter[0]

            # 没有匹配的标注链 → 跳过（题意希望“按标注里写的链构图”）
            if picked is None:
                skipped.append(f"{pdb_id}: ann_chains={ann_chains or '[]'} not in valid_chains={list(available.keys())}")
                continue

            self.items.append((pdb_path, pdb_id, picked))

        if skipped:
            print(f"[INFO] Prefilter skipped {len(skipped)} PDBs (illegal/missing chains). Example:")
            for s in skipped[:10]:
                print("  -", s)

        self._skipped_build: List[str] = []  # 极少数构图时失败的样本记录（理论上不会出现）

    def __len__(self) -> int:
        return len(self.items)

    def _cache_path(self, pdb_id: str) -> str:
        if not self.cache_dir:
            return ""
        return os.path.join(self.cache_dir, f"{pdb_id}.pt")

    def __getitem__(self, idx: int):
        pdb_path, pdb_id, chain_id = self.items[idx]
        cache_path = self._cache_path(pdb_id)
        if cache_path and os.path.isfile(cache_path):
            try:
                return torch.load(cache_path, map_location="cpu")
            except Exception:
                pass

        # 构图（指定链）
        try:
            data, misc = build_one_with_chain(self.builder, pdb_path, chain_id, name=pdb_id)
        except Exception as e:
            # 理论很少发生：已在可用链集合中
            self._skipped_build.append(f"{pdb_id}:{chain_id} build_failed({e})")
            raise

        # 解析 resid 序列（单链，chain 固定）
        N = data.num_nodes
        resid_seq: List[int] = []
        if hasattr(data, "res_ids") and isinstance(data.res_ids, list):
            for s in data.res_ids:
                # 格式类似 "A:123" 或 "A:123A" → 提取数字
                if ":" in s:
                    _c, r = s.split(":", 1)
                else:
                    r = s
                num = ""
                for ch in r:
                    if ch.isdigit():
                        num += ch
                    else:
                        break
                resid_seq.append(int(num) if num else -1)
        else:
            resid_seq = list(range(1, N + 1))

        # 构标签
        y = torch.zeros(N, dtype=torch.long)
        for i in range(N):
            key = (pdb_id, chain_id, int(resid_seq[i]))
            if key in self.ann:
                cls = self.label_to_idx.get(self.ann[key], self.label_to_idx.get("other", 0))
                y[i] = int(cls)
        data.y = y
        data._meta = {"pdb_id": pdb_id, "picked_chain": chain_id}

        if cache_path:
            torch.save(data, cache_path)
        return data


# ------------- 评估、训练 -------------
@torch.no_grad()
def evaluate(model: nn.Module, loader: GeoDataLoader, device: str, ignore_none_in_f1: bool = True) -> Dict[str, float]:
    from sklearn.metrics import f1_score

    model.eval()
    total, correct = 0, 0
    y_true_all: List[int] = []
    y_pred_all: List[int] = []

    for batch in loader:
        batch = batch.to(device)
        logits = model(batch)  # [N, C]
        pred = torch.argmax(logits, dim=-1)
        y = batch.y

        total += y.numel()
        correct += (pred == y).sum().item()

        y_true_all.extend(y.cpu().tolist())
        y_pred_all.extend(pred.cpu().tolist())

    acc = correct / max(1, total)

    labels = sorted(set(y_true_all))
    if ignore_none_in_f1 and (0 in labels):
        labels = [l for l in labels if l != 0]
        f1 = f1_score(y_true_all, y_pred_all, labels=labels, average="macro", zero_division=0) if labels else 0.0
    else:
        f1 = f1_score(y_true_all, y_pred_all, average="macro", zero_division=0)

    return {"acc": acc, "f1_macro": f1}


def compute_class_weights(loader: GeoDataLoader, num_classes: int, device: str) -> torch.Tensor:
    cnt = torch.zeros(num_classes, dtype=torch.float64)
    for batch in loader:
        vals, c = torch.unique(batch.y, return_counts=True)
        cnt[vals.long()] += c.double()
    cnt = torch.clamp(cnt, min=1.0)
    inv = 1.0 / cnt
    w = inv / inv.sum() * num_classes
    return w.float().to(device)


def split_train_val(items: List[str], val_ratio: float, seed: int) -> Tuple[List[str], List[str]]:
    rng = random.Random(seed)
    files = items[:]
    rng.shuffle(files)
    n_val = max(1, int(len(files) * val_ratio))
    return files[n_val:], files[:n_val]


def main():
    cfg = CONFIG
    set_seed(cfg["seed"])
    device = cfg["device"]

    # 读数据
    all_pdb_files = list_pdb_files(cfg["pdb_dir"])
    if not all_pdb_files:
        raise FileNotFoundError(f"未在目录中发现 PDB：{cfg['pdb_dir']}")

    ann = load_annotations(cfg["csv_path"])
    label_vocab = build_label_vocab(ann)
    print(f"[INFO] Label vocab: {label_vocab}")

    # 随机划分文件（注意：过滤发生在 Dataset 内部）
    train_files, val_files = split_train_val(all_pdb_files, cfg["val_ratio"], cfg["seed"])
    # 先用一次 DS 构建看过滤后的样本数（Dataset 内部会过滤非法链）
    tmp_ds = ResidueRoleDataset(train_files, cfg["pdb_dir"], ann, label_vocab, cfg["builder"], cache_dir=None)
    tmp_val_ds = ResidueRoleDataset(val_files, cfg["pdb_dir"], ann, label_vocab, cfg["builder"], cache_dir=None)
    print(f"[INFO] Train samples={len(tmp_ds)} | Val samples={len(tmp_val_ds)}")

    # 重新正式创建（需要缓存就开）
    train_ds = ResidueRoleDataset(train_files, cfg["pdb_dir"], ann, label_vocab, cfg["builder"], cache_dir=cfg["cache_dir"])
    val_ds = ResidueRoleDataset(val_files, cfg["pdb_dir"], ann, label_vocab, cfg["builder"], cache_dir=cfg["cache_dir"])

    train_loader = GeoDataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, num_workers=cfg["num_workers"])
    val_loader = GeoDataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False, num_workers=cfg["num_workers"])

    # 模型
    num_classes = len(label_vocab)
    model = ResidueRoleHead(
        backbone_cfg=cfg["backbone_cfg"],
        num_classes=num_classes,
        freeze_backbone=cfg["freeze_backbone"],
        classifier_dims=cfg["classifier_dims"],
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"])

    # 类权重
    class_weights = compute_class_weights(train_loader, num_classes, device)
    print(f"[INFO] class_weights={class_weights.detach().cpu().numpy().round(4).tolist()}")

    # 训练
    best_val_f1 = -1.0
    patience_left = cfg["early_stop_patience"]
    global_step = 0

    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        running_loss = 0.0

        for i, batch in enumerate(train_loader, start=1):
            batch = batch.to(device)
            logits = model(batch)
            loss = model.compute_loss(logits, batch.y, class_weights=class_weights)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if cfg["grad_clip"]:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            optimizer.step()

            running_loss += loss.item()
            global_step += 1

            if (i % cfg["log_every"]) == 0:
                print(f"[E{epoch:02d}][{i:04d}] loss={running_loss / cfg['log_every']:.4f}")
                running_loss = 0.0

        scheduler.step()

        # 验证
        metrics = evaluate(model, val_loader, device, ignore_none_in_f1=True)
        print(f"[VAL][E{epoch:02d}] acc={metrics['acc']:.4f} | f1_macro(no-None)={metrics['f1_macro']:.4f}")

        if metrics["f1_macro"] > best_val_f1:
            best_val_f1 = metrics["f1_macro"]
            patience_left = cfg["early_stop_patience"]
            os.makedirs("checkpoints", exist_ok=True)
            torch.save({
                "model": model.state_dict(),
                "cfg": cfg,
                "label_vocab": label_vocab,
            }, os.path.join("checkpoints", "residue_role_best.pt"))
        else:
            patience_left -= 1
            if patience_left <= 0:
                print("[INFO] Early stop.")
                break

    print(f"[DONE] Best Val F1 (no-None) = {best_val_f1:.4f}")


if __name__ == "__main__":
    main()