#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sequence_embedder.py

Protein residue-level embedding utilities with a unified interface.
统一的蛋白质残基级嵌入接口与实现（One-Hot / ESM）。

- BaseProteinEmbedder: 抽象基类，统一 __call__/embed 接口（单条或多条序列）
- OneHotEmbedder: 21 维（20 氨基酸 + <UNK>）独热向量
- ESMEmbedder: FAIR ESM-2 模型的残基嵌入，按需加载，支持半精度/批处理/截断

Notes / 注意:
- 所有嵌入器均返回“残基级”序列表示：单条输入 → [L, D]，多条输入 → List[Tensor([L_i, D])]
- 统一在 __call__ 中代理 embed，便于与其他模块解耦
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from tqdm import tqdm


# ---------------------------
# Base class / 抽象基类
# ---------------------------
class BaseProteinEmbedder:
    """Abstract base class for residue-level protein embedders.
    蛋白质残基级嵌入的抽象基类，统一接口，便于替换/扩展。

    Subclasses must implement :meth:`embed`.
    子类需实现 :meth:`embed`。

    Args:
        device (Optional[str]): Torch device spec (e.g., ``"cuda"``, ``"cpu"``, ``"cuda:0"``).
            如果为 ``None``，将自动检测并优先使用 CUDA。
    """

    def __init__(self, device: Optional[str] = None) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # ---- required API ----
    def embed(self, sequences: Union[str, Sequence[str]]) -> Union[torch.Tensor, List[torch.Tensor]]:
        """Compute residue-level embeddings for a sequence or a list of sequences.
        为单条或多条蛋白序列计算残基级嵌入。

        Args:
            sequences (Union[str, Sequence[str]]): A single amino-acid string or a list of strings.
                单条氨基酸序列，或由多条序列组成的列表。

        Returns:
            Union[torch.Tensor, List[torch.Tensor]]:
                - If input is a single string: ``Tensor[L, D]``.
                - If input is a list: ``List[Tensor[L_i, D]]`` (no padding).
                单条输入返回 ``[L, D]``；多条输入返回不对齐的列表（不做 padding）。

        Raises:
            NotImplementedError: If the subclass did not implement this method.
        """
        raise NotImplementedError

    # ---- convenience ----
    def __call__(self, sequences: Union[str, Sequence[str]]) -> Union[torch.Tensor, List[torch.Tensor]]:
        """Alias for :meth:`embed` to support call style.
        便捷调用，等价于 :meth:`embed`。
        """
        return self.embed(sequences)

    # ---- misc ----
    @property
    def name(self) -> str:
        """Human-friendly embedder name. 人类可读名称。"""
        return self.__class__.__name__

    def info_str(self) -> str:
        """Return a short runtime info string for logging.
        返回聚合的运行时信息字符串，用于日志打印。
        """
        return f"[{self.name}] device={self.device}"


# ---------------------------
# Utility / 工具函数
# ---------------------------
_AA20 = "ACDEFGHIKLMNPQRSTVWY"
_AA_TO_IDX = {a: i for i, a in enumerate(_AA20)}
_UNK_IDX = len(_AA20)  # index 20 for unknown


def _truncate(seq: str, max_len: Optional[int], strategy: str = "center") -> str:
    """Truncate a sequence according to a given strategy.
    根据策略截断序列。

    Args:
        seq (str): Amino-acid sequence. 氨基酸序列
        max_len (Optional[int]): If ``None``, no truncation. 若为 ``None`` 不截断
        strategy (str): ``"center"``, ``"left"``, or ``"right"``.

    Returns:
        str: Possibly truncated sequence. 截断后的序列

    Raises:
        ValueError: If an unknown strategy is given.
    """
    if max_len is None or len(seq) <= max_len:
        return seq
    if strategy == "center":
        half = max_len // 2
        return seq[:half] + seq[-(max_len - half):]
    if strategy == "left":
        return seq[-max_len:]
    if strategy == "right":
        return seq[:max_len]
    raise ValueError(f"Unknown truncate strategy: {strategy}")


# ---------------------------
# One-hot embedder / 独热嵌入
# ---------------------------
class OneHotEmbedder(BaseProteinEmbedder):
    """Residue one-hot encoding (20 AA + <UNK> = 21 dims).
    氨基酸独热编码（20 种氨基酸 + 未知 = 21 维）。

    - Case-insensitive; non-standard letters map to <UNK>.
      不区分大小写；非标准字母映射为 <UNK>。

    Args:
        device (Optional[str]): Torch device spec. See :class:`BaseProteinEmbedder`.
        max_len (Optional[int]): If set, truncate sequences longer than this.
            序列最大长度，超出部分将被截断。
        truncate_from (str): Truncation strategy: ``"center"``, ``"left"``, or ``"right"``.
            截断策略：居中/左/右。
    """

    def __init__(
        self,
        device: Optional[str] = None,
        *,
        max_len: Optional[int] = None,
        truncate_from: str = "center",
    ) -> None:
        super().__init__(device=device)
        self.max_len = max_len
        self.truncate_from = truncate_from
        print(self.info_str() + f" | dims=21 (20AA+UNK) | max_len={self.max_len} | trunc={self.truncate_from}")

    @staticmethod
    def _one_hot(seq: str) -> torch.Tensor:
        L = len(seq)
        out = torch.zeros(L, 21, dtype=torch.float32)
        for i, ch in enumerate(seq.upper()):
            out[i, _AA_TO_IDX.get(ch, _UNK_IDX)] = 1.0
        return out

    def embed(self, sequences: Union[str, Sequence[str]]) -> Union[torch.Tensor, List[torch.Tensor]]:
        if isinstance(sequences, str):
            seq = _truncate(sequences, self.max_len, self.truncate_from)
            return self._one_hot(seq)

        outs: List[torch.Tensor] = []
        for seq in sequences:
            seq = _truncate(seq, self.max_len, self.truncate_from)
            outs.append(self._one_hot(seq))
        return outs


# ---------------------------
# ESM embedder / ESM 嵌入
# ---------------------------
class ESMEmbedder(BaseProteinEmbedder):
    """Residue embeddings via FAIR ESM models (lazy-loaded).
    使用 FAIR ESM 模型生成残基级嵌入（按需加载模型）。

    Provides residue-wise embeddings from ESM-2 models with optional FP16,
    truncation and mini-batch list processing.

    Args:
        model_name (str): ESM model ID (e.g. ``"esm2_t12_35M_UR50D"``).
            ESM 模型标识。
        device (Optional[str]): Torch device spec. See :class:`BaseProteinEmbedder`.
            设备选择（缺省自动，CUDA 优先）。
        repr_layer (Optional[int]): Which hidden layer to extract. If ``None``,
            uses the final layer. 选择抽取的隐藏层（不指定则用最后一层）。
        fp16 (bool): If ``True``, run model & tokens in half precision (GPU only).
            是否启用半精度（仅 GPU 有效，可省显存）。
        max_len (Optional[int]): If set, truncate sequences longer than this.
            序列最大长度，超出部分将被截断。
        truncate_from (str): Truncation strategy: ``"center"``, ``"left"``, or ``"right"``.
            截断策略。
        batch_size (int): Per-call mini-batch size for list embedding.
            多序列推理时的每批大小。
        verbose (bool): If ``True``, print model loading / runtime info.
            是否打印加载/运行信息。
    """

    _DEFAULT_MODEL = "esm2_t33_650M_UR50D"

    def __init__(
        self,
        model_name: str = _DEFAULT_MODEL,
        *,
        device: Optional[str] = None,
        repr_layer: Optional[int] = None,
        fp16: bool = False,
        max_len: Optional[int] = None,
        truncate_from: str = "center",
        batch_size: int = 8,
        verbose: bool = True,
    ) -> None:
        super().__init__(device=device)
        self.model_name = model_name
        self.repr_layer = repr_layer  # None → use last
        self.fp16 = bool(fp16) and torch.cuda.is_available()
        self.max_len = max_len
        self.truncate_from = truncate_from
        self.batch_size = max(1, int(batch_size))
        self.verbose = verbose

        self._model = None
        self._alphabet = None
        self._batch_converter = None
        self._last_dim = None  # filled after first forward

        if self.verbose:
            print(self.info_str() + f" | model={self.model_name} | fp16={self.fp16} | max_len={self.max_len} | trunc={self.truncate_from}")

    # ---- lazy loader ----
    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            import esm  # lazy import
        except Exception as e:
            raise RuntimeError(
                "Failed to import `esm`. Please `pip install fair-esm`.\n"
                "无法导入 `esm`，请先安装 fair-esm。"
            ) from e

        model, alphabet = esm.pretrained.load_model_and_alphabet(self.model_name)
        model.eval().to(self.device)
        if self.fp16:
            model.half()

        self._model = model
        self._alphabet = alphabet
        self._batch_converter = alphabet.get_batch_converter()

        if self.verbose:
            n_params = sum(p.numel() for p in model.parameters()) / 1e6
            print(f"[ESM] loaded {self.model_name} ({n_params:.1f}M params) on {self.device}")

    # ---- core API ----
    @torch.inference_mode()
    def embed(self, sequences: Union[str, Sequence[str]]) -> Union[torch.Tensor, List[torch.Tensor]]:
        self._ensure_loaded()

        # normalize inputs
        if isinstance(sequences, str):
            seqs = [_truncate(sequences, self.max_len, self.truncate_from)]
            single = True
        else:
            seqs = [_truncate(s, self.max_len, self.truncate_from) for s in sequences]
            single = False

        outs: List[torch.Tensor] = []
        iterator = range(0, len(seqs), self.batch_size)

        total = len(seqs)
        if self.verbose:
            if total == 1:
                # 单条也保持同一风格（可选）
                print(f"[ESM] total_seqs=1")
            elif total > 1:
                print(f"[ESM] total_seqs={total}")

        processed = 0
        for i in iterator:
            chunk = seqs[i: i + self.batch_size]
            data = [(f"seq{i + j}", s) for j, s in enumerate(chunk)]
            labels, toks = self._tokens_from_data(data)
            toks = toks.to(self.device)
            if self.fp16:
                toks = toks.half()

            # forward
            if self.repr_layer is None:
                repr_layers = [self._model.num_layers]
                layer_id = self._model.num_layers
            else:
                repr_layers = [int(self.repr_layer)]
                layer_id = int(self.repr_layer)

            result = self._model(toks, repr_layers=repr_layers, return_contacts=False)
            reps = result["representations"][layer_id]  # [B, T, D]

            for k, (_, seq) in enumerate(data):
                L = len(seq)
                emb = reps[k, 1: L + 1]  # 去掉CLS/EOS → [L, D]
                if self._last_dim is None:
                    self._last_dim = emb.shape[-1]

                outs.append(emb.detach().float().cpu())
                processed += 1

                # 每 10 条或最后一条打印一次（英文 + 你要的字段）
                if self.verbose and (processed % 10 == 0 or processed == total):
                    dim_show = self._last_dim if self._last_dim is not None else emb.shape[-1]
                    print(f"[ESM] seq {processed}/{total} | len={L} | dim={dim_show}")

        # 末尾补充一行总维度/总数量（与之前风格一致）
        if self.verbose and self._last_dim is not None:
            print(f"[ESM] repr_dim={self._last_dim}, n_seq={total}")

        if single:
            return outs[0]
        return outs

    # ---- helpers ----
    def _tokens_from_data(self, data: List[Tuple[str, str]]) -> Tuple[List[str], torch.Tensor]:
        """Convert labeled sequences into tokens via ESM alphabet.
        经 ESM 字母表将带标签的序列转为 token。

        Args:
            data (List[Tuple[str, str]]): List of (label, sequence). 标签与序列

        Returns:
            Tuple[List[str], torch.Tensor]: Labels and token tensor.
        """
        labels, _, tokens = self._batch_converter(data)
        return labels, tokens

# =========================
# PhysChemEmbedder
# =========================

class PhysChemEmbedder(BaseProteinEmbedder):
    """Residue-level physicochemical descriptors (12 dims) with optional 20-AA one-hot concat.
    使用氨基酸理化性质（12维）做残基级表示，可选拼接 20AA 独热编码。

    The 12 per-residue dimensions are:
      1) hydrophobicity (Kyte–Doolittle)
      2) volume (Chothia, Å^3)
      3) polarity (Grantham)
      4) is_polar (0/1)
      5) is_aromatic (0/1)
      6) is_aliphatic (0/1)
      7) is_charged (0/1)
      8) nominal_charge_at_pH7 (-1/0/+1; His≈+0.1)
      9) hbond_donor_count
     10) hbond_acceptor_count
     11) sidechain_pKa (0 if none)
     12) is_sulfur (0/1)

    Args:
        device (Optional[str]): Torch device spec. See :class:`BaseProteinEmbedder`.
        norm (str): {"none","zscore","minmax"} feature normalization strategy.
            - "none": raw values.
            - "zscore": (x-mean)/std per-dimension (computed over AA20 table).
            - "minmax": (x-min)/(max-min) per-dimension.
        concat_onehot (bool): If True, concatenate 20-AA one-hot (dim += 20).
        allow_unk (bool): If True, unknown residues map to zero vector (kept).
        max_len (Optional[int]): If set, truncate sequences longer than this.
        truncate_from (str): Truncation strategy: "center", "left", or "right".

    中文说明：
        - 维度：12 或 12+20（若 concat_onehot=True）；
        - 未知残基若 allow_unk=True，则用全零理化向量（且 one-hot 全零）；
        - 与 OneHotEmbedder 一样，embed() 支持 str 或 List[str] 并返回 [L,D] 或 List[[L,D]]。
    """

    _AA20 = "ACDEFGHIKLMNPQRSTVWY"  # 20 canonical AAs

    # --- property tables (dicts) ---
    _HYDRO = {  # Kyte–Doolittle hydrophobicity
        'A': 1.8, 'C': 2.5, 'D': -3.5, 'E': -3.5, 'F': 2.8,
        'G': -0.4, 'H': -3.2, 'I': 4.5, 'K': -3.9, 'L': 3.8,
        'M': 1.9, 'N': -3.5, 'P': -1.6, 'Q': -3.5, 'R': -4.5,
        'S': -0.8, 'T': -0.7, 'V': 4.2, 'W': -0.9, 'Y': -1.3
    }
    _VOLUME = {  # Chothia side-chain volume (Å^3), approximate
        'A': 88.6, 'C': 108.5, 'D': 111.1, 'E': 138.4, 'F': 189.9,
        'G': 60.1, 'H': 153.2, 'I': 166.7, 'K': 168.6, 'L': 166.7,
        'M': 162.9, 'N': 114.1, 'P': 112.7, 'Q': 143.8, 'R': 173.4,
        'S': 89.0, 'T': 116.1, 'V': 140.0, 'W': 227.8, 'Y': 193.6
    }
    _POLARITY = {  # Grantham polarity (a.u.)
        'A': 8.1, 'C': 5.5, 'D': 13.0, 'E': 12.3, 'F': 5.2,
        'G': 9.0, 'H': 10.4, 'I': 5.2, 'K': 11.3, 'L': 4.9,
        'M': 5.7, 'N': 11.6, 'P': 8.0, 'Q': 10.5, 'R': 10.5,
        'S': 9.2, 'T': 8.6, 'V': 5.9, 'W': 5.4, 'Y': 6.2
    }
    _AROMATIC = set("FWYH")
    _ALIPHATIC = set("ILV")
    _POLAR = set("STNQCY") | set("DEKRH")
    _CHARGED = set("DEKRH")
    _CHARGE = {  # nominal charge at pH~7
        'D': -1.0, 'E': -1.0, 'K': +1.0, 'R': +1.0, 'H': +0.1
    }
    _HBD = {  # H-bond donors
        'R': 1, 'K': 1, 'H': 1, 'N': 1, 'Q': 1, 'S': 1, 'T': 1, 'Y': 1, 'W': 1
    }
    _HBA = {  # H-bond acceptors
        'D': 2, 'E': 2, 'N': 1, 'Q': 1, 'S': 1, 'T': 1, 'Y': 1, 'H': 0, 'C': 1
    }
    _PKA = {  # sidechain pKa (0 if none)
        'D': 3.9, 'E': 4.2, 'H': 6.0, 'C': 8.3, 'Y': 10.1, 'K': 10.5, 'R': 12.5
    }
    _SULFUR = set("CM")

    def __init__(
        self,
        device: Optional[str] = None,
        *,
        norm: str = "none",
        concat_onehot: bool = False,
        allow_unk: bool = True,
        max_len: Optional[int] = None,
        truncate_from: str = "center",
    ) -> None:
        # 基类只接收 device；max_len / truncate_from 与 OneHotEmbedder 同风格在子类里存
        super().__init__(device=device)
        assert norm in {"none", "zscore", "minmax"}
        assert truncate_from in {"center", "left", "right"}
        self.norm = norm
        self.concat_onehot = bool(concat_onehot)
        self.allow_unk = bool(allow_unk)
        self.max_len = max_len
        self.truncate_from = truncate_from

        # 预构建 AA20 的 12维表 & 归一化统计（仅基于 AA20）
        self._aa2vec = self._build_table()                    # [20, 12]
        self._mean = self._aa2vec.mean(axis=0)
        self._std  = self._aa2vec.std(axis=0) + 1e-8
        self._min  = self._aa2vec.min(axis=0)
        self._max  = self._aa2vec.max(axis=0)

        self._dim = 12 + (20 if self.concat_onehot else 0)
        print(self.info_str() + f" | dim={self._dim} (12 physchem"
              + (f"+20 onehot" if self.concat_onehot else "")
              + f") | norm={self.norm} | max_len={self.max_len} | trunc={self.truncate_from}")

    # ---------- Base-style API ----------
    @property
    def dim(self) -> int:
        """Per-residue embedding dimension."""
        return self._dim

    def embed(self, sequences: Union[str, Sequence[str]]) -> Union[torch.Tensor, List[torch.Tensor]]:
        """Embed one or a list of protein sequences.

        Args:
            sequences (str | Sequence[str]): Single sequence or list of sequences.

        Returns:
            torch.Tensor | List[torch.Tensor]:
                - single: [L, D]
                - list: List of [L, D]
        """
        verbose = bool(getattr(self, "verbose", True))  # 不修改类结构，尽量最小侵入

        if isinstance(sequences, str):
            seq = _truncate(sequences, self.max_len, self.truncate_from)
            emb = self._embed_one(seq)  # [L, D] on CPU
            if verbose:
                print(f"[PhysChem] total_seqs=1")
                print(f"[PhysChem] seq 1/1 | len={emb.shape[0]} | dim={emb.shape[1]}")
            return emb.to(self.device)

        # list case
        total = len(sequences)
        if verbose:
            print(f"[PhysChem] total_seqs={total}")

        outs: List[torch.Tensor] = []
        for idx, seq in enumerate(sequences, start=1):
            seq = _truncate(seq, self.max_len, self.truncate_from)
            emb = self._embed_one(seq)  # [L, D] on CPU
            outs.append(emb.to(self.device))

            if verbose and (idx % 10 == 0 or idx == total):
                print(f"[PhysChem] seq {idx}/{total} | len={emb.shape[0]} | dim={emb.shape[1]}")

        return outs

    # ---------- helpers ----------
    def _build_table(self) -> np.ndarray:
        """Build the [20,12] matrix in fixed AA20 order."""
        rows = [self._aa_row(aa) for aa in self._AA20]
        return np.stack(rows, axis=0).astype(np.float32)

    def _aa_row(self, aa: str) -> np.ndarray:
        """Return 12-dim vector for a single canonical residue."""
        hydro = self._HYDRO[aa]
        vol = self._VOLUME[aa]
        pol = self._POLARITY[aa]
        is_polar = 1.0 if aa in self._POLAR else 0.0
        is_aromatic = 1.0 if aa in self._AROMATIC else 0.0
        is_aliphatic = 1.0 if aa in self._ALIPHATIC else 0.0
        is_charged = 1.0 if aa in self._CHARGED else 0.0
        charge = self._CHARGE.get(aa, 0.0)
        hbd = float(self._HBD.get(aa, 0))
        hba = float(self._HBA.get(aa, 0))
        pka = float(self._PKA.get(aa, 0.0))
        is_sulfur = 1.0 if aa in self._SULFUR else 0.0
        return np.array(
            [hydro, vol, pol, is_polar, is_aromatic, is_aliphatic,
             is_charged, charge, hbd, hba, pka, is_sulfur],
            dtype=np.float32
        )

    def _embed_one(self, seq: str) -> torch.Tensor:
        """Embed a single sequence to [L, D] tensor (on CPU; caller moves to device)."""
        seq = "".join(str(seq).split()).upper()
        L = len(seq)

        # 理化表：未知残基 -> 全零向量（若 allow_unk=False 可改成映射到某个AA）
        mat = np.zeros((L, 12), dtype=np.float32)
        for i, aa in enumerate(seq):
            if aa in self._AA20:
                mat[i] = self._aa_row(aa)
            else:
                if not self.allow_unk:
                    mat[i] = self._aa_row('A')  # 或者定义 UNK policy
                # else: 全零

        # 归一化
        if self.norm == "zscore":
            mat = (mat - self._mean) / self._std
        elif self.norm == "minmax":
            mat = (mat - self._min) / (self._max - self._min + 1e-8)

        # 可选拼接 one-hot(20)
        if self.concat_onehot:
            oh = np.zeros((L, 20), dtype=np.float32)
            for i, aa in enumerate(seq):
                if aa in self._AA20:
                    oh[i, self._AA20.index(aa)] = 1.0
            mat = np.concatenate([mat, oh], axis=1)

        return torch.from_numpy(mat)

# ---------------------------
# Simple test / 自检
# ---------------------------
if __name__ == "__main__":
    seq = ["MKTFFVIVAVLCLLSVAAQQEALAKEH","FVIVAVLCLLSVAAQQEAL"]
    print(ESMEmbedder().embed(seq))