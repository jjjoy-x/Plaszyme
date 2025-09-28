# -*- coding: utf-8 -*-
"""
pdb_residue_reader.py
通用 PDB / mmCIF 读取与“残基级”标注模块（不执行，仅供导入调用）

功能概述
- 统一读取 .pdb / .cif / .mmcif / 以及 .gz 压缩文件
- 兼容多个 model、多个 chain
- 残基级遍历与标准化键值：(model_id, chain_id, resseq, icode)
- 过滤项：是否仅保留标准氨基酸、是否需要 CA 原子、是否包含 HETATM
- 字段抽取：三/一字母氨基酸名、是否标准氨基酸、是否 HETATM、是否缺 CA、
          CA 坐标、残基/原子数、平均 B 因子（AlphaFold/ColabFold 通常为 pLDDT）
- 工具函数：按链构建序列、肽键顺序边、空间近邻边（CA-CA 半径）
- 结果数据结构：List[ResidueRecord(dict-like)] 与若干索引工具

用法（示例，勿直接执行）：
    from pdb_residue_reader import PDBResidueReader, ReaderOptions

    options = ReaderOptions(
        models=None,                 # None=全部 model；或传如 [0]
        chains=None,                 # None=全部链；或传如 ['A','B']
        standard_aa_only=True,       # 仅标准氨基酸
        require_ca=True,             # 需要存在 CA 原子
        include_hetatm=False,        # 是否保留 HETATM 残基
        resolve_altloc='first',      # 可选: 'first' | 'A' | None(保留多构象)
        quiet=True,                  # 降低解析日志
        permissive=True              # 宽松解析
    )
    reader = PDBResidueReader(options)
    structure = reader.load("path/to/structure.pdb.gz")
    residues = reader.collect_residue_records(structure)

    # 构建按链的序列字典
    chain_seqs = reader.build_chain_sequences(residues)

    # 肽键顺序边（同链相邻残基）
    edges_seq = reader.build_sequential_edges(residues)  # List[ (node_key_i, node_key_j) ]

    # 空间近邻边（CA-CA <= cutoff）
    edges_spa = reader.build_spatial_edges(residues, cutoff=8.0)

    # 快速把 residues 转为 pandas.DataFrame（可选）
    # import pandas as pd
    # df = pd.DataFrame(residues)

作者：ChatGPT（通用模块，适用于后续图构建、残基级标注）
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, List, Dict, Tuple, Optional, Any, Set, Union

import gzip
import io
import numpy as np

from Bio.PDB import (
    PDBParser, MMCIFParser, Polypeptide, Selection
)
from Bio.PDB.Atom import Atom
from Bio.PDB.Residue import Residue
from Bio.PDB.Chain import Chain
from Bio.PDB.Structure import Structure
from Bio.PDB.Model import Model
from Bio.PDB.NeighborSearch import NeighborSearch


PathLike = Union[str, Path]
ResidueKey = Tuple[int, str, int, str]  # (model_id, chain_id, resseq, icode)


@dataclass
class ReaderOptions:
    models: Optional[List[int]] = None       # 指定保留的 model id 列表（None=全部）
    chains: Optional[List[str]] = None       # 指定保留的链 ID 列表（None=全部）
    standard_aa_only: bool = True            # 仅标准氨基酸（排除修饰/水/配体）
    require_ca: bool = True                  # 残基必须有 CA 原子
    include_hetatm: bool = False             # 是否包含 HETATM（默认不包含）
    resolve_altloc: Optional[str] = 'first'  # 多构象策略：'first'、'A'、None(保留)
    quiet: bool = True                       # Bio.PDB 解析静默
    permissive: bool = True                  # 宽松解析（容忍某些不规范记录）


class PDBResidueReader:
    """
    残基级读取器：面向“后续建图”的通用识别/抽取模块。
    """
    def __init__(self, options: ReaderOptions = ReaderOptions()):
        self.opt = options
        self._pdb_parser = PDBParser(PERMISSIVE=self.opt.permissive, QUIET=self.opt.quiet)
        self._cif_parser = MMCIFParser(QUIET=self.opt.quiet)

    # -----------------------------
    # 加载与基本遍历
    # -----------------------------
    def _open_text(self, path: Path) -> io.TextIOBase:
        if path.suffix.lower() == ".gz":
            return io.TextIOWrapper(gzip.open(path, "rb"))
        return open(path, "rt")

    def load(self, path: PathLike, struct_id: Optional[str] = None) -> Structure:
        path = Path(path)
        if struct_id is None:
            struct_id = path.stem.replace(".gz", "")

        suffix = path.suffix.lower()
        with self._open_text(path) as handle:
            if suffix in {".cif", ".mmcif"} or path.name.endswith((".cif.gz", ".mmcif.gz")):
                structure = self._cif_parser.get_structure(struct_id, handle)
            elif suffix == ".pdb" or path.name.endswith(".pdb.gz"):
                structure = self._pdb_parser.get_structure(struct_id, handle)
            else:
                # 尝试先按 mmCIF，再按 PDB
                try:
                    handle.seek(0)
                    structure = self._cif_parser.get_structure(struct_id, handle)
                except Exception:
                    handle.seek(0)
                    structure = self._pdb_parser.get_structure(struct_id, handle)
        return structure

    def _iter_target_models(self, structure: Structure) -> Iterator[Model]:
        for model in structure:
            if (self.opt.models is not None) and (model.id not in self.opt.models):
                continue
            yield model

    def _iter_target_chains(self, model: Model) -> Iterator[Chain]:
        for chain in model:
            if (self.opt.chains is not None) and (chain.id not in self.opt.chains):
                continue
            yield chain

    # -----------------------------
    # 残基与原子筛选
    # -----------------------------
    @staticmethod
    def _one_letter(resname3: str) -> str:
        """
        三字母转一字母；若非标准氨基酸则返回 'X'
        """
        res3 = (resname3 or "").upper().strip()
        try:
            return Polypeptide.three_to_one(res3)
        except KeyError:
            return 'X'

    @staticmethod
    def _is_standard_aa(residue: Residue) -> bool:
        return Polypeptide.is_aa(residue, standard=True)

    @staticmethod
    def _is_het_residue(residue: Residue) -> bool:
        # residue.id: (hetflag, resseq, icode)
        hetflag = residue.id[0]
        return (hetflag is not None) and (str(hetflag).strip() != '')

    @staticmethod
    def _first_or_alt(atom: Atom, prefer: Optional[str] = 'A') -> Atom:
        """
        返回单个代表构象的原子实例：
        - 如果 atom 非离散，直接返回；
        - 如果有 altloc，多构象：
            prefer='A' 取 A 构象，有则取 A，否则取列表第一个；
            prefer=None 则直接返回原子（保持离散），上游谨慎处理。
        """
        if not atom.is_disordered():
            return atom
        if prefer is None:
            return atom  # 保留“离散原子对象”，上游自行处理
        alts = atom.disordered_get_list()
        if prefer == 'first' or prefer is True:
            return alts[0]
        # 指定 altloc 代码
        for a in alts:
            if getattr(a, 'get_altloc', lambda: None)() == prefer:
                return a
        return alts[0]

    def _ca_atom(self, residue: Residue) -> Optional[Atom]:
        if 'CA' not in residue:
            return None
        atom = residue['CA']
        return self._first_or_alt(atom, prefer=self.opt.resolve_altloc)

    @staticmethod
    def _residue_center(residue: Residue) -> Optional[np.ndarray]:
        coords = []
        for a in residue.get_atoms():
            try:
                coords.append(np.asarray(a.get_coord(), dtype=float))
            except Exception:
                pass
        if not coords:
            return None
        return np.mean(coords, axis=0)

    @staticmethod
    def _mean_bfactor(residue: Residue) -> Optional[float]:
        vals = []
        for a in residue.get_atoms():
            try:
                vals.append(float(a.get_bfactor()))
            except Exception:
                pass
        return float(np.mean(vals)) if vals else None

    # -----------------------------
    # 主抽取：残基记录
    # -----------------------------
    def collect_residue_records(self, structure: Structure) -> List[Dict[str, Any]]:
        """
        输出：每个元素是一个 dict（ResidueRecord）
        键值（常用）：
            key              : (model_id, chain_id, resseq, icode)
            model_id         : int
            chain_id         : str
            resseq           : int
            icode            : str ('' 表示空)
            resname3         : str
            resname1         : str (非标准氨基酸→'X')
            is_standard_aa   : bool
            is_hetatm        : bool
            has_ca           : bool
            ca_coord         : np.ndarray shape=(3,) 或 None
            center_coord     : np.ndarray shape=(3,) 或 None
            atom_count       : int
            mean_bfactor     : float 或 None   # AlphaFold/ColabFold 常为 pLDDT
        过滤逻辑由 ReaderOptions 控制。
        """
        out: List[Dict[str, Any]] = []

        for model in self._iter_target_models(structure):
            mid = model.id
            for chain in self._iter_target_chains(model):
                cid = chain.id

                for residue in chain:
                    resname3 = residue.get_resname()
                    is_std = self._is_standard_aa(residue)
                    is_het = self._is_het_residue(residue)

                    # 过滤：标准氨基酸
                    if self.opt.standard_aa_only and not is_std:
                        continue

                    # 过滤：HETATM
                    if not self.opt.include_hetatm and is_het:
                        continue

                    # 过滤：需要 CA
                    ca = self._ca_atom(residue)
                    if self.opt.require_ca and (ca is None):
                        continue

                    resseq: int = residue.id[1]
                    icode: str = residue.id[2] if isinstance(residue.id[2], str) else ''
                    ca_coord = None if ca is None else np.asarray(ca.get_coord(), dtype=float)

                    rec: Dict[str, Any] = dict(
                        key=(mid, cid, resseq, icode),
                        model_id=mid,
                        chain_id=cid,
                        resseq=resseq,
                        icode=icode,
                        resname3=resname3,
                        resname1=self._one_letter(resname3),
                        is_standard_aa=is_std,
                        is_hetatm=is_het,
                        has_ca=(ca is not None),
                        ca_coord=ca_coord,
                        center_coord=self._residue_center(residue),
                        atom_count=len(list(residue.get_atoms())),
                        mean_bfactor=self._mean_bfactor(residue),
                    )
                    out.append(rec)

        # 稳定排序：按 model, chain, resseq, icode
        out.sort(key=lambda r: (r['model_id'], r['chain_id'], r['resseq'], r['icode']))
        return out

    # -----------------------------
    # 序列与索引工具
    # -----------------------------
    @staticmethod
    def group_by_chain(residues: List[Dict[str, Any]]) -> Dict[Tuple[int, str], List[Dict[str, Any]]]:
        buckets: Dict[Tuple[int, str], List[Dict[str, Any]]] = {}
        for r in residues:
            buckets.setdefault((r['model_id'], r['chain_id']), []).append(r)
        # 保持原有排序（已按 resseq 排过）
        return buckets

    def build_chain_sequences(self, residues: List[Dict[str, Any]]) -> Dict[Tuple[int, str], str]:
        """
        输出：{ (model_id, chain_id): 'MSEQ...' }
        非标准氨基酸会映射为 'X'（若开启 standard_aa_only 通常不会出现）
        """
        out: Dict[Tuple[int, str], str] = {}
        for key, group in self.group_by_chain(residues).items():
            seq = ''.join([r['resname1'] for r in group])
            out[key] = seq
        return out

    @staticmethod
    def build_key_index(residues: List[Dict[str, Any]]) -> Dict[ResidueKey, int]:
        """
        建立 node_key → 连续索引 的映射，用于后续建图。
        """
        index: Dict[ResidueKey, int] = {}
        for i, r in enumerate(residues):
            index[r['key']] = i
        return index

    # -----------------------------
    # 边构建：顺序边与空间边
    # -----------------------------
    def build_sequential_edges(self, residues: List[Dict[str, Any]]) -> List[Tuple[ResidueKey, ResidueKey]]:
        """
        同链“相邻残基”的顺序边（肽键近似）。返回 node_key 对列表（无向请自行双写）。
        """
        edges: List[Tuple[ResidueKey, ResidueKey]] = []
        for (mid, cid), group in self.group_by_chain(residues).items():
            for i in range(len(group) - 1):
                a = group[i]['key']
                b = group[i + 1]['key']
                edges.append((a, b))
        return edges

    def build_spatial_edges(
        self,
        residues: List[Dict[str, Any]],
        cutoff: float = 8.0,
        use_ca_only: bool = True
    ) -> List[Tuple[ResidueKey, ResidueKey]]:
        """
        CA-CA（或残基质心）半径图。返回 node_key 对列表（i<j，避免重复）。
        - cutoff: Å
        - use_ca_only=True 使用 CA 坐标；否则使用残基所有原子质心
        """
        # 准备坐标与键
        coords: List[np.ndarray] = []
        keys: List[ResidueKey] = []

        for r in residues:
            if use_ca_only and (r['ca_coord'] is not None):
                pt = r['ca_coord']
            else:
                pt = r['center_coord']
            if pt is None:
                continue
            keys.append(r['key'])
            coords.append(pt.astype(float))

        if not coords:
            return []

        # 用 Bio.PDB.NeighborSearch 在全原子层面查近邻较慢；这里直接做向量化半径图（N 可能不大）
        X = np.stack(coords, axis=0)  # (N, 3)
        # 距离上三角阈值
        diffs = X[:, None, :] - X[None, :, :]
        d2 = np.einsum('ijk,ijk->ij', diffs, diffs)  # (N,N)
        # 只取上三角 (i<j)，且距离<=cutoff
        cutoff2 = cutoff * cutoff
        n = X.shape[0]
        edges: List[Tuple[ResidueKey, ResidueKey]] = []
        for i in range(n):
            # j 从 i+1 起，避免重复
            mask = (d2[i, (i + 1):] <= cutoff2)
            if not np.any(mask):
                continue
            js = np.nonzero(mask)[0] + (i + 1)
            ai = keys[i]
            for j in js:
                edges.append((ai, keys[j]))
        return edges


# -----------------------------
# 可选：快速测试/示例（不要直接运行；仅作为说明）
# -----------------------------
if __name__ == "__main__":
    """
    说明：
    - 本模块按需求“仅生成代码，不自行运行”
    - 如需手工测试，请自行在外部脚本中导入并调用
    """
    pass