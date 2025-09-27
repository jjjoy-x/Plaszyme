# -*- coding: utf-8 -*-
"""
pdb_reader.py

一个面向下游建模/特征工程的 PDB/mmCIF 读取类（基于 Biopython）。
目标：简单、统一、可查询。支持：
- 读取 PDB / mmCIF（自动按扩展名选择解析器），选择指定 model 与链；
- 残基级信息：三字母/一字母氨基酸、(chain, resseq, icode) 唯一键；
- 原子级信息：任意原子（如 CA、N、C、O、CB）坐标与 B 因子；处理 altloc（优先占据度或 'A'）；
- 常用“残基中心”坐标（CA / 主链均值 / 重原子均值 / 侧链均值）及有效掩码；
- 链内序列、一字母数组、残基键列表；可用 (chain, resseq, icode) 精确索引残基。

依赖：
    pip install biopython torch

注意：
- 默认去水（HETATM = 'W'），是否保留其它 HETATM 可通过 include_hetatm 控制；
- 非标准氨基酸一字母记为 'X'；
- 本文件仅定义类，不做任何执行。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union, Literal

import os
import torch
from Bio.PDB import PDBParser, MMCIFParser, Polypeptide
from Bio.PDB.Residue import Residue as BioResidue
from Bio.PDB.Atom import Atom as BioAtom
from Bio.PDB.Structure import Structure as BioStructure
from Bio.PDB.Model import Model as BioModel


# =========================
# 数据结构
# =========================

@dataclass(frozen=True)
class ResidueKey:
    """唯一定位一个残基：链 + 残基序号 + 插入码（icode，空时用空格' '）"""
    chain_id: str
    resseq: int
    icode: str = " "

    @staticmethod
    def from_bio(chain_id: str, res: BioResidue) -> "ResidueKey":
        hetflag, resseq, icode = res.get_id()
        return ResidueKey(chain_id, int(resseq), str(icode or " "))

    def as_tuple(self) -> Tuple[str, int, str]:
        return (self.chain_id, self.resseq, self.icode)

    def __str__(self) -> str:
        ic = "" if self.icode.strip() == "" else self.icode
        return f"{self.chain_id}:{self.resseq}{ic}"


@dataclass
class ResidueInfo:
    """对下游友好的残基信息记录"""
    key: ResidueKey                 # 唯一键 (chain, resseq, icode)
    aa_three: str                   # 三字母氨基酸名（如 'LYS'）
    aa_one: str                     # 一字母（非法/非标记为 'X'）
    is_hetatm: bool                 # 是否 HETATM（配体/离子/修饰等）
    altloc_selected: Optional[str]  # 若存在 altloc，选择了哪条（一般 'A' 或占据度最高）
    atoms: Dict[str, torch.Tensor]  # 原子名 → 坐标 [3]，float32（Å）
    b_factors: Dict[str, float]     # 原子名 → B 因子


# =========================
# 工具
# =========================

def _is_water(res: BioResidue) -> bool:
    hetflag, _, _ = res.get_id()
    return hetflag == "W"

def _three_to_one_safe(resname: str) -> str:
    try:
        return Polypeptide.three_to_one(resname)
    except Exception:
        return "X"

def _select_best_altloc(atoms: List[BioAtom]) -> Tuple[Optional[str], List[BioAtom]]:
    """为存在 altloc 的原子选择一个分支（优先 'A'，否则占据度总和最高）。"""
    groups: Dict[str, List[BioAtom]] = {}
    has_alt = False
    for a in atoms:
        alt = a.get_altloc()
        if alt and alt != " ":
            has_alt = True
            groups.setdefault(alt, []).append(a)
    if not has_alt:
        return None, atoms

    # 选择策略：优先 'A'，否则按占据度和
    if "A" in groups:
        chosen = "A"
    else:
        def occ_sum(v: List[BioAtom]) -> float:
            s = 0.0
            for x in v:
                o = x.get_occupancy()
                s += float(o if o is not None else 1.0)
            return s
        chosen = max(groups.keys(), key=lambda k: occ_sum(groups[k]))

    filtered: List[BioAtom] = []
    for a in atoms:
        alt = a.get_altloc()
        if (alt and alt != " " and alt == chosen) or (not alt or alt == " "):
            filtered.append(a)
    return chosen, filtered

def _residue_to_info(chain_id: str, res: BioResidue, include_hetatm: bool) -> Optional[ResidueInfo]:
    hetflag, _, _ = res.get_id()
    is_het = (hetflag != " ")
    if _is_water(res):
        return None
    if is_het and not include_hetatm:
        return None

    atoms = list(res.get_atoms())
    alt_selected, atoms = _select_best_altloc(atoms)

    coords: Dict[str, torch.Tensor] = {}
    bfac: Dict[str, float] = {}
    for a in atoms:
        pos = a.get_coord()
        if pos is None:
            continue
        name = a.get_name().upper()
        coords[name] = torch.tensor(pos, dtype=torch.float32)  # [3]
        b = a.get_bfactor()
        bfac[name] = float(0.0 if b is None else b)

    if len(coords) == 0:
        return None

    info = ResidueInfo(
        key=ResidueKey.from_bio(chain_id, res),
        aa_three=res.get_resname(),
        aa_one=_three_to_one_safe(res.get_resname()),
        is_hetatm=is_het,
        altloc_selected=alt_selected,
        atoms=coords,
        b_factors=bfac,
    )
    return info

def _center_from_atoms(info: ResidueInfo, mode: Literal["CA", "backbone", "heavy_mean", "sidechain"]) -> Optional[torch.Tensor]:
    atoms = info.atoms
    if mode == "CA":
        return atoms.get("CA", None)

    if mode == "backbone":
        names = ["N", "CA", "C", "O"]
        pts = [atoms[n] for n in names if n in atoms]
        if not pts:
            return None
        return torch.stack(pts, dim=0).mean(dim=0)

    if mode == "heavy_mean":
        pts = [v for k, v in atoms.items() if not k.upper().startswith("H")]
        if not pts:
            return None
        return torch.stack(pts, dim=0).mean(dim=0)

    if mode == "sidechain":
        pts = [v for k, v in atoms.items() if k not in {"N", "CA", "C", "O"} and not k.upper().startswith("H")]
        if not pts:
            return None
        return torch.stack(pts, dim=0).mean(dim=0)

    raise ValueError(f"Unknown center mode: {mode}")


# =========================
# 主类
# =========================

class PDBReader:
    """
    读取并提供按链/残基/原子级查询的统一接口。

    基本用法（仅示例，不执行）：
        rdr = PDBReader.from_file("1abc.pdb", model_index=0, include_hetatm=False)
        print(rdr.chain_ids)              # ['A','B',...]
        seq, keys = rdr.sequence("A")     # 一字母序列 & 位置→ResidueKey
        ca, mask = rdr.coords("A", center="CA")  # [L,3], [L]
        cb, m2   = rdr.atom_coords("A", atom="CB")
        info = rdr.get_residue(("A", 123, " "))  # 或 ResidueKey("A",123," ")
    """

    # ---------- 构造 ----------
    def __init__(self, *, struct_id: str, path: str, model: BioModel, include_hetatm: bool = False) -> None:
        self.struct_id = struct_id
        self.path = path
        self.include_hetatm = bool(include_hetatm)
        self._chains: Dict[str, List[ResidueInfo]] = {}
        self._index: Dict[str, Dict[Tuple[int, str], int]] = {}

        # 解析所有链
        for chain in model.get_chains():
            cid = chain.id
            rows: List[ResidueInfo] = []
            for res in chain.get_residues():
                info = _residue_to_info(cid, res, include_hetatm=self.include_hetatm)
                if info is not None:
                    rows.append(info)
            if rows:
                self._chains[cid] = rows
                # 建索引
                idx_map: Dict[Tuple[int, str], int] = {}
                for i, r in enumerate(rows):
                    idx_map[(r.key.resseq, r.key.icode)] = i
                self._index[cid] = idx_map

    @staticmethod
    def from_file(
        file_path: str,
        *,
        model_index: int = 0,
        chains: Optional[Sequence[str]] = None,
        include_hetatm: bool = False,
        permissive: bool = True,
        QUIET: bool = True,
    ) -> "PDBReader":
        """根据扩展名选择解析器，取第 model_index 个模型，支持只读取部分链。"""
        assert os.path.isfile(file_path), f"file not found: {file_path}"
        ext = os.path.splitext(file_path)[1].lower()
        if ext in (".cif", ".mmcif", ".mcif"):
            parser = MMCIFParser(QUIET=QUIET)
        else:
            parser = PDBParser(PERMISSIVE=permissive, QUIET=QUIET)

        struct_id = os.path.basename(file_path)
        structure: BioStructure = parser.get_structure(struct_id, file_path)

        models = list(structure.get_models())
        if not models:
            raise RuntimeError("No model found in structure.")
        if not (0 <= model_index < len(models)):
            raise IndexError(f"model_index out of range: {model_index} (n_models={len(models)})")
        model = models[model_index]

        rdr = PDBReader(struct_id=struct_id, path=file_path, model=model, include_hetatm=include_hetatm)

        if chains is not None:
            keep = set(chains)
            rdr._chains = {cid: lst for cid, lst in rdr._chains.items() if cid in keep}
            rdr._index = {cid: rdr._index[cid] for cid in rdr._chains.keys()}

        return rdr

    # ---------- 基本属性 ----------
    @property
    def chain_ids(self) -> List[str]:
        return list(self._chains.keys())

    def has_chain(self, chain_id: str) -> bool:
        return chain_id in self._chains

    def __len__(self) -> int:
        """结构中残基总数（所有链求和）"""
        return sum(len(v) for v in self._chains.values())

    # ---------- 链与残基查询 ----------
    def residues(self, chain_id: str) -> List[ResidueInfo]:
        if chain_id not in self._chains:
            raise KeyError(f"chain not found: {chain_id}")
        return self._chains[chain_id]

    def residue_keys(self, chain_id: str) -> List[ResidueKey]:
        return [r.key for r in self.residues(chain_id)]

    def aa_one(self, chain_id: str) -> List[str]:
        return [r.aa_one for r in self.residues(chain_id)]

    def aa_three(self, chain_id: str) -> List[str]:
        return [r.aa_three for r in self.residues(chain_id)]

    def get_residue(self, key: Union[ResidueKey, Tuple[str, int, str]]) -> ResidueInfo:
        if not isinstance(key, ResidueKey):
            key = ResidueKey(*key)
        cid = key.chain_id
        if cid not in self._chains:
            raise KeyError(f"chain not found: {cid}")
        idx = self._index[cid].get((key.resseq, key.icode))
        if idx is None:
            raise KeyError(f"residue not found: {key}")
        return self._chains[cid][idx]

    def get_index(self, key: Union[ResidueKey, Tuple[str, int, str]]) -> int:
        if not isinstance(key, ResidueKey):
            key = ResidueKey(*key)
        cid = key.chain_id
        if cid not in self._index:
            raise KeyError(f"chain not found: {cid}")
        idx = self._index[cid].get((key.resseq, key.icode))
        if idx is None:
            raise KeyError(f"residue not found: {key}")
        return idx

    # ---------- 序列 ----------
    def sequence(self, chain_id: str) -> Tuple[str, List[ResidueKey]]:
        """返回链的一字母序列与位置→ResidueKey 映射（长度 L 对齐）。"""
        recs = self.residues(chain_id)
        seq = "".join([r.aa_one for r in recs])
        return seq, [r.key for r in recs]

    # ---------- 坐标（残基中心 / 任意原子） ----------
    def coords(
        self,
        chain_id: str,
        *,
        center: Literal["CA", "backbone", "heavy_mean", "sidechain"] = "CA",
        device: Optional[Union[str, torch.device]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        返回残基中心坐标张量与有效掩码：
            coords: [L, 3] float32（Å）
            mask  : [L] bool
        center:
            - "CA"         : 取 Cα（缺失则该位 mask=False）
            - "backbone"   : N/CA/C/O 的均值（至少存在一个）
            - "heavy_mean" : 所有非氢原子均值
            - "sidechain"  : 侧链原子均值（无侧链则 False）
        """
        if device is None:
            device = "cpu"
        recs = self.residues(chain_id)
        L = len(recs)
        xyz = torch.zeros(L, 3, dtype=torch.float32, device=device)
        mask = torch.zeros(L, dtype=torch.bool, device=device)
        for i, r in enumerate(recs):
            c = _center_from_atoms(r, mode=center)
            if c is not None:
                xyz[i] = c.to(device)
                mask[i] = True
        return xyz, mask

    def atom_coords(
        self,
        chain_id: str,
        atom: str = "CA",
        *,
        device: Optional[Union[str, torch.device]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        返回指定原子（如 'CA','CB','N','C','O'）的坐标与 mask：
            coords: [L, 3]，mask: [L]，缺失该原子的残基 mask=False、坐标置 0。
        """
        if device is None:
            device = "cpu"
        name = atom.upper()
        recs = self.residues(chain_id)
        L = len(recs)
        xyz = torch.zeros(L, 3, dtype=torch.float32, device=device)
        mask = torch.zeros(L, dtype=torch.bool, device=device)
        for i, r in enumerate(recs):
            v = r.atoms.get(name)
            if v is not None:
                xyz[i] = v.to(device)
                mask[i] = True
        return xyz, mask

    def b_factor(
        self,
        chain_id: str,
        atom: str = "CA",
        *,
        device: Optional[Union[str, torch.device]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        返回指定原子的 B 因子与 mask（与 atom_coords 对齐）。
        缺失该原子的残基 B 因子置 0、mask=False。
        """
        if device is None:
            device = "cpu"
        name = atom.upper()
        recs = self.residues(chain_id)
        L = len(recs)
        vec = torch.zeros(L, dtype=torch.float32, device=device)
        mask = torch.zeros(L, dtype=torch.bool, device=device)
        for i, r in enumerate(recs):
            if name in r.b_factors:
                vec[i] = float(r.b_factors[name])
                mask[i] = True
        return vec, mask

    # ---------- 注释对齐（可将外部 (chain, resseq, icode) 集合映射为 per-residue 向量） ----------
    def annotation_mask(
        self,
        chain_id: str,
        annotations: Union[
            Iterable[Tuple[str, int, str]],
            Dict[Tuple[str, int, str], Union[bool, int, float, str]]
        ],
        *,
        as_float: bool = False,
        device: Optional[Union[str, torch.device]] = None,
    ) -> torch.Tensor:
        """
        将外部注释映射为长度为 L 的向量（bool 或 float32）。
        - annotations:
            1) 可迭代的 (chain, resseq, icode) 元组（集合/列表等）；
            2) 字典 {(chain, resseq, icode): 值}（非空/非零视为 True）。
        仅映射指定 chain_id 的条目。
        """
        if device is None:
            device = "cpu"
        recs = self.residues(chain_id)
        L = len(recs)
        out_bool = torch.zeros(L, dtype=torch.bool, device=device)

        def v2b(v) -> bool:
            if isinstance(v, (bool, int, float)):
                return bool(v)
            if isinstance(v, str):
                return len(v.strip()) > 0
            return True

        if isinstance(annotations, dict):
            for i, r in enumerate(recs):
                k = (chain_id, r.key.resseq, r.key.icode)
                if k in annotations and v2b(annotations[k]):
                    out_bool[i] = True
        else:
            for ch, rs, ic in annotations:
                if ch != chain_id:
                    continue
                idx = self._index[chain_id].get((int(rs), str(ic or " ")))
                if idx is not None:
                    out_bool[idx] = True

        return out_bool.float() if as_float else out_bool