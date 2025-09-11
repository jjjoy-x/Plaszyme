# plastic_featurizer.py
# 塑料分子特征提取器 - 智能归一化版本

import os
import logging
from typing import List, Dict, Optional, Tuple
import torch
import pandas as pd
import yaml
from rdkit import Chem
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem import Descriptors, Fragments
from rdkit.Chem.rdPartialCharges import ComputeGasteigerCharges
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PlasticFeaturizer:
    """塑料分子特征提取器 - 基于RDKit分子描述符"""

    def __init__(self, config_path: Optional[str] = None):
        """初始化特征提取器"""
        self.config = self._load_config(config_path)
        self.normalize = self.config.get("normalize", True)

        # 获取描述符列表
        self.standard_descriptors = self._get_standard_descriptors()
        self.fragment_descriptors = self._get_fragment_descriptors()
        self.charge_descriptors = self._get_charge_descriptors()

        # 智能检查：如果启用归一化但用户没有配置HeavyAtomCount，给出提示
        if self.normalize and 'HeavyAtomCount' not in self.standard_descriptors:
            logger.info("🤖 智能模式：检测到归一化开启但未配置HeavyAtomCount，将自动计算")

        # 初始化RDKit计算器
        if self.standard_descriptors:
            self.calculator = MoleculeDescriptors.MolecularDescriptorCalculator(self.standard_descriptors)
        else:
            self.calculator = None

        self._output_feature_names = None
        total_features = len(self.standard_descriptors) + len(self.fragment_descriptors) + len(self.charge_descriptors)

        logger.info(
            f"初始化完成 - 标准:{len(self.standard_descriptors)}, 官能团:{len(self.fragment_descriptors)}, 电荷:{len(self.charge_descriptors)}")

    def _load_config(self, config_path: Optional[str]) -> Dict:
        """加载配置文件"""
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    return yaml.safe_load(f)
            except Exception as e:
                logger.warning(f"配置文件加载失败: {e}")

        # 默认配置：使用全部标准描述符
        return {"normalize": True, "descriptor_names": None}

    def _get_standard_descriptors(self) -> List[str]:
        """获取标准RDKit描述符"""
        all_descriptors = [desc[0] for desc in Descriptors._descList]
        descriptor_names = self.config.get("descriptor_names", [])

        # ✅ 默认排除的描述符（只在 descriptor_names 为 None 或 [] 时生效）
        exclude_by_default = {"Ipc"}

        if descriptor_names is None or descriptor_names == []:
            filtered = [d for d in all_descriptors if d not in exclude_by_default]
            logger.info(
                f"descriptor_names 为 {descriptor_names}，默认使用全部 RDKit 描述符，已排除: {sorted(exclude_by_default)}")
            return filtered

        # ✅ 显式配置时，严格按用户要求保留，即使包含不推荐的也不排除
        return [d for d in descriptor_names if d in all_descriptors]

    def _get_fragment_descriptors(self) -> List[str]:
        """获取官能团描述符"""
        available_fragments = [
            'fr_ester', 'fr_amide', 'fr_ether', 'fr_benzene', 'fr_C_O',
            'fr_alkyl_halide', 'fr_ketone', 'fr_phenol', 'fr_nitrile'
        ]
        descriptor_names = self.config.get("descriptor_names", [])

        # 处理None的情况
        if descriptor_names is None:
            return available_fragments

        return [d for d in descriptor_names if d in available_fragments]

    def _get_charge_descriptors(self) -> List[str]:
        """获取电荷描述符"""
        available_charges = ['MaxPartialCharge', 'MinPartialCharge', 'MaxAbsPartialCharge']
        descriptor_names = self.config.get("descriptor_names", [])

        # 处理None的情况
        if descriptor_names is None:
            return available_charges

        return [d for d in descriptor_names if d in available_charges]

    def featurize_mol(self, mol: Chem.Mol) -> Optional[torch.Tensor]:
        """从RDKit Mol对象提取特征向量"""
        try:
            all_values = {}

            # 1. 标准描述符
            if self.calculator and self.standard_descriptors:
                standard_values = self.calculator.CalcDescriptors(mol)
                all_values.update(dict(zip(self.standard_descriptors, standard_values)))

            # 2. 智能处理HeavyAtomCount：如果需要归一化但用户没有配置，自动计算
            if self.normalize and 'HeavyAtomCount' not in all_values:
                heavy_atom_count = Descriptors.HeavyAtomCount(mol)
                all_values['HeavyAtomCount'] = heavy_atom_count
                logger.info(f"自动计算HeavyAtomCount = {heavy_atom_count}")

            # 3. 官能团描述符
            for frag_name in self.fragment_descriptors:
                try:
                    frag_func = getattr(Fragments, frag_name)
                    all_values[frag_name] = frag_func(mol)
                except AttributeError:
                    all_values[frag_name] = 0

            # 4. 电荷描述符
            if self.charge_descriptors:
                try:
                    ComputeGasteigerCharges(mol)
                    charges = [float(atom.GetProp('_GasteigerCharge')) for atom in mol.GetAtoms()]
                    charges = [c for c in charges if not pd.isna(c)]

                    if charges:
                        if 'MaxPartialCharge' in self.charge_descriptors:
                            all_values['MaxPartialCharge'] = max(charges)
                        if 'MinPartialCharge' in self.charge_descriptors:
                            all_values['MinPartialCharge'] = min(charges)
                        if 'MaxAbsPartialCharge' in self.charge_descriptors:
                            all_values['MaxAbsPartialCharge'] = max(abs(c) for c in charges)
                    else:
                        for charge_desc in self.charge_descriptors:
                            all_values[charge_desc] = 0.0
                except:
                    for charge_desc in self.charge_descriptors:
                        all_values[charge_desc] = 0.0

            # 5. 处理NaN值
            for k, v in all_values.items():
                if pd.isna(v):
                    all_values[k] = 0.0

            # 6. 密度归一化
            if self.normalize:
                all_values = self._normalize_features(all_values)

            # 7. 记录特征名称
            if self._output_feature_names is None:
                self._output_feature_names = list(all_values.keys())

            return torch.tensor(list(all_values.values()), dtype=torch.float32)

        except Exception as e:
            logger.error(f"特征计算失败: {e}")
            return None

    def _normalize_features(self, raw_values: Dict[str, float]) -> Dict[str, float]:
        """智能密度归一化 - 默认归一化所有特征，只排除不需要的"""
        heavy_atoms = raw_values.get("HeavyAtomCount")

        if heavy_atoms is None:
            raise ValueError("归一化模式下缺少HeavyAtomCount，代码逻辑错误")

        if heavy_atoms <= 0:
            logger.warning(f"异常的HeavyAtomCount值: {heavy_atoms}，使用1作为默认值")
            heavy_atoms = 1

        norm_values = raw_values.copy()

        # 不需要归一化的特征（已经是比率、强度或内在性质）
        non_normalizable_features = {
            # 比率和百分比特征（已经归一化）
            "FractionCsp3",  # sp3碳比例（0-1）

            # 强度型特征（不随分子大小线性变化）
            "MolLogP",  # 亲脂性
            "MolMR",  # 分子折射率
            "HallKierAlpha",  # 极化率参数

            # 电荷特征（强度，不是总量）
            "MaxPartialCharge",
            "MinPartialCharge",
            "MaxAbsPartialCharge",
            "MinAbsPartialCharge",

            # 复杂拓扑指数（已考虑分子大小）
            "BalabanJ",  # Balaban指数
            "BertzCT",  # Bertz复杂度
            "Ipc",  # 信息内容指数
            "Kappa1", "Kappa2", "Kappa3",  # 分子形状指数

            # 连接性指数（已归一化）
            "Chi0", "Chi0n", "Chi0v",
            "Chi1", "Chi1n", "Chi1v",
            "Chi2", "Chi2n", "Chi2v",
            "Chi3n", "Chi3v", "Chi4n", "Chi4v",

            # 指纹密度特征（已经是密度）
            "FpDensityMorgan1", "FpDensityMorgan2", "FpDensityMorgan3",

            # 特殊情况
            "HeavyAtomCount",  # 用作归一化基准，本身不归一化
        }

        # 对所有其他特征进行归一化
        for feature_name, feature_value in raw_values.items():
            if feature_name not in non_normalizable_features:
                density_name = feature_name + "Density"
                norm_values[density_name] = feature_value / heavy_atoms

        return norm_values

    def featurize_file(self, file_path: str) -> Optional[torch.Tensor]:
        """从文件提取特征（支持.mol和.sdf）"""
        try:
            file_ext = os.path.splitext(file_path)[1].lower()

            if file_ext == '.mol':
                mol = Chem.MolFromMolFile(file_path, removeHs=False)
            elif file_ext == '.sdf':
                supplier = Chem.SDMolSupplier(file_path, removeHs=False)
                mol = next(supplier) if supplier else None
            else:
                logger.warning(f"不支持的文件格式: {file_ext}")
                return None

            if mol is None:
                logger.warning(f"无法解析文件: {file_path}")
                return None

            return self.featurize_mol(mol)

        except Exception as e:
            logger.error(f"文件处理失败 {file_path}: {e}")
            return None

    def featurize_folder(self, folder_path: str, show_progress: bool = True) -> Tuple[Dict[str, torch.Tensor], Dict]:
        """批量处理文件夹中的分子文件"""
        if not os.path.isdir(folder_path):
            raise ValueError(f"目录不存在: {folder_path}")

        mol_files = [f for f in os.listdir(folder_path) if f.endswith(('.mol', '.sdf'))]
        if not mol_files:
            raise ValueError(f"目录中没有分子文件: {folder_path}")

        feature_dict = {}
        failed_files = []

        iterator = tqdm(mol_files, desc="提取特征") if show_progress else mol_files

        for fname in iterator:
            file_path = os.path.join(folder_path, fname)
            features = self.featurize_file(file_path)

            if features is not None:
                name = os.path.splitext(fname)[0]
                feature_dict[name] = features
            else:
                failed_files.append(fname)

        stats = {
            "total_files": len(mol_files),
            "successful": len(feature_dict),
            "failed": len(failed_files),
            "failed_files": failed_files,
            "feature_dim": len(features) if features is not None else 0
        }

        logger.info(f"特征提取完成: {stats['successful']}/{stats['total_files']} 成功")
        return feature_dict, stats

    def save_features(self, feature_dict: Dict[str, torch.Tensor], output_prefix: str):
        """保存特征到CSV和PyTorch文件"""
        if not feature_dict:
            raise ValueError("特征字典为空")

        # 保存CSV
        csv_path = f"{output_prefix}.csv"
        data_dict = {name: features.tolist() for name, features in feature_dict.items()}
        df = pd.DataFrame(data_dict).T

        if self._output_feature_names and len(self._output_feature_names) == df.shape[1]:
            df.columns = self._output_feature_names
        else:
            df.columns = [f"feature_{i}" for i in range(df.shape[1])]

        df.index.name = "plastic"
        df.to_csv(csv_path)

        # 保存PyTorch文件
        pt_path = f"{output_prefix}.pt"
        save_dict = {
            "features": feature_dict,
            "feature_names": self._output_feature_names,
            "config": self.config,
            "num_features": len(self._output_feature_names) if self._output_feature_names else 0
        }
        torch.save(save_dict, pt_path)

        logger.info(f"特征保存完成: {csv_path}, {pt_path}")
        return csv_path, pt_path

    def get_feature_names(self) -> Optional[List[str]]:
        """获取特征名称列表"""
        return self._output_feature_names

    @classmethod
    def load_features(cls, pt_path: str) -> Dict:
        """加载保存的特征数据"""
        return torch.load(pt_path, map_location='cpu')


# 测试代码
if __name__ == "__main__":
    # 测试用的输入输出路径
    INPUT_DIR = "/Users/shulei/PycharmProjects/Plaszyme/plastic/mols_for_unimol_10_sdf_new"  # 您的SDF/MOL文件夹
    CONFIG_PATH = "/Users/shulei/PycharmProjects/Plaszyme/plastic/mol_features/rdkit_features.yaml"  # 配置文件
    OUTPUT_PREFIX = "/Users/shulei/PycharmProjects/Plaszyme/test/outputs/all_description_new_less"  # 输出前缀

    print("🧪 开始测试塑料特征提取器...")

    try:
        # 初始化提取器
        extractor = PlasticFeaturizer(CONFIG_PATH)

        # 处理文件夹
        if os.path.exists(INPUT_DIR):
            features, stats = extractor.featurize_folder(INPUT_DIR)

            # 保存结果
            os.makedirs(os.path.dirname(OUTPUT_PREFIX), exist_ok=True)
            csv_path, pt_path = extractor.save_features(features, OUTPUT_PREFIX)

            # 输出结果
            print(f"\n✅ 测试完成!")
            print(f"处理: {stats['successful']}/{stats['total_files']} 个文件")
            print(f"特征维度: {stats['feature_dim']}")
            print(f"输出文件: {csv_path}, {pt_path}")

            if stats['failed_files']:
                print(f"⚠️  失败文件: {stats['failed_files']}")

            # 快速验证
            loaded = PlasticFeaturizer.load_features(pt_path)
            print(f"验证加载: {len(loaded['features'])} 种塑料特征")

        else:
            print(f"❌ 输入目录不存在: {INPUT_DIR}")
            print("请修改INPUT_DIR为您的SDF文件夹路径")

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback

        traceback.print_exc()