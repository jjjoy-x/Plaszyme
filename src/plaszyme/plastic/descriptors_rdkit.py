# plastic_featurizer.py
"""Plastic molecular feature extractor module.

This module provides comprehensive feature extraction for plastic polymers using
RDKit molecular descriptors. Supports standard descriptors, fragment counts,
charge descriptors, and intelligent density normalization for consistent
molecular representations across different polymer sizes.
"""

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
    """Plastic molecular feature extractor based on RDKit molecular descriptors.

    Extracts comprehensive molecular features from plastic polymer structures,
    including standard physicochemical descriptors, functional group counts,
    and charge distributions. Supports intelligent density normalization to
    ensure consistent representations across polymers of different sizes.

    Features:
        - Standard RDKit molecular descriptors (200+ features)
        - Functional group fragment counts
        - Gasteiger partial charge statistics
        - Intelligent density normalization
        - Batch processing capabilities
        - Flexible configuration via YAML files

    Attributes:
        config: Configuration dictionary loaded from file or defaults.
        normalize: Whether to apply density normalization.
        standard_descriptors: List of standard RDKit descriptor names.
        fragment_descriptors: List of functional group descriptor names.
        charge_descriptors: List of charge-based descriptor names.
        calculator: RDKit molecular descriptor calculator.
    """

    def __init__(self, config_path: Optional[str] = None):
        """Initialize feature extractor.

        Args:
            config_path: Path to YAML configuration file. If None or file doesn't
                        exist, uses default configuration with all standard descriptors.
        """
        self.config = self._load_config(config_path)
        self.normalize = self.config.get("normalize", True)

        # Get descriptor lists
        self.standard_descriptors = self._get_standard_descriptors()
        self.fragment_descriptors = self._get_fragment_descriptors()
        self.charge_descriptors = self._get_charge_descriptors()

        # Smart check: if normalization enabled but user hasn't configured HeavyAtomCount, give hint
        if self.normalize and 'HeavyAtomCount' not in self.standard_descriptors:
            logger.info(
                "Smart mode: detected normalization enabled but HeavyAtomCount not configured, will auto-calculate")

        # Initialize RDKit calculator
        if self.standard_descriptors:
            self.calculator = MoleculeDescriptors.MolecularDescriptorCalculator(self.standard_descriptors)
        else:
            self.calculator = None

        self._output_feature_names = None
        total_features = len(self.standard_descriptors) + len(self.fragment_descriptors) + len(self.charge_descriptors)

        logger.info(
            f"Initialization complete - standard:{len(self.standard_descriptors)}, "
            f"fragments:{len(self.fragment_descriptors)}, charges:{len(self.charge_descriptors)}")

    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load configuration from YAML file.

        Args:
            config_path: Path to configuration file.

        Returns:
            Configuration dictionary with normalization settings and descriptor names.
        """
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    return yaml.safe_load(f)
            except Exception as e:
                logger.warning(f"Configuration file loading failed: {e}")

        # Default configuration: use all standard descriptors
        return {"normalize": True, "descriptor_names": None}

    def _get_standard_descriptors(self) -> List[str]:
        """Get standard RDKit descriptors based on configuration.

        Returns:
            List of standard descriptor names to compute.
        """
        all_descriptors = [desc[0] for desc in Descriptors._descList]
        descriptor_names = self.config.get("descriptor_names", [])

        # Default exclusions (only applied when descriptor_names is None or [])
        exclude_by_default = {"Ipc"}

        if descriptor_names is None or descriptor_names == []:
            filtered = [d for d in all_descriptors if d not in exclude_by_default]
            logger.info(
                f"descriptor_names is {descriptor_names}, using all RDKit descriptors by default, "
                f"excluded: {sorted(exclude_by_default)}")
            return filtered

        # Explicit configuration: strictly follow user requirements, don't exclude even unrecommended ones
        return [d for d in descriptor_names if d in all_descriptors]

    def _get_fragment_descriptors(self) -> List[str]:
        """Get functional group descriptors based on configuration.

        Returns:
            List of fragment descriptor names to compute.
        """
        available_fragments = [
            'fr_ester', 'fr_amide', 'fr_ether', 'fr_benzene', 'fr_C_O',
            'fr_alkyl_halide', 'fr_ketone', 'fr_phenol', 'fr_nitrile'
        ]
        descriptor_names = self.config.get("descriptor_names", [])

        # Handle None case
        if descriptor_names is None:
            return available_fragments

        return [d for d in descriptor_names if d in available_fragments]

    def _get_charge_descriptors(self) -> List[str]:
        """Get charge-based descriptors based on configuration.

        Returns:
            List of charge descriptor names to compute.
        """
        available_charges = ['MaxPartialCharge', 'MinPartialCharge', 'MaxAbsPartialCharge']
        descriptor_names = self.config.get("descriptor_names", [])

        # Handle None case
        if descriptor_names is None:
            return available_charges

        return [d for d in descriptor_names if d in available_charges]

    def featurize_mol(self, mol: Chem.Mol) -> Optional[torch.Tensor]:
        """Extract feature vector from RDKit Mol object.

        Computes comprehensive molecular features including standard descriptors,
        functional group counts, and charge statistics. Applies intelligent
        density normalization if enabled.

        Args:
            mol: RDKit Mol object representing the polymer structure.

        Returns:
            Feature tensor of shape [num_features] or None if computation fails.
        """
        try:
            all_values = {}

            # 1. Standard descriptors
            if self.calculator and self.standard_descriptors:
                standard_values = self.calculator.CalcDescriptors(mol)
                all_values.update(dict(zip(self.standard_descriptors, standard_values)))

            # 2. Smart handling of HeavyAtomCount: auto-calculate if normalization needed but not configured
            if self.normalize and 'HeavyAtomCount' not in all_values:
                heavy_atom_count = Descriptors.HeavyAtomCount(mol)
                all_values['HeavyAtomCount'] = heavy_atom_count
                logger.info(f"Auto-calculated HeavyAtomCount = {heavy_atom_count}")

            # 3. Fragment descriptors
            for frag_name in self.fragment_descriptors:
                try:
                    frag_func = getattr(Fragments, frag_name)
                    all_values[frag_name] = frag_func(mol)
                except AttributeError:
                    all_values[frag_name] = 0

            # 4. Charge descriptors
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

            # 5. Handle NaN values
            for k, v in all_values.items():
                if pd.isna(v):
                    all_values[k] = 0.0

            # 6. Density normalization
            if self.normalize:
                all_values = self._normalize_features(all_values)

            # 7. Record feature names
            if self._output_feature_names is None:
                self._output_feature_names = list(all_values.keys())

            return torch.tensor(list(all_values.values()), dtype=torch.float32)

        except Exception as e:
            logger.error(f"Feature computation failed: {e}")
            return None

    def _normalize_features(self, raw_values: Dict[str, float]) -> Dict[str, float]:
        """Intelligent density normalization - normalize all features by default, exclude only unnecessary ones.

        Normalizes molecular features by heavy atom count to ensure size-independent
        representations. Automatically excludes features that are already ratios,
        intensities, or intrinsic properties that shouldn't be normalized.

        Args:
            raw_values: Dictionary of raw feature values.

        Returns:
            Dictionary with normalized feature values (original + density features).

        Raises:
            ValueError: If HeavyAtomCount is missing in normalization mode.
        """
        heavy_atoms = raw_values.get("HeavyAtomCount")

        if heavy_atoms is None:
            raise ValueError("Missing HeavyAtomCount in normalization mode, code logic error")

        if heavy_atoms <= 0:
            logger.warning(f"Abnormal HeavyAtomCount value: {heavy_atoms}, using 1 as default")
            heavy_atoms = 1

        norm_values = raw_values.copy()

        # Features that don't need normalization (already ratios, intensities, or intrinsic properties)
        non_normalizable_features = {
            # Ratio and percentage features (already normalized)
            "FractionCsp3",  # sp3 carbon fraction (0-1)

            # Intensity features (don't scale linearly with molecular size)
            "MolLogP",  # Lipophilicity
            "MolMR",  # Molecular refractivity
            "HallKierAlpha",  # Polarizability parameter

            # Charge features (intensity, not total)
            "MaxPartialCharge",
            "MinPartialCharge",
            "MaxAbsPartialCharge",
            "MinAbsPartialCharge",

            # Complex topological indices (already consider molecular size)
            "BalabanJ",  # Balaban index
            "BertzCT",  # Bertz complexity
            "Ipc",  # Information content index
            "Kappa1", "Kappa2", "Kappa3",  # Molecular shape indices

            # Connectivity indices (already normalized)
            "Chi0", "Chi0n", "Chi0v",
            "Chi1", "Chi1n", "Chi1v",
            "Chi2", "Chi2n", "Chi2v",
            "Chi3n", "Chi3v", "Chi4n", "Chi4v",

            # Fingerprint density features (already density)
            "FpDensityMorgan1", "FpDensityMorgan2", "FpDensityMorgan3",

            # Special cases
            "HeavyAtomCount",  # Used as normalization basis, don't normalize itself
        }

        # Normalize all other features
        for feature_name, feature_value in raw_values.items():
            if feature_name not in non_normalizable_features:
                density_name = feature_name + "Density"
                norm_values[density_name] = feature_value / heavy_atoms

        return norm_values

    def featurize_file(self, file_path: str) -> Optional[torch.Tensor]:
        """Extract features from file (supports .mol and .sdf).

        Args:
            file_path: Path to molecular structure file.

        Returns:
            Feature tensor or None if processing fails.
        """
        try:
            file_ext = os.path.splitext(file_path)[1].lower()

            if file_ext == '.mol':
                mol = Chem.MolFromMolFile(file_path, removeHs=False)
            elif file_ext == '.sdf':
                supplier = Chem.SDMolSupplier(file_path, removeHs=False)
                mol = next(supplier) if supplier else None
            else:
                logger.warning(f"Unsupported file format: {file_ext}")
                return None

            if mol is None:
                logger.warning(f"Cannot parse file: {file_path}")
                return None

            return self.featurize_mol(mol)

        except Exception as e:
            logger.error(f"File processing failed {file_path}: {e}")
            return None

    def featurize_folder(self, folder_path: str, show_progress: bool = True) -> Tuple[Dict[str, torch.Tensor], Dict]:
        """Batch process molecular files in folder.

        Args:
            folder_path: Path to folder containing molecular structure files.
            show_progress: Whether to show progress bar during processing.

        Returns:
            Tuple of (feature_dictionary, statistics_dictionary).

        Raises:
            ValueError: If folder doesn't exist or contains no molecular files.
        """
        if not os.path.isdir(folder_path):
            raise ValueError(f"Directory does not exist: {folder_path}")

        mol_files = [f for f in os.listdir(folder_path) if f.endswith(('.mol', '.sdf'))]
        if not mol_files:
            raise ValueError(f"No molecular files in directory: {folder_path}")

        feature_dict = {}
        failed_files = []

        iterator = tqdm(mol_files, desc="Extracting features") if show_progress else mol_files

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

        logger.info(f"Feature extraction complete: {stats['successful']}/{stats['total_files']} successful")
        return feature_dict, stats

    def save_features(self, feature_dict: Dict[str, torch.Tensor], output_prefix: str):
        """Save features to CSV and PyTorch files.

        Args:
            feature_dict: Dictionary mapping polymer names to feature tensors.
            output_prefix: Output file prefix (without extension).

        Returns:
            Tuple of (csv_path, pt_path).

        Raises:
            ValueError: If feature dictionary is empty.
        """
        if not feature_dict:
            raise ValueError("Feature dictionary is empty")

        # Save CSV
        csv_path = f"{output_prefix}.csv"
        data_dict = {name: features.tolist() for name, features in feature_dict.items()}
        df = pd.DataFrame(data_dict).T

        if self._output_feature_names and len(self._output_feature_names) == df.shape[1]:
            df.columns = self._output_feature_names
        else:
            df.columns = [f"feature_{i}" for i in range(df.shape[1])]

        df.index.name = "plastic"
        df.to_csv(csv_path)

        # Save PyTorch file
        pt_path = f"{output_prefix}.pt"
        save_dict = {
            "features": feature_dict,
            "feature_names": self._output_feature_names,
            "config": self.config,
            "num_features": len(self._output_feature_names) if self._output_feature_names else 0
        }
        torch.save(save_dict, pt_path)

        logger.info(f"Features saved: {csv_path}, {pt_path}")
        return csv_path, pt_path

    def get_feature_names(self) -> Optional[List[str]]:
        """Get list of feature names.

        Returns:
            List of feature names or None if not yet computed.
        """
        return self._output_feature_names

    @classmethod
    def load_features(cls, pt_path: str) -> Dict:
        """Load saved feature data.

        Args:
            pt_path: Path to saved PyTorch feature file.

        Returns:
            Dictionary containing features, feature names, config, and metadata.
        """
        return torch.load(pt_path, map_location='cpu')


# Test code
if __name__ == "__main__":
    # Test input/output paths
    INPUT_DIR = "/Users/shulei/PycharmProjects/Plaszyme/plastic/mols_for_unimol_10_sdf_new"  # Your SDF/MOL folder
    CONFIG_PATH = "/Users/shulei/PycharmProjects/Plaszyme/plastic/mol_features/rdkit_features.yaml"  # Config file
    OUTPUT_PREFIX = "/Users/shulei/PycharmProjects/Plaszyme/test/outputs/all_description_new_less"  # Output prefix

    print("Testing plastic feature extractor...")

    try:
        # Initialize extractor
        extractor = PlasticFeaturizer(CONFIG_PATH)

        # Process folder
        if os.path.exists(INPUT_DIR):
            features, stats = extractor.featurize_folder(INPUT_DIR)

            # Save results
            os.makedirs(os.path.dirname(OUTPUT_PREFIX), exist_ok=True)
            csv_path, pt_path = extractor.save_features(features, OUTPUT_PREFIX)

            # Output results
            print(f"\nTest completed!")
            print(f"Processed: {stats['successful']}/{stats['total_files']} files")
            print(f"Feature dimension: {stats['feature_dim']}")
            print(f"Output files: {csv_path}, {pt_path}")

            if stats['failed_files']:
                print(f"Warning: Failed files: {stats['failed_files']}")

            # Quick verification
            loaded = PlasticFeaturizer.load_features(pt_path)
            print(f"Verification: loaded {len(loaded['features'])} polymer features")

        else:
            print(f"Input directory does not exist: {INPUT_DIR}")
            print("Please modify INPUT_DIR to your SDF folder path")

    except Exception as e:
        print(f"Test failed: {e}")
        import traceback

        traceback.print_exc()