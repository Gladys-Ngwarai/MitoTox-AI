import numpy as np

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import DataStructs
from rdkit.Chem import Descriptors
from rdkit.Chem import rdMolDescriptors


# ============================================================
# MITOTOX AI — FINAL FEATURE CONFIGURATION
# ============================================================

MORGAN_RADIUS = 2
MORGAN_NBITS = 2048

RDKIT_DESC_NAMES = [
    "MolWt",
    "MolLogP",
    "NumHDonors",
    "NumHAcceptors",
    "NumRotatableBonds",
    "NumAromaticRings",
    "NumHeteroatoms",
    "FractionCSP3",
    "TPSA",
]


# ============================================================
# MOLECULAR DESCRIPTORS
# ============================================================

def get_descriptors(smiles):

    mol = Chem.MolFromSmiles(str(smiles))

    if mol is None:
        return None

    return {
        "Molecular Weight": Descriptors.MolWt(mol),
        "LogP": Descriptors.MolLogP(mol),
        "H-Bond Donors": Descriptors.NumHDonors(mol),
        "H-Bond Acceptors": Descriptors.NumHAcceptors(mol),
        "Rotatable Bonds": Descriptors.NumRotatableBonds(mol),
        "Aromatic Rings": Descriptors.NumAromaticRings(mol),
        "Heteroatoms": Descriptors.NumHeteroatoms(mol),
        "Fraction CSP3": Descriptors.FractionCSP3(mol),
        "TPSA": rdMolDescriptors.CalcTPSA(mol),
    }


# ============================================================
# FINAL FEATURE VECTOR
#
# 2048 Morgan fingerprint
# + 9 RDKit descriptors
# = 2057 FEATURES
# ============================================================

def featurize(smiles):

    mol = Chem.MolFromSmiles(str(smiles))

    if mol is None:
        return None

    # --------------------------------------------------------
    # 1. Morgan fingerprint
    # --------------------------------------------------------

    fp = AllChem.GetMorganFingerprintAsBitVect(
        mol,
        MORGAN_RADIUS,
        MORGAN_NBITS
    )

    fp_arr = np.zeros(
        MORGAN_NBITS,
        dtype=np.float32
    )

    DataStructs.ConvertToNumpyArray(
        fp,
        fp_arr
    )

    # --------------------------------------------------------
    # 2. RDKit descriptors
    # --------------------------------------------------------

    desc = []

    for name in RDKIT_DESC_NAMES:

        if name == "TPSA":

            value = rdMolDescriptors.CalcTPSA(mol)

        else:

            value = getattr(
                Descriptors,
                name
            )(mol)

        desc.append(value)

    desc_arr = np.array(
        desc,
        dtype=np.float32
    )

    desc_arr = np.nan_to_num(
        desc_arr,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    # --------------------------------------------------------
    # 3. Combine
    # --------------------------------------------------------

    features = np.concatenate([
        fp_arr,
        desc_arr
    ])

    # --------------------------------------------------------
    # 4. SAFETY CHECK
    # --------------------------------------------------------

    if len(features) != 2057:

        raise ValueError(
            f"Feature length mismatch: "
            f"expected 2057, got {len(features)}"
        )

    return features