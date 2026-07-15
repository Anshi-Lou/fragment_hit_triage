from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class ChemDescriptorAvailability:
    rdkit_available: bool
    message: str


def _try_import_rdkit():
    try:
        from rdkit import Chem  # type: ignore
        from rdkit.Chem import Crippen, Descriptors, Lipinski  # type: ignore
        return Chem, Crippen, Descriptors, Lipinski
    except Exception:
        return None, None, None, None


def rdkit_available() -> ChemDescriptorAvailability:
    Chem, *_ = _try_import_rdkit()
    if Chem is None:
        return ChemDescriptorAvailability(False, "RDKit is not installed; using supplied descriptors and SMILES text features only.")
    return ChemDescriptorAvailability(True, "RDKit is available; molecular descriptors can be computed from SMILES.")


def compute_rdkit_descriptors(smiles: str | None) -> dict[str, float]:
    """Compute a compact set of interpretable descriptors. Returns NaN values if RDKit is absent/SMILES invalid."""
    fields = [
        "rdkit_mw", "rdkit_logp", "rdkit_hba", "rdkit_hbd", "rdkit_rotatable_bonds",
        "rdkit_rings", "rdkit_aromatic_rings", "rdkit_tpsa", "rdkit_fraction_csp3",
        "rdkit_formal_charge",
    ]
    empty = {k: math.nan for k in fields}
    if smiles is None or pd.isna(smiles) or str(smiles).strip() == "":
        return empty
    Chem, Crippen, Descriptors, Lipinski = _try_import_rdkit()
    if Chem is None:
        return empty
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return empty
    try:
        return {
            "rdkit_mw": float(Descriptors.MolWt(mol)),
            "rdkit_logp": float(Crippen.MolLogP(mol)),
            "rdkit_hba": float(Lipinski.NumHAcceptors(mol)),
            "rdkit_hbd": float(Lipinski.NumHDonors(mol)),
            "rdkit_rotatable_bonds": float(Lipinski.NumRotatableBonds(mol)),
            "rdkit_rings": float(Lipinski.RingCount(mol)),
            "rdkit_aromatic_rings": float(Lipinski.NumAromaticRings(mol)),
            "rdkit_tpsa": float(Descriptors.TPSA(mol)),
            "rdkit_fraction_csp3": float(Descriptors.FractionCSP3(mol)),
            "rdkit_formal_charge": float(sum(a.GetFormalCharge() for a in mol.GetAtoms())),
        }
    except Exception:
        return empty


def add_rdkit_descriptors(df: pd.DataFrame, smiles_col: str | None) -> pd.DataFrame:
    if smiles_col is None or smiles_col not in df.columns:
        return df
    desc = pd.DataFrame([compute_rdkit_descriptors(s) for s in df[smiles_col]], index=df.index)
    for col in desc.columns:
        if col not in df.columns:
            df[col] = desc[col]
    return df
