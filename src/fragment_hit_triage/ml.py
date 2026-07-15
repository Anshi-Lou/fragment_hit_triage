from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DEFAULT_FEATURE_COLUMNS = [
    "log2fc_numeric", "pvalue_numeric", "is_hit", "iscore", "iscore_percentile",
    "signal_quality_score", "fragment_risk", "protein_risk", "partitioning_risk",
    "fragment_hit_count", "fragment_hit_ratio", "protein_hit_count", "protein_hit_ratio",
    "screen_priority_score", "lower_dose_supported", "competition_supported",
    "orthogonal_supported", "sar_supported", "actionable_family_flag",
    "rdkit_mw", "rdkit_logp", "rdkit_hba", "rdkit_hbd", "rdkit_rotatable_bonds",
    "rdkit_rings", "rdkit_aromatic_rings", "rdkit_tpsa", "rdkit_fraction_csp3", "rdkit_formal_charge",
]


def available_features(df: pd.DataFrame, label_col: str, smiles_col: str | None = "smiles") -> tuple[list[str], str | None]:
    numeric = [c for c in DEFAULT_FEATURE_COLUMNS if c in df.columns and c != label_col]
    usable_smiles = smiles_col if smiles_col in df.columns else None
    return numeric, usable_smiles


def build_pipeline(numeric_cols: list[str], smiles_col: str | None) -> Pipeline:
    transformers: list[tuple[str, Any, Any]] = []
    if numeric_cols:
        num_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler(with_mean=False)),
        ])
        transformers.append(("numeric", num_pipeline, numeric_cols))
    if smiles_col is not None:
        smiles_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value="")),
            ("flatten", _ColumnFlattener()),
            ("hash", HashingVectorizer(analyzer="char", ngram_range=(2, 5), n_features=1024, alternate_sign=False, norm="l2")),
        ])
        transformers.append(("smiles", smiles_pipeline, [smiles_col]))
    if not transformers:
        raise ValueError("No usable numeric features or SMILES column found.")
    pre = ColumnTransformer(transformers=transformers, sparse_threshold=0.3)
    clf = LogisticRegression(max_iter=3000, class_weight="balanced", solver="liblinear")
    return Pipeline([("features", pre), ("classifier", clf)])


class _ColumnFlattener:
    """Sklearn transformer that converts a one-column array/dataframe to a list of strings."""
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            vals = X.iloc[:, 0].fillna("").astype(str).tolist()
        else:
            vals = pd.Series(np.asarray(X).ravel()).fillna("").astype(str).tolist()
        return vals


def train_direct_hit_model(
    scored_df: pd.DataFrame,
    label_col: str,
    outdir: str | Path,
    smiles_col: str | None = "smiles",
    test_size: float = 0.25,
    random_state: int = 7,
) -> dict[str, Any]:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    df = scored_df.copy()
    if label_col not in df.columns:
        raise ValueError(f"Label column {label_col!r} not found in scored dataframe.")
    y = pd.to_numeric(df[label_col], errors="coerce")
    mask = y.notna()
    df = df.loc[mask].copy()
    y = y.loc[mask].astype(int)
    if y.nunique() < 2:
        raise ValueError("Need both positive and negative labels to train a model.")
    numeric_cols, smiles = available_features(df, label_col=label_col, smiles_col=smiles_col)
    pipe = build_pipeline(numeric_cols, smiles)

    # Stratified split only when each class has enough samples.
    vc = y.value_counts()
    can_split = len(df) >= 12 and vc.min() >= 2
    if can_split:
        X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=test_size, random_state=random_state, stratify=y)
        pipe.fit(X_train, y_train)
        pred_prob = pipe.predict_proba(X_test)[:, 1]
        pred = (pred_prob >= 0.5).astype(int)
        metrics = {
            "n_train": int(len(X_train)),
            "n_test": int(len(X_test)),
            "positive_rate_train": float(np.mean(y_train)),
            "positive_rate_test": float(np.mean(y_test)),
            "auroc": float(roc_auc_score(y_test, pred_prob)) if len(np.unique(y_test)) > 1 else None,
            "average_precision": float(average_precision_score(y_test, pred_prob)),
            "f1_at_0_5": float(f1_score(y_test, pred, zero_division=0)),
        }
    else:
        pipe.fit(df, y)
        metrics = {
            "n_train": int(len(df)),
            "n_test": 0,
            "positive_rate_train": float(np.mean(y)),
            "positive_rate_test": None,
            "auroc": None,
            "average_precision": None,
            "f1_at_0_5": None,
            "warning": "Dataset too small for a reliable stratified train/test split; model fit on all labeled rows.",
        }

    bundle = {
        "pipeline": pipe,
        "label_col": label_col,
        "numeric_cols": numeric_cols,
        "smiles_col": smiles,
        "metadata": metrics,
    }
    model_path = outdir / "direct_hit_model.joblib"
    joblib.dump(bundle, model_path)
    with open(outdir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    return {"model_path": str(model_path), "metrics": metrics, "numeric_cols": numeric_cols, "smiles_col": smiles}


def load_model(path: str | Path) -> dict[str, Any]:
    return joblib.load(path)


def predict_with_model(scored_df: pd.DataFrame, model_bundle: dict[str, Any]) -> pd.DataFrame:
    df = scored_df.copy()
    pipe = model_bundle["pipeline"]
    prob = pipe.predict_proba(df)[:, 1]
    df["ml_direct_hit_probability"] = prob
    if "direct_hit_priority_score" in df.columns:
        df["hybrid_priority_score"] = (0.50 * df["direct_hit_priority_score"] + 50.0 * prob).clip(0, 100)
    else:
        df["hybrid_priority_score"] = (100.0 * prob).clip(0, 100)
    return df
