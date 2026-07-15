from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import pandas as pd


def read_table(path: str | Path) -> pd.DataFrame:
    """Read CSV/TSV with a forgiving separator detector."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in {".tsv", ".txt"}:
        try:
            return pd.read_csv(path, sep="\t")
        except Exception:
            return pd.read_csv(path, sep=None, engine="python")
    return pd.read_csv(path, sep=None, engine="python")


def write_table(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() in {".tsv", ".txt"}:
        df.to_csv(path, sep="\t", index=False)
    else:
        df.to_csv(path, index=False)


def normalize_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name).strip().lower())


def find_col(df: pd.DataFrame, candidates: Iterable[str], required: bool = False, label: str | None = None) -> str | None:
    """Find a column by aliases, ignoring case, spaces, punctuation and underscores."""
    norm_to_original = {normalize_name(c): c for c in df.columns}
    for cand in candidates:
        key = normalize_name(cand)
        if key in norm_to_original:
            return norm_to_original[key]
    if required:
        pretty = label or "/".join(candidates)
        raise ValueError(f"Could not find required column: {pretty}. Available columns: {list(df.columns)}")
    return None


def parse_bool_series(s: pd.Series) -> pd.Series:
    """Convert common boolean-like values to 0/1 floats while preserving missing values."""
    if s is None:
        return pd.Series(dtype="float")
    if pd.api.types.is_bool_dtype(s):
        return s.astype(float)
    if pd.api.types.is_numeric_dtype(s):
        return (pd.to_numeric(s, errors="coerce") > 0).astype(float)
    true_values = {"true", "t", "yes", "y", "1", "hit", "enriched", "positive", "pos", "+"}
    false_values = {"false", "f", "no", "n", "0", "nonhit", "not_hit", "negative", "neg", "-"}
    out = []
    for value in s.astype("string"):
        if value is pd.NA or value is None:
            out.append(float("nan"))
        else:
            key = str(value).strip().lower()
            if key in true_values:
                out.append(1.0)
            elif key in false_values:
                out.append(0.0)
            else:
                try:
                    out.append(1.0 if float(key) > 0 else 0.0)
                except Exception:
                    out.append(float("nan"))
    return pd.Series(out, index=s.index, dtype="float")
