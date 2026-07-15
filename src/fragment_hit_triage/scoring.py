from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .chem import add_rdkit_descriptors
from .io import find_col, parse_bool_series
from .recommendations import build_reason, build_recommendation


@dataclass
class TriageConfig:
    log2fc_threshold: float = 2.3
    pvalue_threshold: float = 0.05
    competition_fc_threshold: float = -1.0
    lower_dose_log2fc_threshold: float = 1.0
    high_priority_threshold: float = 70.0
    medium_priority_threshold: float = 50.0


DEFAULT_LABELING_BIAS_PROTEINS = {
    # Conservative examples frequently seen as background/sticky or compartment-associated in FFF/photo-crosslinking data.
    "TIMM17A", "TIMM17B", "TOMM22", "VDAC1", "VDAC2", "HMOX1", "HMOX2",
    "PRCP", "SCARB1", "SCARB2", "PCYOX1", "TMEM97", "GPR107", "KDELR1",
    "PPT1", "LAMP1", "NPC2", "NDUFB3", "COX7A2", "ATP5L", "CYP51A1",
}

PARTITION_TERMS = {
    "lysosome": ["lysosome", "lysosomal", "late endosome", "endolysosome", "autophagosome", "autophagy"],
    "mitochondria": ["mitochond", "mitochondrion", "mitochondrial"],
    "membrane": ["membrane", "transmembrane", "integral membrane"],
    "er_golgi": ["endoplasmic reticulum", " er", "golgi", "secretory pathway"],
}

FAMILY_ACTIONABLE_TERMS = ["kinase", "slc", "transporter", "e3", "ligase", "enzyme", "gpcr", "protease"]


def _num(df: pd.DataFrame, col: str | None, default: float = math.nan) -> pd.Series:
    if col is None or col not in df.columns:
        return pd.Series(default, index=df.index, dtype="float")
    return pd.to_numeric(df[col], errors="coerce")


def _text(df: pd.DataFrame, col: str | None, default: str = "") -> pd.Series:
    if col is None or col not in df.columns:
        return pd.Series(default, index=df.index, dtype="string")
    return df[col].astype("string").fillna(default)


def _clip01(x: pd.Series | np.ndarray | float) -> pd.Series | np.ndarray | float:
    return np.clip(x, 0.0, 1.0)


def _rank01(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    if s.notna().sum() == 0:
        return pd.Series(0.0, index=s.index)
    if s.nunique(dropna=True) <= 1:
        return pd.Series(0.0, index=s.index)
    return s.rank(pct=True, method="average").fillna(0.0)


def _weighted_mean(parts: list[tuple[pd.Series, float]]) -> pd.Series:
    if not parts:
        raise ValueError("No score parts provided")
    idx = parts[0][0].index
    num = pd.Series(0.0, index=idx)
    den = pd.Series(0.0, index=idx)
    for values, weight in parts:
        v = pd.to_numeric(values, errors="coerce")
        mask = v.notna()
        num.loc[mask] += v.loc[mask] * weight
        den.loc[mask] += weight
    return (num / den.replace(0, np.nan)).fillna(0.5)


def infer_columns(df: pd.DataFrame, column_overrides: dict[str, str | None] | None = None) -> dict[str, str | None]:
    overrides = column_overrides or {}
    aliases: dict[str, list[str]] = {
        "fragment_id": ["fragment_id", "fragment", "compound", "compound_id", "fff", "ligand", "molecule_id"],
        "protein": ["protein", "protein_id", "gene", "gene_name", "target", "uniprot", "accession"],
        "smiles": ["smiles", "SMILES", "canonical_smiles", "molecule_smiles"],
        "log2fc": ["log2fc", "log2_fc", "fold_change_log2", "enrichment_fc", "enrichment_log2fc", "log2 fold change"],
        "pvalue": ["pvalue", "p_value", "p", "pval"],
        "adj_pvalue": ["adj_pvalue", "adj_p", "padj", "qvalue", "q_value", "fdr"],
        "hit": ["hit", "is_hit", "enriched", "significant", "selected_hit", "regulated"],
        "intensity": ["intensity", "detected_intensity", "abundance", "ms_intensity", "protein_intensity"],
        "replicate_corr": ["replicate_corr", "replicate_correlation", "rep_corr", "pearson_r"],
        "replicate_cv": ["replicate_cv", "cv", "coefficient_variation"],
        "replicate_sd": ["replicate_log2fc_sd", "log2fc_sd", "replicate_sd"],
        "crf_log2fc": ["crf_log2fc", "control_log2fc", "crf_fc", "crf_enrichment"],
        "no_uv_log2fc": ["no_uv_log2fc", "nouv_log2fc", "no_uv_fc"],
        "bead_log2fc": ["bead_log2fc", "bead_background", "beads_log2fc"],
        "compartment": ["compartment", "localization", "go_cc", "cellular_component", "subcellular_location"],
        "family": ["family", "protein_family", "go_mf", "molecular_function", "class", "target_class"],
        "known_labeling_bias": ["known_labeling_bias", "labeling_bias", "background_contaminant", "photocrosslinker_bias"],
        "fragment_promiscuity_probability": ["predicted_promiscuity_probability", "promiscuity_probability", "fragment_promiscuity_probability", "promiscuity_score"],
        "clogp": ["clogp", "logp", "wildman_crippen_logp", "wc_logp", "rdkit_logp"],
        "mw": ["mw", "molecular_weight", "molwt", "rdkit_mw"],
        "aromatic_rings": ["aromatic_rings", "num_aromatic_rings", "rdkit_aromatic_rings"],
        "formal_charge": ["formal_charge", "charge", "rdkit_formal_charge"],
        "lower_dose_log2fc": ["lower_dose_log2fc", "low_dose_log2fc", "25um_log2fc", "repeat_25um_log2fc"],
        "lower_dose_hit": ["lower_dose_hit", "low_dose_hit", "25um_hit"],
        "competition_fc": ["competition_fc", "competition_log2fc", "competition_fold_change", "analog_competition_fc"],
        "competition_supported": ["competition_supported", "competed", "analog_competed", "free_analog_competition"],
        "orthogonal_supported": ["orthogonal_supported", "orthogonal_validation", "recombinant_positive", "spr_positive", "mst_positive", "dsf_positive", "nmr_positive", "cetsa_positive", "nanobret_positive", "functional_assay_positive"],
        "sar_supported": ["sar_supported", "sar_coherent", "sar", "coherent_sar"],
        "label": ["true_direct_hit", "direct_hit_label", "label", "validated_hit", "binding_hit"],
    }
    out: dict[str, str | None] = {}
    for key, cands in aliases.items():
        if key in overrides and overrides[key]:
            out[key] = overrides[key]
        else:
            out[key] = find_col(df, cands)
    # Required columns are checked later to allow app-based warnings.
    return out


def infer_hits(df: pd.DataFrame, cols: dict[str, str | None], config: TriageConfig) -> pd.Series:
    hit_col = cols.get("hit")
    if hit_col is not None:
        parsed = parse_bool_series(df[hit_col])
        if parsed.notna().sum() > 0:
            return parsed.fillna(0).astype(int)
    log2fc = _num(df, cols.get("log2fc"), default=0.0)
    p_col = cols.get("adj_pvalue") or cols.get("pvalue")
    pvalue = _num(df, p_col) if p_col else pd.Series(np.nan, index=df.index)
    if pvalue.notna().sum() > 0:
        return ((log2fc >= config.log2fc_threshold) & (pvalue <= config.pvalue_threshold)).astype(int)
    return (log2fc >= config.log2fc_threshold).astype(int)


def _contains_any(text: Any, terms: list[str]) -> bool:
    value = str(text).lower() if text is not None and not pd.isna(text) else ""
    return any(t.lower() in value for t in terms)


def _compute_partitioning(df: pd.DataFrame, fragment_col: str, hit_col: str, compartment_col: str | None) -> pd.DataFrame:
    fragments = pd.Index(df[fragment_col].astype(str).unique(), name=fragment_col)
    if compartment_col is None or compartment_col not in df.columns:
        return pd.DataFrame({fragment_col: fragments, "partitioning_risk": 0.0, "top_partitioning_signature": "none", "partitioning_odds_ratio": 1.0})

    work = df[[fragment_col, hit_col, compartment_col]].copy()
    work[fragment_col] = work[fragment_col].astype(str)
    work["compartment_text"] = work[compartment_col].astype("string").str.lower().fillna("")
    total_bg = len(work)
    out_rows = []
    for frag, g in work.groupby(fragment_col, dropna=False):
        hits = g[g[hit_col] == 1]
        if len(hits) == 0:
            out_rows.append({fragment_col: str(frag), "partitioning_risk": 0.0, "top_partitioning_signature": "none", "partitioning_odds_ratio": 1.0})
            continue
        best_risk = 0.0
        best_name = "none"
        best_or = 1.0
        hit_total = len(hits)
        for name, terms in PARTITION_TERMS.items():
            bg_term = work["compartment_text"].apply(lambda x: _contains_any(x, terms)).sum()
            hit_term = hits["compartment_text"].apply(lambda x: _contains_any(x, terms)).sum()
            # A single target in a compartment is not enough to call partitioning;
            # partitioning is a whole-interactome pattern.
            if hit_term < 3:
                continue
            # Haldane-Anscombe corrected odds ratio.
            a = hit_term + 0.5
            b = hit_total - hit_term + 0.5
            c = max(bg_term - hit_term, 0) + 0.5
            d = max(total_bg - bg_term - hit_total + hit_term, 0) + 0.5
            odds = (a / b) / (c / d)
            frac = hit_term / max(hit_total, 1)
            risk = float(np.clip((math.log2(max(odds, 1e-6)) / 3.0) * min(1.0, frac * 2.0), 0.0, 1.0))
            if risk > best_risk:
                best_risk, best_name, best_or = risk, name, float(odds)
        out_rows.append({fragment_col: str(frag), "partitioning_risk": best_risk, "top_partitioning_signature": best_name, "partitioning_odds_ratio": best_or})
    return pd.DataFrame(out_rows)


def score_interactions(
    interactions: pd.DataFrame,
    protein_annotations: pd.DataFrame | None = None,
    config: TriageConfig | None = None,
    column_overrides: dict[str, str | None] | None = None,
) -> pd.DataFrame:
    """Score fragment-protein measurements and classify direct-hit priority.

    Parameters
    ----------
    interactions:
        One row per fragment-protein measurement.
    protein_annotations:
        Optional table keyed by protein with columns such as compartment, family,
        known_labeling_bias, abundance.
    config:
        Thresholds for hit calling and validation support.
    column_overrides:
        Optional mapping from canonical keys to actual column names.
    """
    config = config or TriageConfig()
    df = interactions.copy()
    cols = infer_columns(df, column_overrides)

    frag_col = cols.get("fragment_id")
    prot_col = cols.get("protein")
    if frag_col is None or prot_col is None:
        raise ValueError("Input must contain fragment and protein columns. Typical names: fragment_id and protein.")

    # Merge protein annotations if supplied.
    if protein_annotations is not None and len(protein_annotations) > 0:
        ann = protein_annotations.copy()
        ann_cols = infer_columns(ann)
        ann_prot = ann_cols.get("protein")
        if ann_prot is None:
            raise ValueError("Protein annotation table must contain a protein/gene/protein_id column.")
        suffix_cols = [c for c in ann.columns if c != ann_prot]
        ann = ann[[ann_prot] + suffix_cols].drop_duplicates(ann_prot)
        df = df.merge(ann, how="left", left_on=prot_col, right_on=ann_prot, suffixes=("", "_ann"))
        # Re-infer after merge; annotations can supply missing columns.
        cols = infer_columns(df, column_overrides)

    smiles_col = cols.get("smiles")
    df = add_rdkit_descriptors(df, smiles_col)
    cols = infer_columns(df, column_overrides)

    df[frag_col] = df[frag_col].astype(str)
    df[prot_col] = df[prot_col].astype(str)
    log2fc = _num(df, cols.get("log2fc"), default=0.0).fillna(0.0)
    df["log2fc_numeric"] = log2fc
    p_col = cols.get("adj_pvalue") or cols.get("pvalue")
    pvalue = _num(df, p_col)
    df["pvalue_numeric"] = pvalue
    df["is_hit"] = infer_hits(df, cols, config)

    # Aggregated fragment/protein counts and enrichment sums.
    positive_enrichment = log2fc.clip(lower=0.0) * df["is_hit"]
    df["positive_enrichment"] = positive_enrichment

    frag_stats = df.groupby(frag_col).agg(
        fragment_measured_count=(prot_col, "nunique"),
        fragment_hit_count=("is_hit", "sum"),
        fragment_enrichment_sum=("positive_enrichment", "sum"),
    ).reset_index()
    frag_stats["fragment_hit_ratio"] = frag_stats["fragment_hit_count"] / frag_stats["fragment_measured_count"].clip(lower=1)

    prot_stats = df.groupby(prot_col).agg(
        protein_measured_count=(frag_col, "nunique"),
        protein_hit_count=("is_hit", "sum"),
        protein_enrichment_sum=("positive_enrichment", "sum"),
    ).reset_index()
    prot_stats["protein_hit_ratio"] = prot_stats["protein_hit_count"] / prot_stats["protein_measured_count"].clip(lower=1)

    df = df.merge(frag_stats, on=frag_col, how="left").merge(prot_stats, on=prot_col, how="left")

    # iScore: strong enrichment divided by fragment/protein promiscuity penalties.
    global_median = float(np.nanmedian(log2fc)) if log2fc.notna().sum() else 0.0
    frag_penalty_rank = _rank01(np.log1p(df["fragment_enrichment_sum"]))
    protein_penalty_rank = _rank01(np.log1p(df["protein_enrichment_sum"]))
    median_corrected = (df["log2fc_numeric"] - global_median).clip(lower=0.0)
    df["iscore"] = median_corrected / np.sqrt((1.0 + frag_penalty_rank) * (1.0 + protein_penalty_rank))
    df["iscore_percentile"] = _rank01(df["iscore"])

    # Signal quality.
    log2fc_score = pd.Series(_clip01((df["log2fc_numeric"] - 1.0) / 3.5), index=df.index)
    if pvalue.notna().sum() > 0:
        p_score = pd.Series(_clip01((-np.log10(pvalue.clip(lower=1e-300))) / 5.0), index=df.index)
    else:
        p_score = pd.Series(np.where(df["is_hit"] == 1, 0.65, np.nan), index=df.index)

    rep_parts: list[tuple[pd.Series, float]] = []
    rep_corr = _num(df, cols.get("replicate_corr"))
    if rep_corr.notna().sum() > 0:
        rep_parts.append((pd.Series(_clip01((rep_corr - 0.3) / 0.6), index=df.index), 1.0))
    rep_cv = _num(df, cols.get("replicate_cv"))
    if rep_cv.notna().sum() > 0:
        rep_parts.append((pd.Series(_clip01(1.0 - rep_cv / 0.5), index=df.index), 1.0))
    rep_sd = _num(df, cols.get("replicate_sd"))
    if rep_sd.notna().sum() > 0:
        rep_parts.append((pd.Series(_clip01(1.0 - rep_sd / 1.0), index=df.index), 1.0))
    rep_score = _weighted_mean(rep_parts) if rep_parts else pd.Series(np.nan, index=df.index)

    intensity = _num(df, cols.get("intensity"))
    intensity_score = _rank01(np.log10(intensity.clip(lower=1.0))) if intensity.notna().sum() > 0 else pd.Series(np.nan, index=df.index)

    control_cols = [cols.get("crf_log2fc"), cols.get("no_uv_log2fc"), cols.get("bead_log2fc")]
    control_values = pd.concat([_num(df, c) for c in control_cols if c is not None], axis=1)
    if control_values.shape[1] > 0 and control_values.notna().sum().sum() > 0:
        max_control = control_values.max(axis=1, skipna=True)
        control_score = pd.Series(_clip01(1.0 - max_control.fillna(0.0).clip(lower=0) / 2.0), index=df.index)
    else:
        control_score = pd.Series(np.nan, index=df.index)

    df["signal_quality_score"] = _weighted_mean([
        (log2fc_score, 0.35),
        (p_score, 0.25),
        (rep_score, 0.15),
        (intensity_score, 0.10),
        (control_score, 0.15),
    ])

        # Fragment-level risk.
    # Ensure hit-count columns exist.
    if "fragment_hit_count" not in df.columns:
        if "ligHits" in df.columns:
            df["fragment_hit_count"] = pd.to_numeric(df["ligHits"], errors="coerce").fillna(0)
        elif "ligand_hit_count" in df.columns:
            df["fragment_hit_count"] = pd.to_numeric(df["ligand_hit_count"], errors="coerce").fillna(0)
        elif "fragment_id" in df.columns:
            df["fragment_hit_count"] = df.groupby("fragment_id")["fragment_id"].transform("count")
        elif "fragId" in df.columns:
            df["fragment_hit_count"] = df.groupby("fragId")["fragId"].transform("count")
        else:
            df["fragment_hit_count"] = 0

    if "protein_hit_count" not in df.columns:
        if "protHits" in df.columns:
            df["protein_hit_count"] = pd.to_numeric(df["protHits"], errors="coerce").fillna(0)
        elif "target_hit_count" in df.columns:
            df["protein_hit_count"] = pd.to_numeric(df["target_hit_count"], errors="coerce").fillna(0)
        elif "protein" in df.columns:
            df["protein_hit_count"] = df.groupby("protein")["protein"].transform("count")
        elif "geneName" in df.columns:
            df["protein_hit_count"] = df.groupby("geneName")["geneName"].transform("count")
        else:
            df["protein_hit_count"] = 0

    if "fragment_hit_ratio" not in df.columns:
        if "fragment_measured_count" in df.columns:
            denom = pd.to_numeric(df["fragment_measured_count"], errors="coerce").replace(0, np.nan)
            df["fragment_hit_ratio"] = pd.to_numeric(df["fragment_hit_count"], errors="coerce") / denom
        else:
            df["fragment_hit_ratio"] = 0
        df["fragment_hit_ratio"] = df["fragment_hit_ratio"].fillna(0)

    frag_count_risk = _rank01(np.log1p(pd.to_numeric(df["fragment_hit_count"], errors="coerce").fillna(0)))
    frag_ratio_risk = _rank01(pd.to_numeric(df["fragment_hit_ratio"], errors="coerce").fillna(0))
    pred_prom = _num(df, cols.get("fragment_promiscuity_probability"))
    frag_count_risk = _rank01(np.log1p(df["fragment_hit_count"]))
    frag_ratio_risk = _rank01(df["fragment_hit_ratio"])
    pred_prom = _num(df, cols.get("fragment_promiscuity_probability"))

    clogp = _num(df, cols.get("clogp"))
    mw = _num(df, cols.get("mw"))
    aromatic = _num(df, cols.get("aromatic_rings"))
    charge = _num(df, cols.get("formal_charge"))
    physchem_parts: list[pd.Series] = []
    if clogp.notna().sum() > 0:
        physchem_parts.append(pd.Series(_clip01((clogp - 2.5) / 2.5), index=df.index))
    if mw.notna().sum() > 0:
        physchem_parts.append(pd.Series(_clip01((mw - 300.0) / 200.0), index=df.index))
    if aromatic.notna().sum() > 0:
        physchem_parts.append(pd.Series(_clip01((aromatic - 2.0) / 3.0), index=df.index))
    if charge.notna().sum() > 0:
        physchem_parts.append(pd.Series(_clip01((charge.abs() - 1.0) / 2.0), index=df.index))
    physchem_risk = pd.concat(physchem_parts, axis=1).mean(axis=1) if physchem_parts else pd.Series(np.nan, index=df.index)

    df["fragment_risk"] = _weighted_mean([
        (frag_count_risk, 0.35),
        (frag_ratio_risk, 0.25),
        (pred_prom, 0.25),
        (physchem_risk, 0.15),
    ])

    # Protein-level risk.
    protein_count_risk = _rank01(np.log1p(df["protein_hit_count"]))
    protein_ratio_risk = _rank01(df["protein_hit_ratio"])
    known_bias_col = cols.get("known_labeling_bias")
    if known_bias_col is not None:
        known_bias = parse_bool_series(df[known_bias_col]).fillna(0.0)
    else:
        known_bias = df[prot_col].str.upper().isin(DEFAULT_LABELING_BIAS_PROTEINS).astype(float)

    compartment = _text(df, cols.get("compartment"))
    sticky_compartment = compartment.apply(lambda x: 1.0 if _contains_any(x, ["membrane", "mitochond", "lysosome", "er", "endoplasmic", "golgi"]) else 0.0)
    abundance_risk = _rank01(np.log10(intensity.clip(lower=1.0))) if intensity.notna().sum() > 0 else pd.Series(np.nan, index=df.index)

    df["protein_risk"] = _weighted_mean([
        (protein_count_risk, 0.35),
        (protein_ratio_risk, 0.20),
        (known_bias, 0.25),
        (sticky_compartment, 0.10),
        (abundance_risk, 0.10),
    ])

    # Interactome-level partitioning risk, mapped from compartment enrichment among fragment hits.
    part = _compute_partitioning(df, frag_col, "is_hit", cols.get("compartment"))
    df = df.merge(part, on=frag_col, how="left")
    df["partitioning_risk"] = df["partitioning_risk"].fillna(0.0)
    df["top_partitioning_signature"] = df["top_partitioning_signature"].fillna("none")

    # Optional validation evidence.
    lower_dose_log2fc = _num(df, cols.get("lower_dose_log2fc"))
    lower_dose_hit_col = cols.get("lower_dose_hit")
    lower_by_log2fc = (lower_dose_log2fc >= config.lower_dose_log2fc_threshold).astype(float) if lower_dose_log2fc.notna().sum() > 0 else pd.Series(np.nan, index=df.index)
    lower_by_bool = parse_bool_series(df[lower_dose_hit_col]) if lower_dose_hit_col is not None else pd.Series(np.nan, index=df.index)
    lower_supported = pd.concat([lower_by_log2fc, lower_by_bool], axis=1).max(axis=1, skipna=True)

    comp_fc = _num(df, cols.get("competition_fc"))
    comp_by_fc = (comp_fc <= config.competition_fc_threshold).astype(float) if comp_fc.notna().sum() > 0 else pd.Series(np.nan, index=df.index)
    comp_bool_col = cols.get("competition_supported")
    comp_by_bool = parse_bool_series(df[comp_bool_col]) if comp_bool_col is not None else pd.Series(np.nan, index=df.index)
    comp_supported = pd.concat([comp_by_fc, comp_by_bool], axis=1).max(axis=1, skipna=True)

    orth_col = cols.get("orthogonal_supported")
    orth_supported = parse_bool_series(df[orth_col]) if orth_col is not None else pd.Series(np.nan, index=df.index)
    sar_col = cols.get("sar_supported")
    sar_supported = parse_bool_series(df[sar_col]) if sar_col is not None else pd.Series(np.nan, index=df.index)

    df["lower_dose_supported"] = lower_supported.fillna(0.0)
    df["competition_supported"] = comp_supported.fillna(0.0)
    df["orthogonal_supported"] = orth_supported.fillna(0.0)
    df["sar_supported"] = sar_supported.fillna(0.0)

    validation_matrix = pd.concat([lower_supported, comp_supported, orth_supported, sar_supported], axis=1)
    validation_present = validation_matrix.notna().any(axis=1)
    validation_bonus = validation_matrix.fillna(0.0).mean(axis=1)
    df["validation_bonus"] = np.where(validation_present, validation_bonus, np.nan)

    # Screen-only and validation-aware scores.
    screen_score = 100.0 * (
        0.28 * df["signal_quality_score"]
        + 0.28 * df["iscore_percentile"]
        + 0.18 * (1.0 - df["fragment_risk"])
        + 0.16 * (1.0 - df["protein_risk"])
        + 0.10 * (1.0 - df["partitioning_risk"])
    )
    df["screen_priority_score"] = screen_score.clip(0, 100)
    has_validation = df["validation_bonus"].notna()
    df["direct_hit_priority_score"] = np.where(
        has_validation,
        0.85 * df["screen_priority_score"] + 15.0 * df["validation_bonus"],
        df["screen_priority_score"],
    )
    df["direct_hit_priority_score"] = df["direct_hit_priority_score"].clip(0, 100)

    def classify(row: pd.Series) -> str:
        if row["is_hit"] < 1:
            return "not_primary_hit"
        if row["signal_quality_score"] < 0.40:
            return "low_confidence_signal"
        if row["partitioning_risk"] >= 0.75 and row["direct_hit_priority_score"] < 75:
            return "likely_partitioning_or_compartment_signal"
        if row["fragment_risk"] >= 0.80 and row["protein_risk"] >= 0.65:
            return "likely_promiscuity_or_labeling_bias"
        if row["direct_hit_priority_score"] >= config.high_priority_threshold and row["iscore_percentile"] >= 0.65:
            return "high_priority_direct_hit"
        if row["direct_hit_priority_score"] >= config.medium_priority_threshold:
            return "medium_priority_validate"
        return "low_priority_or_indirect"

    df["triage_class"] = df.apply(classify, axis=1)
    df["explanation"] = df.apply(build_reason, axis=1)
    df["recommended_next_step"] = df.apply(build_recommendation, axis=1)

    # Optional family/actionable term flag.
    family = _text(df, cols.get("family"))
    df["actionable_family_flag"] = family.apply(lambda x: int(_contains_any(x, FAMILY_ACTIONABLE_TERMS)))

    # Friendly canonical output first, then all original columns.
    canonical = {
        "fragment_id": frag_col,
        "protein": prot_col,
        "smiles": cols.get("smiles"),
        "compartment": cols.get("compartment"),
        "family": cols.get("family"),
    }
    for out_name, source_col in canonical.items():
        if source_col is not None and source_col in df.columns and out_name not in df.columns:
            df[out_name] = df[source_col]

    preferred = [
        "fragment_id", "protein", "smiles", "log2fc_numeric", "pvalue_numeric", "is_hit",
        "direct_hit_priority_score", "screen_priority_score", "triage_class", "iscore", "iscore_percentile",
        "signal_quality_score", "fragment_risk", "protein_risk", "partitioning_risk", "top_partitioning_signature",
        "fragment_hit_count", "fragment_hit_ratio", "protein_hit_count", "protein_hit_ratio",
        "lower_dose_supported", "competition_supported", "orthogonal_supported", "sar_supported", "validation_bonus",
        "compartment", "family", "actionable_family_flag", "explanation", "recommended_next_step",
    ]
    preferred = [c for c in preferred if c in df.columns]
    rest = [c for c in df.columns if c not in preferred]
    return df[preferred + rest]
