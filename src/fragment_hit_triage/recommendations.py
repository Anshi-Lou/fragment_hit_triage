from __future__ import annotations

import pandas as pd


def _missing_or_false(row: pd.Series, col: str) -> bool:
    return col not in row.index or pd.isna(row[col]) or row[col] in {0, 0.0, False, "", "no", "No", "false", "False"}


def build_recommendation(row: pd.Series) -> str:
    """Suggest next validation step based on current evidence and failure mode."""
    if row.get("is_hit", 0) < 1:
        return "Do not prioritize until enrichment passes the primary hit threshold."

    if row.get("signal_quality_score", 0) < 0.45:
        return "Repeat the pulldown with duplicates and stronger controls; current signal quality is weak."

    if row.get("partitioning_risk", 0) >= 0.70:
        return "Check cellular partitioning first: confocal localization plus lower-dose repeat before SAR work."

    if row.get("protein_risk", 0) >= 0.75:
        return "Protein looks frequent-hitter/background-prone; test recombinant protein labeling or orthogonal biophysics."

    if row.get("fragment_risk", 0) >= 0.75:
        return "Fragment looks promiscuous; run 10–20x free analog competition and compare close analog SAR."

    missing_lower = _missing_or_false(row, "lower_dose_supported")
    missing_comp = _missing_or_false(row, "competition_supported")
    missing_orth = _missing_or_false(row, "orthogonal_supported")
    missing_sar = _missing_or_false(row, "sar_supported")

    if missing_lower:
        return "Next: lower-dose repeat, e.g. retest at half concentration while preserving enrichment."
    if missing_comp:
        return "Next: 10–20x non-clickable/elaborated analog competition against the FFF pulldown."
    if missing_orth:
        return "Next: orthogonal validation such as recombinant labeling, SPR/MST/DSF/NMR, CETSA/NanoBRET, or function."
    if missing_sar:
        return "Next: analog SAR series; true hits should show coherent changes rather than all hydrophobic analogs competing."
    return "Validation package is strong; move to fragment elaboration and target-specific functional follow-up."


def build_reason(row: pd.Series) -> str:
    positives: list[str] = []
    risks: list[str] = []

    if row.get("signal_quality_score", 0) >= 0.70:
        positives.append("strong primary signal")
    if row.get("iscore_percentile", 0) >= 0.80:
        positives.append("high iScore percentile")
    if row.get("fragment_risk", 1) <= 0.35:
        positives.append("fragment is relatively selective")
    if row.get("protein_risk", 1) <= 0.35:
        positives.append("protein is not a frequent hitter")
    if row.get("validation_bonus", 0) >= 0.60:
        positives.append("validation evidence present")

    if row.get("fragment_risk", 0) >= 0.70:
        risks.append("fragment promiscuity risk")
    if row.get("protein_risk", 0) >= 0.70:
        risks.append("protein frequent-hitter/background risk")
    if row.get("partitioning_risk", 0) >= 0.70:
        sig = row.get("top_partitioning_signature", "compartment")
        risks.append(f"possible {sig} partitioning")
    if row.get("signal_quality_score", 1) < 0.45:
        risks.append("weak or poorly controlled signal")

    pos_text = "; ".join(positives[:4]) if positives else "no strong positive evidence yet"
    risk_text = "; ".join(risks[:4]) if risks else "no dominant risk flag"
    return f"Positive evidence: {pos_text}. Risk flags: {risk_text}."
