from __future__ import annotations

import io
import sys
from pathlib import Path
from typing import Iterable

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from fragment_hit_triage.ml import load_model, predict_with_model
from fragment_hit_triage.scoring import TriageConfig, infer_columns, score_interactions


st.set_page_config(page_title="Fragment Hit Triage", layout="wide")
st.title("Fragment–Protein Direct Hit Triage")
st.caption("Stable large-file viewer/scorer for fragment–protein direct-hit prioritization.")


# -----------------------------
# Helpers
# -----------------------------
@st.cache_data(show_spinner=False, max_entries=4)
def read_uploaded_table(file_name: str, file_bytes: bytes) -> pd.DataFrame:
    """Read uploaded CSV/TSV once and cache it by file name + bytes."""
    name = file_name.lower()
    if name.endswith(".tsv") or name.endswith(".txt"):
        return pd.read_csv(io.BytesIO(file_bytes), sep="\t")
    return pd.read_csv(io.BytesIO(file_bytes), sep=None, engine="python")


def read_uploaded(uploaded) -> pd.DataFrame:
    return read_uploaded_table(uploaded.name, uploaded.getvalue())


def is_scored_table(df: pd.DataFrame) -> bool:
    return "direct_hit_priority_score" in df.columns and "triage_class" in df.columns


def choose_score_column(df: pd.DataFrame) -> str:
    for col in ["hybrid_priority_score", "direct_hit_priority_score", "screen_priority_score"]:
        if col in df.columns:
            return col
    return df.columns[0]


def choose_probability_column(df: pd.DataFrame) -> str | None:
    for col in ["ml_direct_hit_probability", "direct_hit_probability", "ml_probability"]:
        if col in df.columns:
            return col
    return None


def ensure_optional_control_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Avoid scoring crash when control columns are absent."""
    out = df.copy()
    for col in ["crf_log2fc", "no_uv_log2fc", "bead_log2fc"]:
        if col not in out.columns:
            out[col] = 0.0
    return out


def prefilter_for_web(df: pd.DataFrame, max_rows: int) -> pd.DataFrame:
    """Keep a manageable subset for Streamlit.

    This is an approximate web-mode subset. Use command-line scoring for exact global
    fragment/protein risk because risk metrics depend on all rows.
    """
    if len(df) <= max_rows:
        return df

    cols = infer_columns(df)
    out = df.copy()
    priority = pd.Series(0.0, index=out.index)

    # Prefer explicit hit/mdf columns if present.
    hit_col = cols.get("hit")
    if hit_col and hit_col in out.columns:
        hit_num = pd.to_numeric(out[hit_col], errors="coerce")
        if hit_num.notna().sum() > 0:
            priority += hit_num.fillna(0).astype(float) * 100.0
        else:
            priority += out[hit_col].astype(str).str.lower().isin(["1", "true", "hit", "enriched", "yes"]).astype(float) * 100.0

    log2_col = cols.get("log2fc")
    if log2_col and log2_col in out.columns:
        priority += pd.to_numeric(out[log2_col], errors="coerce").fillna(0.0).clip(lower=0.0)

    p_col = cols.get("adj_pvalue") or cols.get("pvalue")
    if p_col and p_col in out.columns:
        p = pd.to_numeric(out[p_col], errors="coerce")
        priority += (-p.clip(lower=1e-300).map(lambda x: pd.NA if pd.isna(x) else __import__("math").log10(x))).fillna(0.0).clip(lower=0.0)

    out["_web_prefilter_priority"] = priority
    out = out.sort_values("_web_prefilter_priority", ascending=False).head(max_rows).drop(columns=["_web_prefilter_priority"])
    return out


def top_for_display(df: pd.DataFrame, sort_col: str, n: int, classes: list[str] | None, min_prob: float | None) -> pd.DataFrame:
    out = df.copy()
    if classes and "triage_class" in out.columns:
        out = out[out["triage_class"].isin(classes)]
    prob_col = choose_probability_column(out)
    if min_prob is not None and prob_col is not None:
        out = out[pd.to_numeric(out[prob_col], errors="coerce").fillna(0.0) >= min_prob]
    if sort_col in out.columns:
        out = out.sort_values(sort_col, ascending=False)
    return out.head(n)


def compact_columns(df: pd.DataFrame) -> list[str]:
    prob_col = choose_probability_column(df)
    score_col = choose_score_column(df)
    wanted = [
        "fragment_id", "protein", "smiles",
        prob_col, score_col,
        "direct_hit_priority_score", "triage_class", "is_hit",
        "log2fc_numeric", "pvalue_numeric", "iscore_percentile", "iscore",
        "fragment_risk", "protein_risk", "partitioning_risk",
        "fragment_hit_count", "protein_hit_count",
        "top_partitioning_signature", "recommended_next_step", "explanation",
    ]
    return [c for c in wanted if c is not None and c in df.columns]


# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.header("Inputs")
    interactions_file = st.file_uploader("Interactions CSV/TSV or scored predictions", type=["csv", "tsv", "txt"])
    ann_file = st.file_uploader("Optional protein annotations", type=["csv", "tsv", "txt"])
    model_file = st.file_uploader("Optional trained ML model (.joblib)", type=["joblib"])

    st.header("Stable mode")
    st.write("For million-row files, do not display or rescore everything in the browser.")
    fast_mode = st.checkbox("Fast web mode: prefilter before scoring", value=True)
    max_process_rows = st.number_input("Max rows to score in app", min_value=1000, max_value=2_000_000, value=1_400_000, step=10_000)
    max_display_rows = st.number_input("Max rows to display", min_value=50, max_value=1_400_000, value=2000, step=500)

    st.header("Thresholds")
    log2fc_thr = st.number_input("Primary log2FC threshold", value=2.3, step=0.1)
    p_thr = st.number_input("p/adj-p threshold", value=0.05, step=0.01, format="%.4f")
    high_thr = st.number_input("High priority score", value=70.0, step=1.0)
    med_thr = st.number_input("Medium priority score", value=50.0, step=1.0)

    st.header("Display filters")
    min_ml_prob = st.slider("Minimum ML probability shown", 0.0, 1.0, 0.0, 0.05)
    run_button = st.button("Run / Refresh", type="primary")
    if st.button("Clear Streamlit cache"):
        st.cache_data.clear()
        st.rerun()


if interactions_file is None:
    st.info("Upload `finalScreen_for_triage.csv`, `finalScreen_ml_predictions_with_smiles.csv`, or a smaller top-hit CSV.")
    st.stop()

# Read input once.
with st.spinner("Reading uploaded table..."):
    interactions = read_uploaded(interactions_file)

st.write(f"Loaded **{len(interactions):,} rows** and **{len(interactions.columns):,} columns** from `{interactions_file.name}`.")

already_scored = is_scored_table(interactions)
if already_scored:
    st.info("This file already contains scoring columns, so the app will skip rule-based scoring. This is the most stable mode for large files.")
elif not run_button:
    st.warning("Click **Run / Refresh** in the sidebar to score this table. For the full 1.34M-row official file, command-line scoring is recommended.")
    st.stop()

# Score / predict.
if already_scored:
    scored = interactions.copy()
else:
    raw_count = len(interactions)
    to_score = interactions.copy()
    if fast_mode and raw_count > int(max_process_rows):
        to_score = prefilter_for_web(to_score, int(max_process_rows))
        st.warning(
            f"Fast web mode kept {len(to_score):,} of {raw_count:,} rows before scoring. "
            "Scores are suitable for browsing top candidates, but exact global risk ranking should be generated by command line."
        )

    to_score = ensure_optional_control_columns(to_score)
    annotations = read_uploaded(ann_file) if ann_file is not None else None
    config = TriageConfig(
        log2fc_threshold=log2fc_thr,
        pvalue_threshold=p_thr,
        high_priority_threshold=high_thr,
        medium_priority_threshold=med_thr,
    )
    with st.spinner("Scoring fragment-protein rows..."):
        scored = score_interactions(to_score, protein_annotations=annotations, config=config)

# Apply model only when needed.
if model_file is not None and choose_probability_column(scored) is None:
    tmp_path = ROOT / ".tmp_uploaded_model.joblib"
    try:
        tmp_path.write_bytes(model_file.getvalue())
        with st.spinner("Running ML prediction..."):
            scored = predict_with_model(scored, load_model(tmp_path))
    finally:
        tmp_path.unlink(missing_ok=True)
elif model_file is not None:
    st.info("ML probability columns already exist in the uploaded file, so the uploaded model was not reapplied.")

st.success(f"Ready: {len(scored):,} rows available in app.")

# Metrics.
metric_cols = st.columns(5)
metric_cols[0].metric("Rows", f"{len(scored):,}")
if "triage_class" in scored.columns:
    metric_cols[1].metric("High priority", int((scored["triage_class"] == "high_priority_direct_hit").sum()))
    metric_cols[2].metric("Medium priority", int((scored["triage_class"] == "medium_priority_validate").sum()))
    metric_cols[3].metric("Partitioning risk", int((scored["triage_class"] == "likely_partitioning_or_compartment_signal").sum()))
    metric_cols[4].metric("Promiscuity/bias", int((scored["triage_class"] == "likely_promiscuity_or_labeling_bias").sum()))

score_col = choose_score_column(scored)
prob_col = choose_probability_column(scored)

# User can choose class filters after seeing available classes.
class_filter = None
if "triage_class" in scored.columns:
    available_classes = sorted(scored["triage_class"].dropna().astype(str).unique().tolist())
    default_classes = [c for c in ["high_priority_direct_hit", "medium_priority_validate"] if c in available_classes]
    class_filter = st.multiselect("Classes to display", options=available_classes, default=default_classes or available_classes)

sort_options = [c for c in [prob_col, "hybrid_priority_score", "direct_hit_priority_score", "screen_priority_score", "iscore_percentile", "log2fc_numeric"] if c and c in scored.columns]
if not sort_options:
    sort_options = scored.columns.tolist()
sort_col = st.selectbox("Sort displayed table by", sort_options, index=0)

display_df = top_for_display(scored, sort_col=sort_col, n=int(max_display_rows), classes=class_filter, min_prob=min_ml_prob if prob_col else None)

left, right = st.columns([2, 1])
with left:
    st.subheader(f"Top displayed rows: {len(display_df):,}")
    cols = compact_columns(display_df)
    display_df = display_df.loc[:, ~display_df.columns.duplicated()].copy()
    cols = [c for c in cols if c in display_df.columns]
    cols = list(dict.fromkeys(cols))
    view_df = display_df[cols] if cols else display_df
    st.dataframe(view_df, use_container_width=True, height=620)

with right:
    if "triage_class" in scored.columns:
        st.subheader("Class distribution")
        st.bar_chart(scored["triage_class"].value_counts())
    if score_col in scored.columns:
        st.subheader("Score distribution")
        st.bar_chart(pd.to_numeric(scored[score_col], errors="coerce").round(-1).value_counts().sort_index())

st.subheader("Downloads")
st.download_button(
    "Download displayed rows CSV",
    data=display_df.to_csv(index=False).encode("utf-8"),
    file_name="fragment_hit_triage_displayed_rows.csv",
    mime="text/csv",
)

if len(scored) <= 200_000:
    st.download_button(
        "Download all rows currently loaded in app CSV",
        data=scored.to_csv(index=False).encode("utf-8"),
        file_name="fragment_hit_triage_scores.csv",
        mime="text/csv",
    )
else:
    st.info("Full CSV download is disabled above 200,000 rows to keep the browser stable. Use command-line output files for full results.")
