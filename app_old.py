from __future__ import annotations

import io
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from fragment_hit_triage.io import read_table
from fragment_hit_triage.ml import load_model, predict_with_model
from fragment_hit_triage.scoring import TriageConfig, score_interactions


st.set_page_config(page_title="Fragment Hit Triage", layout="wide")
st.title("Fragment–Protein Direct Hit Triage")
st.caption("Rank signals as direct-hit candidates versus promiscuity, labeling bias, or compartment-partitioning artifacts.")

with st.sidebar:
    st.header("Inputs")
    interactions_file = st.file_uploader("Interactions CSV/TSV", type=["csv", "tsv", "txt"])
    ann_file = st.file_uploader("Optional protein annotations", type=["csv", "tsv", "txt"])
    model_file = st.file_uploader("Optional trained ML model (.joblib)", type=["joblib"])
    st.header("Thresholds")
    log2fc_thr = st.number_input("Primary log2FC threshold", value=2.3, step=0.1)
    p_thr = st.number_input("p/adj-p threshold", value=0.05, step=0.01, format="%.4f")
    high_thr = st.number_input("High priority score", value=70.0, step=1.0)
    med_thr = st.number_input("Medium priority score", value=50.0, step=1.0)


def _read_uploaded(uploaded) -> pd.DataFrame:
    name = uploaded.name.lower()
    data = uploaded.getvalue()
    if name.endswith(".tsv") or name.endswith(".txt"):
        return pd.read_csv(io.BytesIO(data), sep="\t")
    return pd.read_csv(io.BytesIO(data), sep=None, engine="python")


if interactions_file is None:
    st.info("Upload an interactions table, or use the included sample from the command line: `python score.py --interactions data/sample_interactions.csv --out outputs/sample_scores.csv`.")
    st.stop()

interactions = _read_uploaded(interactions_file)
annotations = _read_uploaded(ann_file) if ann_file is not None else None
config = TriageConfig(
    log2fc_threshold=log2fc_thr,
    pvalue_threshold=p_thr,
    high_priority_threshold=high_thr,
    medium_priority_threshold=med_thr,
)

try:
    scored = score_interactions(interactions, protein_annotations=annotations, config=config)
    if model_file is not None:
        # Streamlit file uploader is not a filesystem path; save temporarily.
        tmp_path = ROOT / ".tmp_uploaded_model.joblib"
        tmp_path.write_bytes(model_file.getvalue())
        scored = predict_with_model(scored, load_model(tmp_path))
        tmp_path.unlink(missing_ok=True)
except Exception as e:
    st.error(str(e))
    st.stop()

st.success(f"Scored {len(scored):,} fragment-protein rows.")

metric_cols = st.columns(4)
metric_cols[0].metric("High priority", int((scored["triage_class"] == "high_priority_direct_hit").sum()))
metric_cols[1].metric("Medium priority", int((scored["triage_class"] == "medium_priority_validate").sum()))
metric_cols[2].metric("Partitioning risk", int((scored["triage_class"] == "likely_partitioning_or_compartment_signal").sum()))
metric_cols[3].metric("Promiscuity/bias risk", int((scored["triage_class"] == "likely_promiscuity_or_labeling_bias").sum()))

left, right = st.columns([2, 1])
with left:
    st.subheader("Ranked signals")
    default_cols = [
        "fragment_id", "protein", "direct_hit_priority_score", "triage_class", "log2fc_numeric",
        "iscore_percentile", "fragment_risk", "protein_risk", "partitioning_risk",
        "top_partitioning_signature", "recommended_next_step",
    ]
    if "ml_direct_hit_probability" in scored.columns:
        default_cols.insert(3, "ml_direct_hit_probability")
        default_cols.insert(4, "hybrid_priority_score")
    st.dataframe(scored.sort_values("direct_hit_priority_score", ascending=False)[[c for c in default_cols if c in scored.columns]], use_container_width=True)

with right:
    st.subheader("Class distribution")
    st.bar_chart(scored["triage_class"].value_counts())
    st.subheader("Score distribution")
    st.bar_chart(scored["direct_hit_priority_score"].round(-1).value_counts().sort_index())

st.subheader("Full output")
st.dataframe(scored, use_container_width=True)

csv = scored.to_csv(index=False).encode("utf-8")
st.download_button("Download scored CSV", data=csv, file_name="fragment_hit_triage_scores.csv", mime="text/csv")
