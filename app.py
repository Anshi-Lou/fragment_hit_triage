from __future__ import annotations

import io
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from fragment_hit_triage.ml import load_model, predict_with_model
from fragment_hit_triage.scoring import TriageConfig, infer_columns, score_interactions


DEFAULT_MODEL_PATH = (
    ROOT
    / "models"
    / "official_direct_hit_with_smiles"
    / "direct_hit_model.joblib"
)


st.set_page_config(page_title="Fragment Hit Triage", layout="wide")
st.title("Fragment–Protein Direct Hit Triage")
st.caption(
    "Stable large-file viewer/scorer with an automatically loaded built-in "
    "direct-hit prediction model."
)


# -----------------------------
# Cached loaders
# -----------------------------
@st.cache_data(show_spinner=False, max_entries=4)
def read_uploaded_table(file_name: str, file_bytes: bytes) -> pd.DataFrame:
    """Read an uploaded CSV/TSV once and cache it by name and content."""
    name = file_name.lower()
    if name.endswith(".tsv") or name.endswith(".txt"):
        return pd.read_csv(io.BytesIO(file_bytes), sep="\t")
    return pd.read_csv(io.BytesIO(file_bytes), sep=None, engine="python")


@st.cache_resource(show_spinner=False)
def load_default_model() -> dict:
    """Load the model bundled with the repository once per app process."""
    if not DEFAULT_MODEL_PATH.is_file():
        raise FileNotFoundError(
            "Built-in model not found. Expected file:\n"
            f"{DEFAULT_MODEL_PATH}"
        )
    return load_model(DEFAULT_MODEL_PATH)


# -----------------------------
# Helpers
# -----------------------------
def read_uploaded(uploaded) -> pd.DataFrame:
    return read_uploaded_table(uploaded.name, uploaded.getvalue())


def is_scored_table(df: pd.DataFrame) -> bool:
    return (
        "direct_hit_priority_score" in df.columns
        and "triage_class" in df.columns
    )


def choose_score_column(df: pd.DataFrame) -> str:
    for col in [
        "hybrid_priority_score",
        "direct_hit_priority_score",
        "screen_priority_score",
    ]:
        if col in df.columns:
            return col
    return df.columns[0]


def choose_probability_column(df: pd.DataFrame) -> str | None:
    for col in [
        "ml_direct_hit_probability",
        "direct_hit_probability",
        "ml_probability",
    ]:
        if col in df.columns:
            return col
    return None


def ensure_optional_control_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Avoid scoring failures when optional control columns are absent."""
    out = df.copy()
    for col in ["crf_log2fc", "no_uv_log2fc", "bead_log2fc"]:
        if col not in out.columns:
            out[col] = 0.0
    return out


def ensure_model_input_columns(
    df: pd.DataFrame,
    model_bundle: dict,
) -> tuple[pd.DataFrame, list[str]]:
    """Add absent model columns so sklearn preprocessing can handle them.

    Missing numeric features are represented as NaN and imputed by the fitted
    pipeline. A missing SMILES field is represented by an empty string.
    """
    out = df.copy()
    missing: list[str] = []

    for col in model_bundle.get("numeric_cols", []):
        if col not in out.columns:
            out[col] = float("nan")
            missing.append(col)

    smiles_col = model_bundle.get("smiles_col")
    if smiles_col and smiles_col not in out.columns:
        out[smiles_col] = ""
        missing.append(smiles_col)

    return out, missing


def prefilter_for_web(df: pd.DataFrame, max_rows: int) -> pd.DataFrame:
    """Keep a manageable subset for Streamlit.

    This is an approximate web-mode subset. Exact fragment/protein risk metrics
    depend on all rows, so command-line scoring remains preferable for exact
    full-data analysis.
    """
    if len(df) <= max_rows:
        return df

    cols = infer_columns(df)
    out = df.copy()
    priority = pd.Series(0.0, index=out.index)

    hit_col = cols.get("hit")
    if hit_col and hit_col in out.columns:
        hit_num = pd.to_numeric(out[hit_col], errors="coerce")
        if hit_num.notna().sum() > 0:
            priority += hit_num.fillna(0).astype(float) * 100.0
        else:
            priority += (
                out[hit_col]
                .astype(str)
                .str.lower()
                .isin(["1", "true", "hit", "enriched", "yes"])
                .astype(float)
                * 100.0
            )

    log2_col = cols.get("log2fc")
    if log2_col and log2_col in out.columns:
        priority += (
            pd.to_numeric(out[log2_col], errors="coerce")
            .fillna(0.0)
            .clip(lower=0.0)
        )

    p_col = cols.get("adj_pvalue") or cols.get("pvalue")
    if p_col and p_col in out.columns:
        p_values = pd.to_numeric(out[p_col], errors="coerce").clip(
            lower=1e-300
        )
        priority += (
            -p_values.map(
                lambda value: (
                    pd.NA
                    if pd.isna(value)
                    else __import__("math").log10(value)
                )
            )
        ).fillna(0.0).clip(lower=0.0)

    out["_web_prefilter_priority"] = priority
    return (
        out.sort_values("_web_prefilter_priority", ascending=False)
        .head(max_rows)
        .drop(columns=["_web_prefilter_priority"])
    )


def top_for_display(
    df: pd.DataFrame,
    sort_col: str,
    n: int,
    classes: list[str] | None,
    min_prob: float | None,
) -> pd.DataFrame:
    out = df.copy()

    if classes and "triage_class" in out.columns:
        out = out[out["triage_class"].isin(classes)]

    prob_col = choose_probability_column(out)
    if min_prob is not None and prob_col is not None:
        out = out[
            pd.to_numeric(out[prob_col], errors="coerce")
            .fillna(0.0)
            .ge(min_prob)
        ]

    if sort_col in out.columns:
        out = out.sort_values(sort_col, ascending=False)

    return out.head(n)


def compact_columns(df: pd.DataFrame) -> list[str]:
    prob_col = choose_probability_column(df)
    score_col = choose_score_column(df)

    wanted = [
        "fragment_id",
        "protein",
        "smiles",
        prob_col,
        score_col,
        "direct_hit_priority_score",
        "triage_class",
        "is_hit",
        "log2fc_numeric",
        "pvalue_numeric",
        "iscore_percentile",
        "iscore",
        "fragment_risk",
        "protein_risk",
        "partitioning_risk",
        "fragment_hit_count",
        "protein_hit_count",
        "top_partitioning_signature",
        "recommended_next_step",
        "explanation",
    ]

    return [
        col
        for col in wanted
        if col is not None and col in df.columns
    ]


# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.header("Inputs")

    interactions_file = st.file_uploader(
        "Interactions CSV/TSV or scored predictions",
        type=["csv", "tsv", "txt"],
    )
    ann_file = st.file_uploader(
        "Optional protein annotations",
        type=["csv", "tsv", "txt"],
    )

    if DEFAULT_MODEL_PATH.is_file():
        st.success("Built-in ML model ready")
        st.caption(
            "The model is loaded automatically from "
            "`models/official_direct_hit_with_smiles/`."
        )
    else:
        st.error("Built-in ML model is missing")

    st.header("Stable mode")
    st.write(
        "For million-row files, avoid displaying or rescoring every row "
        "in the browser."
    )
    fast_mode = st.checkbox(
        "Fast web mode: prefilter before scoring",
        value=True,
    )
    max_process_rows = st.number_input(
        "Max rows to score in app",
        min_value=1_000,
        max_value=2_000_000,
        value=1_400_000,
        step=10_000,
    )
    max_display_rows = st.number_input(
        "Max rows to display",
        min_value=50,
        max_value=1_400_000,
        value=2_000,
        step=500,
    )

    st.header("Thresholds")
    log2fc_thr = st.number_input(
        "Primary log2FC threshold",
        value=2.3,
        step=0.1,
    )
    p_thr = st.number_input(
        "p/adj-p threshold",
        value=0.05,
        step=0.01,
        format="%.4f",
    )
    high_thr = st.number_input(
        "High priority score",
        value=70.0,
        step=1.0,
    )
    med_thr = st.number_input(
        "Medium priority score",
        value=50.0,
        step=1.0,
    )

    st.header("Display filters")
    min_ml_prob = st.slider(
        "Minimum ML probability shown",
        0.0,
        1.0,
        0.0,
        0.05,
    )
    run_button = st.button("Run / Refresh", type="primary")

    if st.button("Clear Streamlit cache"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()


if interactions_file is None:
    st.info(
        "Upload `finalScreen_for_triage.csv`, "
        "`finalScreen_ml_predictions_with_smiles.csv`, "
        "or a smaller top-hit CSV."
    )
    st.stop()


# -----------------------------
# Read and score
# -----------------------------
with st.spinner("Reading uploaded table..."):
    interactions = read_uploaded(interactions_file)

st.write(
    f"Loaded **{len(interactions):,} rows** and "
    f"**{len(interactions.columns):,} columns** from "
    f"`{interactions_file.name}`."
)

already_scored = is_scored_table(interactions)

if already_scored:
    st.info(
        "This file already contains rule-based scoring columns, so the app "
        "will skip rule-based scoring."
    )
elif not run_button:
    st.warning(
        "Click **Run / Refresh** in the sidebar to score this table. "
        "For the full official dataset, command-line scoring is more stable."
    )
    st.stop()

if already_scored:
    scored = interactions.copy()
else:
    raw_count = len(interactions)
    to_score = interactions.copy()

    if fast_mode and raw_count > int(max_process_rows):
        to_score = prefilter_for_web(to_score, int(max_process_rows))
        st.warning(
            f"Fast web mode kept {len(to_score):,} of {raw_count:,} rows "
            "before scoring. These scores are suitable for browsing top "
            "candidates, but exact global risk ranking should be generated "
            "with the command-line workflow."
        )

    to_score = ensure_optional_control_columns(to_score)
    annotations = (
        read_uploaded(ann_file)
        if ann_file is not None
        else None
    )
    config = TriageConfig(
        log2fc_threshold=log2fc_thr,
        pvalue_threshold=p_thr,
        high_priority_threshold=high_thr,
        medium_priority_threshold=med_thr,
    )

    with st.spinner("Scoring fragment-protein rows..."):
        scored = score_interactions(
            to_score,
            protein_annotations=annotations,
            config=config,
        )


# -----------------------------
# Automatic built-in ML prediction
# -----------------------------
if choose_probability_column(scored) is None:
    try:
        with st.spinner("Loading built-in ML model..."):
            default_model = load_default_model()

        model_input, missing_model_cols = ensure_model_input_columns(
            scored,
            default_model,
        )

        if missing_model_cols:
            missing_preview = ", ".join(missing_model_cols[:8])
            suffix = "..." if len(missing_model_cols) > 8 else ""
            st.warning(
                "Some model input columns were absent and were filled with "
                f"default missing values: {missing_preview}{suffix}"
            )

        if (
            default_model.get("smiles_col")
            and default_model["smiles_col"] not in scored.columns
        ):
            st.warning(
                "The built-in model was trained with SMILES features, but "
                "the uploaded table has no SMILES column. Prediction can run, "
                "but chemical-structure information will be unavailable."
            )

        with st.spinner("Running built-in ML prediction..."):
            scored = predict_with_model(model_input, default_model)

        st.info("ML probabilities were generated with the built-in model.")

    except Exception as exc:
        st.error(f"Built-in model prediction failed: {exc}")
        st.exception(exc)
        st.stop()
else:
    st.info(
        "The uploaded table already contains an ML probability column, so "
        "the built-in model was not applied again."
    )


st.success(f"Ready: {len(scored):,} rows available in app.")


# -----------------------------
# Summary metrics and display
# -----------------------------
metric_cols = st.columns(5)
metric_cols[0].metric("Rows", f"{len(scored):,}")

if "triage_class" in scored.columns:
    metric_cols[1].metric(
        "High priority",
        int(
            (
                scored["triage_class"]
                == "high_priority_direct_hit"
            ).sum()
        ),
    )
    metric_cols[2].metric(
        "Medium priority",
        int(
            (
                scored["triage_class"]
                == "medium_priority_validate"
            ).sum()
        ),
    )
    metric_cols[3].metric(
        "Partitioning risk",
        int(
            (
                scored["triage_class"]
                == "likely_partitioning_or_compartment_signal"
            ).sum()
        ),
    )
    metric_cols[4].metric(
        "Promiscuity/bias",
        int(
            (
                scored["triage_class"]
                == "likely_promiscuity_or_labeling_bias"
            ).sum()
        ),
    )

score_col = choose_score_column(scored)
prob_col = choose_probability_column(scored)

class_filter = None
if "triage_class" in scored.columns:
    available_classes = sorted(
        scored["triage_class"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )
    default_classes = [
        value
        for value in [
            "high_priority_direct_hit",
            "medium_priority_validate",
        ]
        if value in available_classes
    ]
    class_filter = st.multiselect(
        "Classes to display",
        options=available_classes,
        default=default_classes or available_classes,
    )

sort_options = [
    col
    for col in [
        prob_col,
        "hybrid_priority_score",
        "direct_hit_priority_score",
        "screen_priority_score",
        "iscore_percentile",
        "log2fc_numeric",
    ]
    if col and col in scored.columns
]

if not sort_options:
    sort_options = scored.columns.tolist()

sort_col = st.selectbox(
    "Sort displayed table by",
    sort_options,
    index=0,
)

display_df = top_for_display(
    scored,
    sort_col=sort_col,
    n=int(max_display_rows),
    classes=class_filter,
    min_prob=min_ml_prob if prob_col else None,
)

left, right = st.columns([2, 1])

with left:
    st.subheader(f"Top displayed rows: {len(display_df):,}")
    cols = compact_columns(display_df)
    display_df = (
        display_df.loc[:, ~display_df.columns.duplicated()]
        .copy()
    )
    cols = [col for col in cols if col in display_df.columns]
    cols = list(dict.fromkeys(cols))
    view_df = display_df[cols] if cols else display_df

    st.dataframe(
        view_df,
        use_container_width=True,
        height=620,
    )

with right:
    if "triage_class" in scored.columns:
        st.subheader("Class distribution")
        st.bar_chart(scored["triage_class"].value_counts())

    if score_col in scored.columns:
        st.subheader("Score distribution")
        score_distribution = (
            pd.to_numeric(scored[score_col], errors="coerce")
            .round(-1)
            .value_counts()
            .sort_index()
        )
        st.bar_chart(score_distribution)


# -----------------------------
# Downloads
# -----------------------------
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
    st.info(
        "Full CSV download is disabled above 200,000 rows to keep the "
        "browser stable. Use command-line output files for full results."
    )
