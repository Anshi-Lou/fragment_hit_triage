import pandas as pd
from pathlib import Path

inp = Path("outputs/finalScreen_ml_predictions_with_smiles.csv")
out = Path("outputs/top_ml_direct_hits.csv")

df = pd.read_csv(inp)

prob_col = None
for c in ["ml_direct_hit_probability", "direct_hit_probability", "ml_probability"]:
    if c in df.columns:
        prob_col = c
        break

if prob_col is None:
    raise ValueError("No ML probability column found. Columns are: " + str(df.columns.tolist()))

score_col = "hybrid_priority_score" if "hybrid_priority_score" in df.columns else "direct_hit_priority_score"

print("Rows:", len(df))
print("Probability column:", prob_col)
print("Score column:", score_col)

if "triage_class" in df.columns:
    print("\nTriage class counts:")
    print(df["triage_class"].value_counts())

    hits = df[
        (df["triage_class"] == "high_priority_direct_hit")
        & (df[prob_col] >= 0.7)
    ].copy()
else:
    hits = df[df[prob_col] >= 0.7].copy()

# If threshold is too strict, loosen it.
if len(hits) == 0:
    print("\nNo hits with probability >= 0.7. Trying >= 0.5.")
    if "triage_class" in df.columns:
        hits = df[
            (df["triage_class"] == "high_priority_direct_hit")
            & (df[prob_col] >= 0.5)
        ].copy()
    else:
        hits = df[df[prob_col] >= 0.5].copy()

# If still empty, just export top 500 by ML probability and score.
if len(hits) == 0:
    print("\nStill no strict high-priority hits. Exporting top 500 ranked predictions.")
    hits = df.sort_values([prob_col, score_col], ascending=False).head(500).copy()
else:
    hits = hits.sort_values([prob_col, score_col], ascending=False)

wanted_cols = [
    "fragment_id", "protein", "smiles",
    prob_col, score_col,
    "direct_hit_priority_score", "triage_class",
    "iscore", "iscore_percentile",
    "fragment_risk", "protein_risk", "partitioning_risk",
    "log2FC", "pvalue", "adj_pvalue",
    "recommended_next_step"
]

cols = [c for c in wanted_cols if c in hits.columns]
hits[cols].to_csv(out, index=False)

print("\nSaved:", len(hits), "rows to", out)
print(hits[cols].head(30).to_string(index=False))
