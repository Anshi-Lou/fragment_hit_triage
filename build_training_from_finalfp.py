import numpy as np
import pandas as pd
from pathlib import Path

screen_path = Path("data/finalScreen_for_triage.csv")
fp_path = Path("data/finalFp.tsv")

out_full = Path("data/finalScreen_labeled_for_training.csv")
out_balanced = Path("data/finalScreen_training_balanced.csv")

screen = pd.read_csv(screen_path)
fp = pd.read_csv(fp_path, sep="\t")

# Make sure optional scoring-control columns exist.
for c in ["crf_log2fc", "no_uv_log2fc", "bead_log2fc"]:
    if c not in screen.columns:
        screen[c] = 0.0

# Normalize finalFp columns.
fp["fragment_id"] = fp["gen1Lig"].astype(str).str.strip()
fp["protein"] = fp["geneName"].fillna(fp["accession"]).astype(str).str.strip()

fp["competition_fc"] = pd.to_numeric(fp["l2fcM"], errors="coerce")
if fp["competition_fc"].isna().all():
    fp["competition_fc"] = pd.to_numeric(fp["l2fc"], errors="coerce")

fp["ml10adjP_num"] = pd.to_numeric(fp["ml10adjP"], errors="coerce")
fp["mdfClass_num"] = pd.to_numeric(fp["mdfClass"], errors="coerce")

# Strong negative competition = likely direct binding support.
# l2fcM <= -1 means at least ~2-fold reduction under competition.
# ml10adjP >= 1.3 means adj p <= 0.05.
fp["competition_supported"] = (
    (fp["competition_fc"] <= -1.0)
    & (fp["ml10adjP_num"] >= 1.3)
    & (fp["mdfClass_num"] >= 2)
).astype(int)

# For each original fragment-protein pair, keep strongest competition evidence.
fp_sorted = fp.sort_values(["fragment_id", "protein", "competition_fc"], ascending=[True, True, True])
best = fp_sorted.groupby(["fragment_id", "protein"], as_index=False).first()

agg = fp.groupby(["fragment_id", "protein"], as_index=False).agg(
    competition_fc=("competition_fc", "min"),
    competition_supported=("competition_supported", "max"),
    competition_tested=("competition_fc", "count"),
)

best_cols = ["fragment_id", "protein", "gen2Lig", "conc", "expId"]
best = best[best_cols].rename(columns={
    "gen2Lig": "best_competitor",
    "conc": "competition_concentration",
    "expId": "competition_exp_id",
})

competition = agg.merge(best, on=["fragment_id", "protein"], how="left")

# Merge competition evidence into finalScreen table.
df = screen.merge(
    competition,
    on=["fragment_id", "protein"],
    how="left"
)

# Define screen hit.
if "hit" in df.columns:
    is_screen_hit = pd.to_numeric(df["hit"], errors="coerce").fillna(0).astype(int).eq(1)
elif "mdfClass" in df.columns:
    is_screen_hit = pd.to_numeric(df["mdfClass"], errors="coerce").fillna(0).ge(2)
else:
    pcol = "adj_pvalue" if "adj_pvalue" in df.columns else "pvalue"
    is_screen_hit = (
        pd.to_numeric(df["log2FC"], errors="coerce").fillna(0).ge(2.3)
        & pd.to_numeric(df[pcol], errors="coerce").fillna(1).le(0.05)
    )

df["is_screen_hit_for_label"] = is_screen_hit.astype(int)
df["competition_supported"] = df["competition_supported"].fillna(0).astype(int)

# Build weak supervised label.
df["true_direct_hit"] = np.nan

# Positive: screen hit + competed in finalFp.
df.loc[
    (df["is_screen_hit_for_label"] == 1)
    & (df["competition_supported"] == 1),
    "true_direct_hit"
] = 1

# Negative type 1: assayed by competition but not competed.
df.loc[
    (df["is_screen_hit_for_label"] == 1)
    & df["competition_tested"].notna()
    & (df["competition_supported"] == 0),
    "true_direct_hit"
] = 0

# Negative type 2: clear non-hit background.
log2fc = pd.to_numeric(df["log2FC"], errors="coerce").fillna(0)
pval = pd.to_numeric(df["adj_pvalue"] if "adj_pvalue" in df.columns else df["pvalue"], errors="coerce").fillna(1)

clear_nonhit = (
    (df["is_screen_hit_for_label"] == 0)
    & (log2fc < 1.0)
    & (pval > 0.1)
)

df.loc[clear_nonhit, "true_direct_hit"] = 0

# Save full labeled table.
df.to_csv(out_full, index=False)

# Build balanced training subset so training is not dominated by millions of non-hits.
labeled = df[df["true_direct_hit"].notna()].copy()
labeled["true_direct_hit"] = labeled["true_direct_hit"].astype(int)

pos = labeled[labeled["true_direct_hit"] == 1]
neg = labeled[labeled["true_direct_hit"] == 0]

if len(pos) == 0:
    raise ValueError("No positive labels were found. Loosen thresholds or inspect finalFp.tsv.")

n_neg = min(len(neg), max(5000, 5 * len(pos)))
neg_sample = neg.sample(n=n_neg, random_state=7) if len(neg) > n_neg else neg

balanced = pd.concat([pos, neg_sample], axis=0).sample(frac=1, random_state=7)
balanced.to_csv(out_balanced, index=False)

print("Full labeled table:", out_full)
print(df["true_direct_hit"].value_counts(dropna=False))

print("\nBalanced training table:", out_balanced)
print(balanced["true_direct_hit"].value_counts())

print("\nTop positive examples:")
print(
    pos[[
        "fragment_id", "protein", "log2FC", "competition_fc",
        "best_competitor", "competition_supported", "true_direct_hit"
    ]].head(20).to_string(index=False)
)
