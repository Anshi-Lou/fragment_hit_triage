import pandas as pd
from pathlib import Path

infile = Path("data/finalScreen.tsv")
outfile = Path("data/finalScreen_for_triage.csv")

df = pd.read_csv(infile, sep="\t")

print("Original columns:")
print(df.columns.tolist())

out = pd.DataFrame()

# Official Ligand Discovery column -> our project column
out["fragment_id"] = df["fragId"].astype(str)

if "geneName" in df.columns:
    protein = df["geneName"].copy()
    if "accession" in df.columns:
        protein = protein.fillna(df["accession"])
    out["protein"] = protein.astype(str)
else:
    out["protein"] = df["accession"].astype(str)

if "accession" in df.columns:
    out["accession"] = df["accession"].astype(str)

out["log2FC"] = pd.to_numeric(df["l2fc"], errors="coerce")

if "l2fcM" in df.columns:
    out["median_adjusted_log2FC"] = pd.to_numeric(df["l2fcM"], errors="coerce")

# Official table stores -log10(p), so convert back to p-value
if "ml10p" in df.columns:
    out["pvalue"] = 10 ** (-pd.to_numeric(df["ml10p"], errors="coerce"))

if "ml10adjP" in df.columns:
    out["adj_pvalue"] = 10 ** (-pd.to_numeric(df["ml10adjP"], errors="coerce"))

# mdfClass: 2/3 are medium/high confidence hits in the official app
if "mdfClass" in df.columns:
    out["hit"] = (pd.to_numeric(df["mdfClass"], errors="coerce").fillna(0) >= 2).astype(int)
    out["mdfClass"] = df["mdfClass"]

for col in ["protHits", "ligHits"]:
    if col in df.columns:
        out[col] = df[col]

out.to_csv(outfile, index=False)

print(f"\nWrote: {outfile}")
print("New columns:")
print(out.columns.tolist())
print(out.head())
