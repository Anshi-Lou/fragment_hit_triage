import pandas as pd
from pathlib import Path

smiles = pd.read_csv("data/cemm_smiles.csv")
smiles = smiles.rename(columns={"fid": "fragment_id"})
smiles["fragment_id"] = smiles["fragment_id"].astype(str).str.strip()

files = [
    "data/finalScreen_for_triage.csv",
    "data/finalScreen_labeled_for_training.csv",
    "data/finalScreen_training_balanced.csv",
]

for f in files:
    p = Path(f)
    if not p.exists():
        print("Skip missing:", f)
        continue

    df = pd.read_csv(p)
    df["fragment_id"] = df["fragment_id"].astype(str).str.strip()

    if "smiles" in df.columns:
        df = df.drop(columns=["smiles"])

    out = df.merge(smiles[["fragment_id", "smiles"]], on="fragment_id", how="left")

    matched = out["smiles"].notna().mean()
    out.to_csv(p, index=False)

    print(f)
    print("rows:", len(out))
    print("SMILES matched:", round(matched * 100, 2), "%")
    print()
