from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from fragment_hit_triage.io import read_table, write_table
from fragment_hit_triage.ml import load_model, predict_with_model
from fragment_hit_triage.scoring import TriageConfig, score_interactions


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Score fragment-protein data and optionally add supervised ML probabilities.")
    p.add_argument("--interactions", required=True)
    p.add_argument("--protein-annotations", default=None)
    p.add_argument("--model", default=None, help="Optional models/direct_hit/direct_hit_model.joblib")
    p.add_argument("--out", default="predictions.csv")
    p.add_argument("--log2fc-threshold", type=float, default=2.3)
    p.add_argument("--pvalue-threshold", type=float, default=0.05)
    return p


def main() -> None:
    args = build_parser().parse_args()
    interactions = read_table(args.interactions)
    annotations = read_table(args.protein_annotations) if args.protein_annotations else None
    config = TriageConfig(log2fc_threshold=args.log2fc_threshold, pvalue_threshold=args.pvalue_threshold)
    scored = score_interactions(interactions, protein_annotations=annotations, config=config)
    if args.model:
        bundle = load_model(args.model)
        scored = predict_with_model(scored, bundle)
    write_table(scored, args.out)
    print(f"Wrote {len(scored):,} rows to {args.out}")


if __name__ == "__main__":
    main()
