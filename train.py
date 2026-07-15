from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from fragment_hit_triage.io import read_table, write_table
from fragment_hit_triage.ml import train_direct_hit_model
from fragment_hit_triage.scoring import TriageConfig, score_interactions


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train a supervised model for validated direct fragment-protein hits.")
    p.add_argument("--interactions", required=True, help="CSV/TSV with fragment-protein measurements and a label column.")
    p.add_argument("--protein-annotations", default=None)
    p.add_argument("--label-col", default="true_direct_hit", help="Binary label: 1=true/optimizable direct hit, 0=indirect/background/non-hit.")
    p.add_argument("--outdir", default="models/direct_hit")
    p.add_argument("--scored-out", default=None, help="Optional path to write engineered/scored training table.")
    p.add_argument("--log2fc-threshold", type=float, default=2.3)
    p.add_argument("--pvalue-threshold", type=float, default=0.05)
    p.add_argument("--test-size", type=float, default=0.25)
    return p


def main() -> None:
    args = build_parser().parse_args()
    interactions = read_table(args.interactions)
    annotations = read_table(args.protein_annotations) if args.protein_annotations else None
    config = TriageConfig(log2fc_threshold=args.log2fc_threshold, pvalue_threshold=args.pvalue_threshold)
    scored = score_interactions(interactions, protein_annotations=annotations, config=config)
    # Preserve label if scoring canonicalized/merged columns but did not include it at the front.
    if args.label_col not in scored.columns and args.label_col in interactions.columns:
        scored[args.label_col] = interactions[args.label_col].values
    if args.scored_out:
        write_table(scored, args.scored_out)
    result = train_direct_hit_model(scored, label_col=args.label_col, outdir=args.outdir, test_size=args.test_size)
    print(f"Saved model to {result['model_path']}")
    print("Metrics:")
    for k, v in result["metrics"].items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
