from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from fragment_hit_triage.io import read_table, write_table
from fragment_hit_triage.scoring import TriageConfig, score_interactions


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Score fragment-protein signals for direct-hit priority.")
    p.add_argument("--interactions", required=True, help="CSV/TSV with one row per fragment-protein measurement.")
    p.add_argument("--protein-annotations", default=None, help="Optional CSV/TSV keyed by protein/gene with compartment/family/background flags.")
    p.add_argument("--out", default="triage_scores.csv", help="Output CSV/TSV path.")
    p.add_argument("--log2fc-threshold", type=float, default=2.3)
    p.add_argument("--pvalue-threshold", type=float, default=0.05)
    p.add_argument("--competition-fc-threshold", type=float, default=-1.0, help="Competition log2FC cutoff. More negative means stronger competition.")
    p.add_argument("--lower-dose-log2fc-threshold", type=float, default=1.0)
    return p


def main() -> None:
    args = build_parser().parse_args()
    interactions = read_table(args.interactions)
    annotations = read_table(args.protein_annotations) if args.protein_annotations else None
    config = TriageConfig(
        log2fc_threshold=args.log2fc_threshold,
        pvalue_threshold=args.pvalue_threshold,
        competition_fc_threshold=args.competition_fc_threshold,
        lower_dose_log2fc_threshold=args.lower_dose_log2fc_threshold,
    )
    scored = score_interactions(interactions, protein_annotations=annotations, config=config)
    write_table(scored, args.out)
    print(f"Wrote {len(scored):,} scored rows to {args.out}")
    top = scored.sort_values("direct_hit_priority_score", ascending=False).head(10)
    print("\nTop signals:")
    cols = ["fragment_id", "protein", "direct_hit_priority_score", "triage_class", "iscore_percentile", "fragment_risk", "protein_risk", "partitioning_risk"]
    print(top[[c for c in cols if c in top.columns]].to_string(index=False))


if __name__ == "__main__":
    main()
