# Fragment–Protein Direct Hit Triage

This project implements a practical scoring and modeling workflow for deciding whether a **fragment–protein chemoproteomics signal** is likely to be:

1. a real, optimizable direct binding hit,
2. a fragment-promiscuity artifact,
3. a protein frequent-hitter / photocrosslinker-labeling-bias artifact, or
4. an indirect cellular partitioning / compartment-enrichment signal.

It is a runnable baseline implementation. It is not a full reproduction of the Ligand Discovery production stack, which uses FFF-specific descriptors, TabPFN classifiers, and richer interactome-signature modeling. This implementation is deliberately lightweight so it can run locally on a normal laptop.

## What the code does

For each fragment–protein row, the scorer computes:

- primary hit status from `log2FC > 2.3` and optional `pvalue/adj_pvalue < 0.05`, unless a hit column is already supplied;
- fragment-level risk: hit count, hit ratio, predicted promiscuity probability, and optional physicochemical flags;
- protein-level risk: protein frequent-hitter count, known labeling-bias/background flags, sticky compartments, and optional abundance;
- interactome-level partitioning risk: compartment enrichment among a fragment's hit proteins;
- iScore-like specificity score: enrichment strength penalized by fragment and protein promiscuity;
- validation evidence: lower-dose repeat, analog competition, orthogonal validation, and SAR coherence;
- final `direct_hit_priority_score`, `triage_class`, explanation, and recommended next experiment.

## Folder structure

```text
fragment_hit_triage/
├── app.py                         # Streamlit UI
├── score.py                       # Rule-based triage CLI
├── train.py                       # Train supervised direct-hit model
├── predict.py                     # Score + optional ML probability
├── requirements.txt
├── data/
│   ├── sample_interactions.csv
│   └── sample_protein_annotations.csv
└── src/fragment_hit_triage/
    ├── scoring.py
    ├── ml.py
    ├── io.py
    ├── chem.py
    └── recommendations.py
```

## Setup

From PowerShell, CMD, Anaconda Prompt, or terminal:

```bash
cd fragment_hit_triage
python -m venv .venv
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# macOS/Linux:
# source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

RDKit is optional. Without RDKit, the code still works using supplied columns and SMILES text features for the ML model. With RDKit installed, extra descriptors such as MW, LogP, HBA/HBD, aromatic rings, TPSA, and formal charge are computed automatically.

## Quick test

```bash
python score.py --interactions data/sample_interactions.csv --out outputs/sample_scores.csv
```

Expected output columns include:

- `direct_hit_priority_score`
- `triage_class`
- `iscore`
- `iscore_percentile`
- `fragment_risk`
- `protein_risk`
- `partitioning_risk`
- `explanation`
- `recommended_next_step`

## Train a supervised model

If your data contains a binary label such as `true_direct_hit`:

```bash
python train.py \
  --interactions data/sample_interactions.csv \
  --label-col true_direct_hit \
  --outdir models/direct_hit \
  --scored-out outputs/training_scored.csv
```

Then use it:

```bash
python predict.py \
  --interactions data/sample_interactions.csv \
  --model models/direct_hit/direct_hit_model.joblib \
  --out outputs/sample_predictions.csv
```

The prediction output adds:

- `ml_direct_hit_probability`
- `hybrid_priority_score`

## Run the web app

```bash
python -m streamlit run app.py
```

Upload your interactions table and optionally upload a protein annotation table or a trained `.joblib` model.

## Input schema

Minimum required columns:

```csv
fragment_id,protein,log2FC,pvalue
C001,DDB1,4.1,0.001
C001,CDK2,0.2,0.8
```

Recommended columns:

```csv
fragment_id,smiles,protein,log2FC,pvalue,intensity,replicate_corr,crf_log2fc,no_uv_log2fc,bead_log2fc,compartment,family,known_labeling_bias,predicted_promiscuity_probability,lower_dose_log2fc,competition_fc,orthogonal_validation,sar_coherent,true_direct_hit
```

Optional protein annotation table:

```csv
protein,compartment,family,known_labeling_bias
DDB1,nucleus,E3 ligase adapter,0
TIMM17A,mitochondrial inner membrane,translocase/background,1
```

## How to interpret classes

- `high_priority_direct_hit`: strong enrichment, high iScore percentile, low promiscuity/background risk, and ideally validation support.
- `medium_priority_validate`: plausible, but needs lower-dose repeat, analog competition, or orthogonal validation.
- `likely_partitioning_or_compartment_signal`: fragment hit pattern is enriched for a compartment such as lysosome, mitochondria, ER/Golgi, or membranes.
- `likely_promiscuity_or_labeling_bias`: fragment and/or protein is broad-hitting or background-prone.
- `low_confidence_signal`: signal is not strong, not reproducible, low-intensity, or not well controlled.
- `not_primary_hit`: does not pass the primary hit threshold.

## Notes

The scoring weights are transparent and editable in `src/fragment_hit_triage/scoring.py`. For a real research project, keep this baseline as a triage layer, then add experimentally grounded labels from lower-dose repeats, competition experiments, recombinant/biophysical assays, and SAR to train a stronger supervised model.
