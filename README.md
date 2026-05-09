# MM-GraphSurv

Hierarchical multi-modal graph survival model for ICU outcomes.

Each modality (dynamic vitals, static demographics, ICD codes, radiology
embeddings) is represented as its own intra-modality graph; per-window
graphs are stitched together by a hierarchical fusion block with learned
cross-modality edges, processed by a GIN, and fed to per-cause discrete-time
hazard heads. Modality drop-out at inference is supported for ablating
input streams without retraining.

## Requirements

```
pip install -r requirements.txt
```

PyTorch ≥ 2.0 (GPU recommended for training, CPU is fine for inference).
Clinical-Longformer is pulled from HuggingFace on first use for radiology
report embeddings.

## Data

This repo contains modelling code only — preprocessing pipelines for each
cohort live upstream and produce the `.npy` arrays this code consumes from
`data/{dataset}/processed/`. Supported datasets:

| Dataset | Risks                            | Modalities                |
|---------|----------------------------------|---------------------------|
| MIMIC-IV| ICU mortality                    | dynamic, static, ICD, rad |
| eICU    | mortality, discharge (competing) | dynamic, static, ICD      |
| MC-MED  | ICU admission, inpatient care    | dynamic, static, ICD, rad |
| HiRID   | mortality (incl. circ. failure)  | dynamic, static           |

Window grid is 6 × 4-hour blocks over the first 24 h of admission. ICD and
radiology blocks are soft-pooled to 50 nodes each via two-layer MLP
soft-assignment.

## Training

The main model on one dataset across the standard five seeds:

```
python scripts/train_mmg.py --dataset mimic
python scripts/train_mmg.py --dataset eicu  --seeds 42,123,7
python scripts/train_mmg.py --dataset hirid --force
```

Checkpoints land at `checkpoints/{dataset}_mmgraphsurv_seed{seed}/` with
`best_model.pth` + `metrics.json`. Re-runs skip completed seeds unless
`--force` is set.

Baselines (each takes the same `--dataset` flag):

```
scripts/train_cox.py
scripts/train_deepsurv.py
scripts/train_deephit.py
scripts/train_dynamic_deephit.py
scripts/train_dysurv.py
scripts/train_survtrace.py
```

## Post-hoc analysis

```
scripts/calibrate.py                # MM-GraphSurv calibration (isotonic / temperature)
scripts/calibrate_cox_deepsurv.py   # same for the linear baselines
scripts/compute_auc.py              # AUROC / time-dependent AUC table
```

## Layout

```
src/
  models/        MMGraphSurv, fusion, GIN/SoftPool components
  preprocessing/ per-dataset loaders + Longformer cache
  eval/          metrics, calibration, Aalen-Johansen recalibration
  train.py       training loop (used by all scripts)
  losses.py      NLL + ranking + interpretability regulariser
configs/         per-dataset YAML (cohort, model, loss, training)
scripts/         training + analysis entry points
```

## Configuration

Per-dataset YAML in `configs/`. Hyperparameter overrides go in the YAML;
scripts only set seed and output paths. The model section controls graph
construction (`k_neighbors`, `alpha_ema`), GIN width (`gin_dims`), pooled
node counts (`pool_dims`), competing-risk options
(`cause_specific_proj`, `cause_specific_gin`, `per_cause_hazard`), and
ablation switches (`fusion_mode: graph|mlp_concat`,
`adjacency_mode: learned|uniform_frozen`).

## Notes

- `checkpoints/` and `data/` are gitignored. Trained weights are not
  distributed with the repo.
- The fusion block keeps an interpretability matrix `I^(m→n)` per
  cross-modal edge; `forward()` returns the last-window fused
  `A_fused` / `I_fused` for attribution.
- For competing-risk datasets, the loss combines per-cause NLL, a
  pairwise ranking term, the interpretability regulariser, and an
  optional CIF-smoothness penalty (see `configs/{dataset}.yaml` →
  `loss`).
