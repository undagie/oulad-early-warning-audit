# Auditing Temporal Leakage and Rule Reliance in Early Warning of Students at Risk

Code for the study *Auditing Temporal Leakage and Rule Reliance in Early Warning of Students at Risk* (manuscript under revision at *Statistics, Optimization and
Information Computing*). The repository implements a cutoff-first early-warning benchmark on
the Open University Learning Analytics Dataset (OULAD) and the audits reported in the paper:

- **Temporal validity.** Every source table is truncated at the prediction cutoff before any
  join or aggregation; a deliberately leaky boundary condition, a temporally invariant
  negative control, score-visibility lags, and three cohort definitions bound the result.
- **Neural–symbolic reliance.** A recurrent predictor (GRU) is combined with seven rule-based
  alerts through a late logistic fusion whose standardized coefficients, residual-signal
  diagnostics, and equivalence tests measure how much the rules add to ranking (they add
  almost nothing on full data).
- **Rule-based alert audit.** Availability, sparsity, per-rule diagnostic validity with
  student-clustered intervals, redundancy, leave-one-rule-out, threshold and weight
  sensitivity, subgroup performance, and a structural contrast with SHAP and counterfactuals.

## Installation

Python 3.10 or newer. A CUDA GPU is optional but shortens training considerably.

```bash
pip install -r requirements.txt          # minimum versions
# or, to reproduce the exact environment used for the paper:
pip install -r requirements-lock.txt
```

## Data

OULAD is not redistributed here. Download it from the Open University
(<https://analyse.kmi.open.ac.uk/open_dataset>) or the UCI repository and extract the seven CSV
files into `data/raw/`:

`courses.csv`, `studentInfo.csv`, `studentRegistration.csv`, `assessments.csv`,
`studentAssessment.csv`, `studentVle.csv`, `vle.csv`.

OULAD is released under CC BY 4.0 (Kuzilek, Hlosta, and Zdrahal, *Scientific Data* 4, 170171, 2017).

## Reproducing the analyses

```bash
python scripts/check_data.py                                  # structure and base rates

# 1. Main condition: per-seed tuning of GRU, LSTM, Transformer, XGBoost and Static-only
#    on each seed's own validation split; per-row predictions are cached.
python scripts/rev_run.py --condition main

# 2. Sensitivity conditions (reuse the per-seed choices of the main condition)
for c in leaky lag7 lag14 lag21 fixed_cohort enrolment_cohort transfer_disjoint; do
  python scripts/rev_run.py --condition $c
done
python scripts/rev_run.py --condition lomo --seeds 42 43 44   # leave one module out

# 3. Supporting analyses
python scripts/rev_cohort_accounting.py                       # cohort rule and exclusions
python scripts/rev_data_efficiency.py --window 2              # matched train/validation budget
python scripts/rev_explainers.py --seed 42                    # grouped Shapley, TreeSHAP, DiCE

# 4. All tables from the cached predictions (no retraining)
python scripts/rev_analyze.py
```

Predictions are written to `outputs/revision/preds/<condition>/wK_sS.parquet` (validation and
test rows, every model, rule activations and rule inputs) with a JSON file holding the selected
hyperparameters, the full tuning log, and the fusion coefficients. Tables are written to
`outputs/revision/tables/`. On one consumer GPU the main condition takes about two hours and
all sensitivity conditions together about three more.

`results/` contains the per-seed tables and tuning logs produced for the paper.

## Layout

```
config/config.yaml        windows, seeds, split ratios, label definition
src/nsoulad/
  data/                   loader, labels, cutoff-first windowing, cohort rules, splits
  features/               weekly engagement sequence, assessment, course structure, static encoder
  symbolic/               rule-based alerts R1-R7 and the engine (thresholds fit on training data)
  models/                 GRU/LSTM/Transformer, XGBoost, Static-only, logistic fusion
  evaluation/             metrics, statistics (corrected resampled t, TOST, cluster bootstrap),
                          revision analyses
  revision.py             one (condition, window, seed) run with per-seed tuning
scripts/                  experiment runners; rev_*.py reproduce the revised paper
tests/                    pytest suite on a controlled synthetic OULAD fixture
docs/                     analysis protocol for the neural-uncertain-band diagnostic
```

## Tests

```bash
python -m pytest -q
```

## License and citation

Code: MIT License (see `LICENSE`). If you use this code, please cite the paper and the archived
software release (see `CITATION.cff`).
