# ACV Refrigerant-Leak Fault Localization — NebulaX Hackathon 2026, PS3 (ACV subsystem)

Unsupervised anomaly detection that ranks a train's 8 cars by how likely each
is to be the one with a refrigerant-leak / ACV fault, from multi-day sensor
logs sampled every 30 seconds.

**This is the final, settled pipeline: a single model, Elliptic Envelope,
tuned for higher sensitivity to small deviations.** This build originally
implemented and evaluated 5 unsupervised models (Isolation Forest, PCA
Reconstruction, Local Outlier Factor, and a small autoencoder, in addition
to Elliptic Envelope), each at two hyperparameter presets, then briefly kept
Elliptic Envelope + Isolation Forest as a pair for methodological diversity.
Elliptic Envelope alone is the final choice: it matches Isolation Forest's
accuracy while producing more informative results on the one file both
missed. See "Why this model, at these settings" below for the full
reasoning, and this repository's git history for the complete comparisons
(5 models, then 2, that led here).

**To score any case file yourself** (training, validation, or the real test
file, whichever you point it at):

```bash
python scripts/score_test_file.py path/to/any_case.xlsx
```

This prints a full ranking of that file's 8 cars, most-to-least likely
faulty, and writes it to `results/test_submission.csv` in the exact PS3
format. See "Scoring a case file" below for details and options.

## Scope

This build covers only the 8-parameter schema shared by **Model A** (cases
01/02/03) and **Model C** (cases 05/06) — 5 of the 6 labeled files.

`acv_case_04.xlsx` (**Model B**, ~60 parameters) is **deliberately excluded**:
it uses a materially different, much richer feature space, and with only one
labeled file for that schema, it can't use the same leave-one-file-out
validation the other 5 files share. It needs its own pipeline. This is a
**known risk, not solved here** — see "Known risks" below.

## Data

```
data/train/
  acv_case_01.xlsx   acv_case_02.xlsx   acv_case_03.xlsx   (Model A, train 620/620/619)
  acv_case_05.xlsx   acv_case_06.xlsx                      (Model C, train 407/408)
  labels.csv                                               (filename, faulty_car)
```

(`Train_Labels.csv` at the repo root is the organizers' original label file,
covering all 6 cases including the excluded `acv_case_04.xlsx`; `data/train/labels.csv`
is the 5-row subset actually used by this pipeline, built from the labels
given for this build's scope and cross-checked against it.)

Each file: one sheet ("Sheet1"), one row per timestamp, columns named
`Car {01..08} - {Parameter}` plus `Car model` / `Train number` / `Time`.
Verified directly against the files (not assumed): Model A uses `Outdoor
Average Temperature`; Model C uses `Outside Temperature Sensor Reading` — both
normalize here to one column, `Outside Temperature`.

### Data-reality findings (verified against the real files, resolved before building the pipeline)

1. **Whole-timestamp sensor dropout, ~9-14% of rows per file** (0% in
   `acv_case_03.xlsx`). At the affected timestamps *all 8 cars, all 8
   parameters* are `NaN` simultaneously — a clean, unbiased block dropout, not
   per-car or per-parameter missingness. Confirmed by checking that
   "any car NaN" == "all 8 cars NaN" at every affected timestamp, for every
   file. See `results/missing_value_report.csv` for the exact per-file
   drop-vs-forward-fill decision (4 of 5 files exceed the 5% threshold and use
   forward-fill within each car's own time series; `acv_case_03.xlsx` is
   clean and uses drop).

2. **Model C's non-end cars never have a real "Outside Temperature" reading.**
   Verified: in `acv_case_05.xlsx`/`acv_case_06.xlsx`, cars 02-07 report the
   literal string `"Invalid"` for this sensor on essentially every non-dropout
   row (never a numeric value), while cars 01/08 always report a real reading.
   This is a physical/schema fact (only the end cars carry this sensor on
   Model C), not a data quality bug — confirmed it's unrelated to the general
   `ACV Information Valid` status flag, which is `Valid` for those same rows.
   **Resolved per user decision:** at each timestamp, broadcast the mean of
   whichever end car(s) (01/08) have a valid reading to any car missing its
   own (a car's own valid reading, when present, is always kept — a true
   no-op for Model A, where every car already reports its own value). This is
   a deterministic, within-timestamp, within-file sensor fusion using only
   redundant information already present in that row's timestamp — not a
   fitted statistic — so, like the one-hot category vocabulary, it's applied
   once globally rather than refit per CV fold.

3. **Rare categorical values** (<1% of non-missing rows, global frequency
   across all 5 files) are merged into an `"Other"` bucket per parameter —
   see `results/rare_category_merges.csv` for the exact list and frequencies.

4. **`Load Halved` is constant** (`"Normal"`) in every non-missing row across
   all 5 labeled files — no other value is ever observed. Its one-hot
   encoding is a single always-1 column. This is also the reason Elliptic
   Envelope's covariance matrix is reported as not full rank (see below).

## Pipeline architecture

```
src/acv_fault/
  config.py            constants (car numbers, parameter names, thresholds)
  data_loading.py       .xlsx -> long format; Outside Temperature alias normalization + end-car fix
  preprocessing.py       missing-value handling, rare-category merge, one-hot encoding,
                          cross_car_temp_deviation engineering, fold-safe standardization
  aggregation.py          per-timestamp scores -> per-car median/p90 -> ranking (+ tie detection)
  evaluation.py           leave-one-case-out loop, PS3 primary_metric + MRR
  submission.py           PS3-format CSV writer + score_new_file() entrypoint
  models/
    elliptic_envelope.py   sklearn EllipticEnvelope, LedoitWolf fallback on singular covariance
scripts/
  run_pipeline.py        end-to-end LOCO evaluation entrypoint
  score_test_file.py     score one case file (train/validation/test) via submission.score_new_file()
results/                 generated reports (see "Results" below)
```

### Fold-safety discipline

The task requires preprocessing to be refit inside each cross-validation fold,
never fit once globally. This codebase makes that a structural property
rather than a discipline to remember:

- `preprocessing.prepare_global()` runs only the steps that are safe to fit
  once across all 5 files: the one-hot category vocabulary and rare-category
  merge (a fixed domain fact, not a learned distribution — explicitly allowed
  by the task spec), the per-file missing-value drop/forward-fill decision
  (a deterministic rule evaluated on that file's own rows), and the raw
  (unstandardized) `cross_car_temp_deviation` feature (a deterministic
  per-timestamp computation within one file, not a fitted statistic).
- `preprocessing.fit_standardizer()` / `apply_standardizer()` are the only
  functions that touch a fitted statistic (per-`car_model` mean/std). Every
  call site in `evaluation.py` fits on the training-pool rows only and
  applies (never refits) to the held-out file.
- The model is constructed fresh per fold in `evaluation.py` and fit only on
  that fold's training-pool feature matrix.

### Model (anomaly score: higher = more anomalous)

**Elliptic Envelope / Mahalanobis distance** — `sklearn.covariance.EllipticEnvelope`;
catches a failed fit and falls back to a `LedoitWolf` shrinkage covariance
(manual squared Mahalanobis distance) for that fold, reporting which one.
`support_fraction=0.35` (sklearn default: `None`, ≈0.5 for this data's size)
fits the covariance to a smaller, purer "core" of the training data — the
same absolute deviation counts for more Mahalanobis distance against a
tighter estimate of normal. `contamination=0.15` (sklearn default `0.1`)
only affects the count-only `flag()` method (see "Anomaly counts" below),
never the ranking.

### Aggregation

Per car: **median** and **90th-percentile (p90)** anomaly score across that
car's rows in the file, both reported (never just one). Ranking = descending
sort by score.

**Tie detection:** this sensor data is heavily duplicated (steady-state
readings repeat for long stretches — 90-96% exact-duplicate feature rows,
checked directly in one fold), so it's common for multiple cars to land on
the *exact same* aggregated score. A rank decided by a tie is not genuine
model discrimination — it's whatever order the tie-break (deterministically,
ascending car number: `aggregation.rank_cars` sorts with `kind="stable"`)
happens to produce. `aggregation.n_tied_with_car` and
`evaluation.RunResult.n_tied_with_true_car` detect this explicitly per run
rather than silently reporting a tie-broken rank as a real one.

## Evaluation: leave-one-case-out, scored with PS3's formula

For each of the 5 files held out in turn: refit standardization + the model
on the pooled *other 4* files only, score the held-out file, aggregate
(median and p90), rank, and score with PS3's formula:

```
primary_metric = (n - (r - 1)) / n      n = 8, r = 1-indexed rank of the true faulty car
mrr_secondary  = 1 / r                  (secondary, not used for model selection)
```

### Results (best → worst by primary_metric_mean)

| aggregation | primary_metric_mean | mrr_secondary_mean | n files with tied rank | rank of true car per file (01, 02, 03, 05, 06) |
|---|---|---|---|---|
| **p90** | **0.925** | 0.850 | **0** | 1, 1, 1, 4, 1 |
| median | 0.775 | 0.583 | 1 | 2, 6, 1, 4, 1 |

**p90 is the aggregation to use** — higher score, and zero tie-decided ranks
(every rank it reports is genuine model discrimination). It correctly
identifies the true fault in **4 of the 5 labeled files** (`acv_case_01/02/03/06.xlsx`);
its one miss (`acv_case_05.xlsx`, rank 4) is discussed below.

Full per-run detail: `results/loco_run_results.csv` (10 rows: 5 files × 2
aggregations). Per-fold notes and every tie-decided rank:
`results/fold_notes.txt`. Rows dropped/imputed per file:
`results/missing_value_report.csv`. Rare categories merged:
`results/rare_category_merges.csv`. Elliptic Envelope needed the LedoitWolf
fallback in **0 of 5** folds — all 5 native fits succeeded despite sklearn's
"covariance matrix... not full rank" warning (caused by the constant
`Load Halved=Normal` column); the fallback code path exists and is exercised
whenever a fit actually raises.

### Anomaly counts (`flag()`, independent of the ranking above)

`results/anomaly_counts.csv` / `results/anomaly_counts_summary.csv` —
22.0% of all rows (61,100 / 277,848) flagged across the 5 folds.
**Caveat:** in the `acv_case_06.xlsx` fold, `flag()` marks **100%** of that
file's rows anomalous — for every car, not just the faulty one. This is not
a bug; it's a real, independently-corroborated property of this specific
fold: `acv_case_05.xlsx` (Mar 2021, ~15-35°C outside temperature) and
`acv_case_06.xlsx` (Jul 2020, ~30-46°C) are different seasons, and when
`acv_case_06.xlsx` is held out, its training pool's *only* Model C file is
`acv_case_05.xlsx` — so the fitted "normal" baseline for Model C is
calibrated on a different season's data, and case 06's hot-summer readings
look shifted across *all 8 cars uniformly* (a file-level effect, not a
car-level fault). The ranking is far less affected by this (it only compares
cars *within* the same file, where a uniform shift roughly cancels out), but
the raw anomaly count should not be trusted in isolation on that file.

## Why case 05 is hard to detect

Neither aggregation ranks `acv_case_05.xlsx`'s true fault (car 04) first —
both land it at rank 4. Investigated directly against the real per-car
sensor values, compared against an "easy" case (`acv_case_01.xlsx`, car 01,
correctly ranked #1) as a contrast:

1. **Much smaller effect size, and no corroborating signal.** Case 01's fault
   car shows up on multiple independent signals at once: its indoor
   temperature is the clear outlier (highest mean of the 8 cars, ~40% higher
   std than peers), it uniquely enters rare "Invalid"/"Other" categorical
   states that no other car in that file ever reaches, and it even runs a
   different cooling-setpoint distribution than its peers. Case 05's fault
   car (04) shows only a marginally elevated indoor temperature (highest of
   the 8, but by a margin smaller than ordinary between-car noise) and is
   otherwise indistinguishable from its healthy peers on every categorical
   and setpoint dimension checked.

2. **The engineered deviation feature can't tell direction.**
   `cross_car_temp_deviation` (each car's *absolute* deviation from the
   fleet's median indoor temperature) can't distinguish "running warm" from
   "running cold." Checking the *signed* deviation day by day in
   `acv_case_05.xlsx`: car 04 (the true fault) runs consistently **warmer**
   than the fleet median every day (+0.02, +0.21, +0.12, +0.37 °C) — exactly
   what a refrigerant leak reducing cooling effectiveness would cause. Car 05
   (the car the model ranks highly instead) runs consistently **colder**
   every day (−0.23, −0.63, −0.46, −0.17 °C) — a *larger* deviation, but in
   the physically wrong direction for this fault type.

3. **The fault is still developing, and no feature looks at trend.** Car 04
   is the *only* car in the file whose deviation from the fleet median grows
   over the 4-day window (fitted slope +0.065/day; every other car's
   deviation shrinks over the same window) — consistent with a
   slowly-worsening leak — but `median` and `p90` both collapse the whole
   window into one number and discard *when* the deviation happened.

**Not implemented in this build, per explicit scope decision:** a signed
(directional) deviation feature and/or a within-file trend feature would
likely help specifically with this file — both change the model's *inputs*,
which was scoped out in favor of a hyperparameter-only sensitivity pass
(below), which was tried and did not unlock this file either.

## Why this model, at these settings

This build went through three stages before settling here, each verified by
re-running the same leave-one-case-out evaluation rather than assumed:

1. **All 5 models the task suggested** (Isolation Forest, Local Outlier
   Factor, Elliptic Envelope, PCA Reconstruction, a small autoencoder), each
   at this project's original hyperparameters and at a "high sensitivity"
   preset (each model's own documented lever tuned toward reacting to
   smaller deviations) — 10 (model, aggregation) combinations. Measured by
   how many of the 5 files each model's best combination correctly ranks the
   true fault #1 in: **Elliptic Envelope and Isolation Forest tied for best
   at 4 of 5**, at both presets. PCA Reconstruction looked competitive at
   first (it never produced a tie-decided rank) but its actual accuracy was
   a real step down — 3 of 5 at best, dropping to 2 of 5 under higher
   sensitivity, including ranking the true fault 7th of 8 cars on
   `acv_case_02.xlsx`. "Never tie-decided" described the integrity of its
   ranks, not their correctness. Local Outlier Factor (1 of 5 at best, and
   numerically unstable under `p90` on this heavily-duplicated data) and the
   autoencoder (2 of 5 at best) were clearly weaker throughout.
2. **Elliptic Envelope + Isolation Forest**, kept together for a period
   specifically for methodological diversity (covariance/distance-based vs.
   isolation-based) despite tying on raw accuracy, since they cover
   different detection paradigms.
3. **Elliptic Envelope alone** — the final choice. It matches Isolation
   Forest's accuracy exactly (identical 4-of-5 ranking, identical stability
   under higher sensitivity) and is the more informative of the two on the
   one file both miss: Elliptic Envelope's case-05 result is a narrow 2-way
   tie (some real signal), while Isolation Forest's is a complete 8-way tie
   (zero discriminating signal for that file). With the diversity goal
   dropped, a single simpler, equally-accurate, more-informative model was
   preferred over carrying a second one for its own sake.

Neither a second model nor either hyperparameter preset unlocks case 05 —
see "Why case 05 is hard to detect" above for the data-level reason a
feature change (not tried here) is the more promising next step.

## Scoring a case file

```bash
python scripts/score_test_file.py path/to/any_case.xlsx
python scripts/score_test_file.py path/to/any_case.xlsx --aggregation median --out results/my_submission.csv
```

Works on **any file matching this pipeline's schema** — one of the 5
labeled training files, a validation file you hold out yourself, or the
eventual real `acv_test_case.xlsx` — you choose which. It fits the
standardizer and model on all 5 labeled files (reusing the *already-fitted*
category vocabulary/rare-category map from them, never refit on the file
being scored — there's no held-out label left to protect once this is a
real, unlabeled file), scores every car in the given file, and prints/writes
its full ranking. `--aggregation` defaults to `p90` (the better of the two
per the results above); `--out` defaults to `results/test_submission.csv`.

Programmatically, this is one call — `submission.score_new_file(xlsx_path,
prepared_labeled_df, prep_report, "elliptic_envelope", aggregation_name)` —
see `src/acv_fault/submission.py`.

### Submission-format dry-run (against the 5 labeled files)

`results/submission_dryrun.csv` — the LOCO held-out predictions under the
best aggregation (`p90`), in the exact PS3 format:

```
file_id,ranked_cars
acv_case_01.xlsx,01|02|08|06|07|05|04|03
acv_case_02.xlsx,02|01|03|05|04|06|08|07
acv_case_03.xlsx,03|04|08|02|01|07|06|05
acv_case_05.xlsx,01|02|07|04|03|05|06|08
acv_case_06.xlsx,06|01|08|07|03|02|04|05
```

Car identifiers are exactly as they appear in each file's own column headers
("03", not "Car 3"), pipe-separated, most-to-least likely faulty. This is a
**format dry-run against the labeled files only, produced by the LOCO
evaluation** (each row scored by a model that never saw that file during
training) — it's a validity check on the format and process, not a real
submission. For an actual new file, use `scripts/score_test_file.py` above.

## How to run

```bash
uv venv .venv && source .venv/bin/activate
uv pip install -r requirements.txt
PYTHONPATH=src python scripts/run_pipeline.py                         # full LOCO evaluation, ~1 minute
PYTHONPATH=src python scripts/score_test_file.py path/to/file.xlsx    # score one case file
```

## Written summary

### (a) Best model/aggregation, and how consistent it is

Elliptic Envelope at `p90` aggregation is the settled choice (primary_metric
0.925, MRR 0.850, zero tie-decided ranks — every reported rank is genuine
model discrimination). **Performance is not uniform across files — it's
concentrated.** It nails `acv_case_01/02/03/06.xlsx` and misses only
`acv_case_05.xlsx` (car 04), landing it at rank 4 rather than 1. So the
reported numbers are really "4 files this pipeline solves well, plus one
(`acv_case_05.xlsx`, Model C, car 04) it doesn't yet solve" — see "Why case
05 is hard to detect" for the specific, data-grounded reasons, and "Why this
model, at these settings" for what was tried (a 5-model, then 2-model,
two-preset comparison) and didn't change that.

### (b) Known risk: this build cannot score a Model B-schema test file

**This is the most important caveat in this submission.** This entire
pipeline — the 8-parameter feature list, the one-hot vocabulary, the
standardization groups, the model — is built around the schema shared by
Model A and Model C. `acv_case_04.xlsx` (Model B) uses a materially
different, ~60-parameter schema and was **deliberately excluded** from this
build: with only one labeled Model B file, there's no way to leave-one-file-out
validate a Model B pipeline the way this one is validated, and folding a
60-parameter schema into this feature space would require its own design
work, not a bolt-on.

The actual held-out test file's schema (`acv_test_case.xlsx`, not yet
provided) is **unknown ahead of time**. If it turns out to be Model
A/C-shaped, `scripts/score_test_file.py` handles it today, unchanged. **If
it turns out to be Model B-shaped (like the excluded `acv_case_04.xlsx`),
this model cannot score it** — its columns won't match the fitted one-hot
vocabulary or feature list, and forcing it through would either crash (via
the `assemble_feature_columns` assertion, which is designed to fail loudly
rather than silently score garbage) or silently drop most of the real signal.
This is flagged here as a known gap to revisit if/when the test file's schema
is confirmed — not solved in this build.

### (c) Train/validation split: leave-one-case-out by file, and why

Every fold trains on the pooled rows of 4 whole files and evaluates on the
5th, entire, held-out file — never a random split across timestamps or cars.
This is the right split for this data for two reasons specific to it:

1. **The unit of prediction is a whole file** (rank all 8 cars of one train's
   multi-day log), and the unit of "newness" a real submission faces is a
   whole new file (a train/time-window never seen during training) — so the
   validation split should match that unit exactly. A random row-level split
   would let rows from the same train and nearby timestamps as a held-out row
   sit in the training set, which is not the generalization gap PS3 actually
   tests.
2. **Timestamps within one file are heavily autocorrelated** (30-second
   sampling, steady-state runs, ~90%+ exact-duplicate feature rows in the
   folds checked directly — e.g. 96% of training rows and 90% of held-out
   rows were bit-identical in feature space in the `acv_case_06.xlsx`-held-out
   fold). A random split would leak near-duplicate rows from the same
   steady-state stretch across the train/test boundary, making the model
   look far better than it would on a genuinely new file.

With only 5 labeled files, leave-one-file-out is also the only split that
uses every file as a held-out test exactly once while still training on the
other 4 — maximizing both how much of the labeled data gets evaluated and how
much is available to fit each fold, which matters given how little
independent data this is.

## Known risks (summary)

- **Model B schema gap** — see (b) above. The single biggest risk to a real
  submission if the test file's schema doesn't match Model A/C.
- **One hard case, not five easy ones** — the model never ranks
  `acv_case_05.xlsx`'s fault first; the reported average leans on 4 files
  this pipeline handles well plus one it doesn't yet solve.
- **Ties are common** on this heavily-duplicated sensor data — always check
  `n_tied_with_true_car` / `n_files_with_tied_rank` before trusting a
  specific rank, not just the mean `primary_metric`. The settled combination
  (`p90`) has zero.
- **Anomaly *counts* (`flag()`) saturate near 100% on `acv_case_06.xlsx`** —
  a file-level seasonal-baseline-shift effect (see "Anomaly counts" above),
  not a car-level signal. Don't read that count in isolation on that file;
  the ranking is much less affected.
- **Case 05's fault has a real but weak, direction-ambiguous, still-
  developing signature** — see "Why case 05 is hard to detect" above. The
  current `cross_car_temp_deviation` feature can't tell "running warm" from
  "running cold," and no feature captures a car's deviation trending upward
  over the file's multi-day window — both fixable with feature engineering,
  not attempted in this build.
- **Small n** — 5 labeled files (2 of them Model C) means every conclusion
  here is drawn from very little independent data; treat the results table
  as directional, not definitive.
