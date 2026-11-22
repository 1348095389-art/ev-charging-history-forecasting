# evfc: day-ahead station-level EV charging demand with limited station history

Code for a study of day-ahead forecasting of public EV charging demand at stations with little or no history
(manuscript for World Electric Vehicle Journal). For each forecast origin (00:00) the next 24 hourly energies (kWh) of
a station are forecast when only k days of that station's own history are available (k = 0, 1, 3, 7, 14, 28, full).
Compared methods: a pooled, history-aware gradient-boosted tree model (scikit-learn HistGradientBoosting) with
Mondrian split-conformal intervals, a global LSTM, zero-shot Chronos-2 and TimesFM 3.0 (each with and without peer
covariates), a scaled peer profile, a seasonal naive forecast, a peer-profile forecast and the zero forecast. Evaluation uses station-disjoint 5-fold cross-fitting on a 12-month test year in two cities, plus
a realism check on genuinely new stations (first 56 days after their real commissioning date).

## Data sources and licences

| City | Dataset | Licence | File expected in `data/raw/` |
|---|---|---|---|
| Boulder, CO, USA | City of Boulder Open Data, "Electric Vehicle Charging Station Data", ArcGIS Hub dataset `95992b3938be4622b07f0b05eba95d4c_0` | CC0 1.0 | `boulder_ev_sessions.csv.gz` |
| Palo Alto, CA, USA | City of Palo Alto Open Data, dataset 194693 "Electric Vehicle Charging Station Usage (July 2011 - Dec 2020)" | ODC PDDL (as marked on the portal) | `paloalto_ev_sessions.csv` |

The raw data are not redistributed in this package. `scripts/01_download.py` downloads both files from the URLs in
`evfc/download.py`; open-data portal URLs change over time, so if a download fails, fetch the CSV manually and save
it under the file name above (the Boulder file is the CSV export, gzip-compressed).
A copy of the exact raw files used in the study is distributed separately as `evfc_raw_data.zip` (both licences permit
redistribution) and is part of the archived deposit. Copy both files to `data/raw/` to skip step 01. SHA-256 of the files used:

| File | Rows | SHA-256 |
|---|---|---|
| `boulder_ev_sessions.csv.gz` | 148,136 | `7106adfcc6a26d6513482f48a1676f6b5442c81436f30459da34b6a6b911f2c8` |
| `paloalto_ev_sessions.csv` | 259,415 | `e65f5f5d3861cdd6bf3da2de97aabe590d7708e001db1532202eec52081ba82c` |

Step 02 on these files gives 29 Boulder stations (26 eligible, 12 fleet stations excluded), 33 Palo Alto stations (31 eligible),
and 8 and 25 genuinely new station candidates.

## Environment

Verified run: Python 3.11.17 on Windows (platform string Windows-10-10.0.26200), 20 CPU cores, no GPU, with the pinned versions in
`requirements.txt`: numpy 2.4.6, pandas 2.3.3, scipy 1.17.1, scikit-learn 1.9.1, joblib 1.6.0, threadpoolctl 3.7.0,
matplotlib 3.11.2, requests 2.34.2.

```
python -m venv .venv
.venv\Scripts\activate          # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

Only step 05 needs a GPU: Chronos-2 zero-shot inference with `chronos-forecasting` 2.3.2, model `amazon/chronos-2`,
and PyTorch with CUDA (the original run used an NVIDIA DGX Spark, GB10). Use a separate environment on the GPU host
(`scripts/05_chronos_infer.sh`). The precomputed Chronos-2 outputs (`chronos_out/chronos_{boulder,paloalto}_{pseudo,new}.npz`)
can be used instead of re-running step 05; step 04 checks that their row counts match the inputs. They are distributed as `evfc_chronos2_outputs.zip` (unzip into the package root; SHA-256 sums inside).

## Run order

All scripts are run from the package root and only communicate through files.

| Step | Command | Output | Measured runtime |
|---|---|---|---|
| 01 | `python scripts/01_download.py` | `data/raw/*` (skipped if present) | network dependent |
| 02 | `python scripts/02_prepare.py` | `data/*_hourly.pkl`, cleaned sessions, cleaning logs, outage shares, `data/setup.pkl` (station lists, eligibility, new-station candidates) | 31 s |
| 03 | `python scripts/03_experiment.py --n-jobs 8` | `results/{city}_pseudo_models.pkl`, `results/{city}_new_models.pkl`, `results/{city}_info.pkl` | 1234 s (Boulder 512 s + 94 s, Palo Alto 521 s + 104 s) |
| 04 | `python scripts/04_chronos_inputs.py` | `chronos_in/{city}_{pseudo,new}.npz`, `chronos_in/meta.pkl` | 6 s |
| 05 | `bash scripts/05_chronos_infer.sh` (GPU host) | `chronos_out/chronos_{city}_{pseudo,new}.npz` | GPU, not timed here |
| 06 | `python scripts/06_merge_evaluate.py` | `results/{city}_{pseudo,new}.pkl`, `tables/*.csv` (incl. `chronos_fallback_sensitivity.csv`), `tables/numbers.txt` | 261 s |
| 07 | `python scripts/07_figures.py` | `figs/fig1..fig5` (`.png`, 300 dpi, and `.pdf`) | 13 s |
| 08 | `python scripts/08_latex_tables.py` | `paper/tables/tab_*.tex` | 2 s |
| 09 (robustness, about 25 min) | `python scripts/09_seed_robustness.py --n-jobs 8` | `tables/seed_robustness.csv`, `paper/tables/tab_seed.tex` | 1498 s (seeds 1 and 2, both cities) |
| check | `python scripts/compare_outputs.py <reference_dir> [report.json]` | per-file comparison with a previous run | under 1 s |

Runtimes are wall-clock times of fresh processes on the machine above. Step 03 runs the model fits of each city in a
thread pool (`--n-jobs`, default 8); results do not depend on the number of threads. All random choices use
`seed=0` (fold assignment, sampling of the simulated history level per training row, GBDT `random_state`).

Note for sandboxed Windows hosts: if creating a `multiprocessing` thread pool fails with `PermissionError [WinError 5]`
(named pipes denied), steps 03 and 09 automatically switch joblib's threading backend, which scikit-learn uses for
feature binning, to an equivalent `concurrent.futures` pool (`evfc/_compat.py`). On normal hosts nothing changes.

## Robustness analyses

- Fold-seed robustness (step 09): the cross-fitted experiment is repeated with `seed=1` and `seed=2` (no ablations),
  which changes the assignment of test stations to folds, the sampled history levels and the selected configurations.
  For each seed, city and k the GBDT hourly RMSE and the marginal 80% conformal coverage are combined with the main
  run (seed 0) in `tables/seed_robustness.csv` (city names `Boulder`, `PaloAlto`) and `paper/tables/tab_seed.tex`.
- Chronos-2 fallback sensitivity (step 06): a Chronos-2 context with no finite value (station in outage over its
  whole window) is replaced by the peer-profile forecast. `tables/chronos_fallback_sensitivity.csv` gives, per city
  and k (k > 0, observed hours only), the share of affected rows and the RMSE of the GBDT and of the Chronos-2 mean on
  all rows and on rows without fallback.
- Days-to-par sensitivity (step 06, `tables/numbers.txt`): smallest k whose RMSE is within 1%, 2%, 5% or 10% of the
  minimum over k, for the GBDT and the Chronos-2 mean.

## Reproducibility

Re-running steps 02, 03, 04, 06, 07 and 08 in fresh processes reproduced the reference outputs of the original
analysis exactly: every `tables/*.csv` and every `paper/tables/*.tex` is byte-identical (maximum absolute numeric
difference 0), and every figure PNG is byte-identical. The figure PDFs differ only in the embedded `/CreationDate`
timestamp. Step 09 reproduced the seed-1 and seed-2 values reported in the paper to the printed precision
(3 decimals).

## Paper tables and figures

| Paper item | Script | Output file | Source table(s) |
|---|---|---|---|
| Figure 1 (protocol schematic) | 07 | `figs/fig1_protocol.{png,pdf}` | none |
| Figure 2 (station service periods, network energy) | 07 | `figs/fig2_data.{png,pdf}` | `data/*_hourly.pkl` |
| Figure 3 (RMSE and daily WAPE vs days of history) | 07 | `figs/fig3_history.{png,pdf}` | `tables/rmse_bootstrap_pseudo.csv`, `tables/point_metrics_pseudo.csv` |
| Figure 4 (interval coverage, per-station coverage at 7 days) | 07 | `figs/fig4_coverage.{png,pdf}` | `tables/interval_metrics_pseudo.csv`, `results/{city}_pseudo.pkl` |
| Figure 5 (genuinely new stations) | 07 | `figs/fig5_newstations.{png,pdf}` | `tables/rampup_new.csv`, `tables/point_metrics_new.csv`, `tables/interval_metrics_new.csv` |
| Table `tab:data` (datasets) | 08 | `paper/tables/tab_data.tex` | `tables/data_summary.csv` |
| Table `tab:rmse` (hourly RMSE) | 08 | `paper/tables/tab_rmse.tex` | `tables/point_metrics_pseudo.csv` |
| Table `tab:dwape` (station-day energy WAPE) | 08 | `paper/tables/tab_dwape.tex` | `tables/point_metrics_pseudo.csv` |
| Table `tab:mae` (hourly MAE) | 08 | `paper/tables/tab_mae.tex` | `tables/point_metrics_pseudo.csv` |
| Table `tab:tests` (paired station tests, Holm) | 08 | `paper/tables/tab_tests.tex` | `tables/tests_pseudo.csv` |
| Table `tab:cov` (coverage and width) | 08 | `paper/tables/tab_cov.tex` | `tables/interval_metrics_pseudo.csv` |
| Table `tab:ablation` (GBDT ablations) | 08 | `paper/tables/tab_ablation.tex` | `tables/ablation_pseudo.csv` (tests in `tables/tests_ablation.csv`) |
| Table `tab:new` (genuinely new stations) | 08 | `paper/tables/tab_new.tex` | `tables/point_metrics_new.csv`, `tables/interval_metrics_new.csv`, `tables/rampup_new.csv` |
| Table `tab:seed` (fold-seed robustness, appendix) | 09 | `paper/tables/tab_seed.tex` | `tables/seed_robustness.csv` |
| Chronos-2 empty-context fallback sensitivity (text) | 06 | `tables/chronos_fallback_sensitivity.csv` | `results/{city}_pseudo.pkl` |
| Numbers quoted in the text, incl. days-to-par at 1%, 2%, 5%, 10% | 06 | `tables/numbers.txt` | `tables/point_metrics_pseudo.csv` |
| Per-station RMSE (supplementary) | 06 | `tables/per_station_rmse_pseudo.csv` | `results/{city}_pseudo.pkl` |

## Package layout

- `evfc/prep.py`: session loading (city-specific parsing and station renames), cleaning, hourly allocation,
  commissioning windows and outage flags.
- `evfc/exp.py`: `City` container, features, GBDT fit and predict, baselines, Mondrian split-conformal intervals,
  metrics.
- `evfc/run.py`: station-disjoint cross-fitted experiment with inner tuning and ablations (`run_city`).
- `evfc/newsite.py`: genuinely-new-station experiment by commissioning batch (`run_city_new`).
- `evfc/evaluate.py`: point and interval tables, per-station RMSE, Wilcoxon tests with Holm adjustment and station
  bootstrap ratios.
- `evfc/pipeline.py`: glue used by the scripts (study configuration, eligibility, new-station candidates, Chronos-2
  input and merge, quantile-mean forecast, empty-context fallback, bootstrap RMSE, data summary).
- `evfc/latex.py`, `evfc/figstyle.py`: LaTeX table writers and figure style helpers.
- `evfc/chronos_infer.py`: Chronos-2 inference (GPU). `evfc/download.py`: data download.
- `evfc/verify_refs.py`, `evfc/verify_arxiv.py`: helpers used to check bibliography entries against Crossref and
  arXiv; not part of the forecasting pipeline.
- `evfc/_compat.py`: thread-pool fallback for sandboxed Windows hosts (see above).

## Revision (round 2) additions

The revision adds a global LSTM, Chronos-2 with peer covariates, TimesFM 3.0 (univariate and with peer covariates),
a scaled peer profile, a physical-plausibility cap, interval (Winkler) scores, a mature-station control for the
new-station protocol and a sensitivity run without outage masking. Steps 10 to 15 run after steps 01 to 09.

| Step | Command | Output | Notes |
|---|---|---|---|
| 10 | `python scripts/10_newsite_control.py` | `results/{city}_control.pkl` | mature stations under the new-station batch protocol |
| 11 | `python scripts/11_gpu_bundle.py` | `gpu/inputs/{city}.npz`, `gpu/row_keys.pkl` | inputs for the GPU models |
| GPU | `bash gpu/run_all.sh` (GPU host) | 14 `.npz` files: `chronos_cov_*`, `timesfm3_*`, `timesfm3_cov_*` for each city and setting, `lstm_{city}.npz` | one NVIDIA L40, about 2 h; see `gpu/README.txt` |
| 12 | `python scripts/12_merge_gpu.py <gpu_output_dir>` | `results/{city}_{pseudo,new}_r2.pkl` | merges the GPU outputs by row keys |
| 13 | `python scripts/13_evaluate_r2.py` | `tables/r2_point_capped.csv`, `r2_point_uncapped.csv`, `r2_interval_capped.csv`, `r2_tests.csv`, `r2_new_point.csv`, `r2_new_interval.csv` | about 20 min (station bootstrap) |
| 14 | `python scripts/14_report_r2.py tables <manuscript_dir>` | manuscript `tables/tab_*.tex` and `figs/fig3_history.{png,pdf}` | |
| 15 | `python scripts/15_nomask_sensitivity.py` | `../evfc_nomask/results/*` | reruns steps 02 and 03 with outage masking disabled |
| 16 | `python scripts/16_revision_extras.py results tables <manuscript_dir>` | `tables/r2_cap_shares.csv`, `r2_cap192_sensitivity.csv`, `newsite_gap_bootstrap.csv`; manuscript `tab_capshare`, `tab_cov90`, `tab_winkler_tests_full`, `tab_gap`, `tab_nomask` | about 10 min; a priori 38.4 kWh cap (two connectors at the SAE J1772 Level 2 maximum) and new-minus-control coverage gap with station and batch bootstrap |
| 17 | `python scripts/17_newsite_tests_figs.py results tables <manuscript_dir>` | `tables/r2_new_tests.csv`, manuscript `tab_new_tests`, `figs/fig4_coverage`, `figs/fig5_newstations` | paired station tests on genuinely new stations; figures with TimesFM 3.0 + peer covariates |
| 18 | `python scripts/18_typeset_tables.py <manuscript_dir>` | manuscript `tables/*.tex` edited in place | true minus signs, en dashes in day ranges, flexible float placement in the main text; appendix tables fixed in place ([H]) so the references follow the appendix |

GPU environment of the reported run: PyTorch 2.8 development build for CUDA 12.8 on one NVIDIA L40, `chronos-forecasting` 2.3.2 (`amazon/chronos-2`),
`timesfm` 3.0.2 (`google/timesfm-3.0-pytorch`, `TimesFM3Forecaster`). The TimesFM 3.0 weights are released for
non-commercial use; check the model card before other uses. The raw GPU outputs of the reported run are deposited with
the code, so steps 12 to 14 can be run without a GPU. `gpu/` scripts honour `FORCE_CPU=1` (slow) and `SMOKE=1`
(a small test run). `gpu/run_all.sh` expects `inputs/` (from step 11) next to it and writes to `$OUT`
(default `/workspace/wevj_out`, the path of the reported run; set `OUT=...` to change it).

Physical-plausibility cap (step 13, `evfc/evaluate_r2.py`): every forecast and interval bound of every method is
capped at the largest hourly energy observed at any station of the city in the training period (Boulder 14.22 kWh,
Palo Alto 16.94 kWh). The cap uses no test data. It was introduced after inspecting the foundation-model outputs,
so `r2_point_uncapped.csv` reports all point metrics without it.

Quantile means: Chronos-2 returns quantiles at 0.05, 0.1, 0.25, 0.5, 0.75, 0.9 and 0.95 (weights 0.075, 0.1, 0.2,
0.25, 0.2, 0.1, 0.075); TimesFM 3.0 returns quantiles at 0.1 to 0.9 (weights 0.15, 0.1 x 7, 0.15). Each weight is
the probability mass between the midpoints to the neighbouring levels. Rows with history but an all-missing own window
receive the peer-profile forecast and its conformal intervals, for every foundation-model variant (step 12); no covariate forecast is reported at zero history.

Holm adjustment in `r2_tests.csv`: one family for all RMSE comparisons and one for all interval-score comparisons.

Verification of the revision steps (round 3; the round-4 rerun of steps 12 to 18 after the fallback change was run the same way): running steps 12, 13, 14 and 16 in fresh processes from the deposited GPU outputs
reproduced every `tables/r2_*.csv` byte for byte and every manuscript table (18 `.tex` files) byte for byte; the
three step-16 CSVs agree to within floating-point rounding (maximum absolute difference 5e-16).

The raw GPU outputs (14 `.npz` files, 246.5 MB) are not inside this code archive; deposit them alongside it
(`wevj_out.tar.gz` with its SHA-256 file) and pass the extracted directory to step 12. SHA-256 of the archive used:
`a1f25429ba41f68cf8e09be02a01bb4e6a5ac041dbc4395dee3800b103871207`.

Reproduction without any GPU: raw data (`evfc_raw_data.zip`), steps 02 to 04, precomputed Chronos-2 outputs
(`evfc_chronos2_outputs.zip`), step 06, then the GPU outputs (`wevj_out.tar.gz`) and steps 12 to 18.



## Licence

Code: MIT (see `LICENSE`). Data: see the licences of the two source datasets above.

## Citation

If you use this code or data in your work, please cite both the manuscript and this archived software:

* **Software (Zenodo):** [10.5281/zenodo.24431971](https://doi.org/10.5281/zenodo.24431971) (v1.0.0)
* **All versions:** [10.5281/zenodo.24431970](https://doi.org/10.5281/zenodo.24431970)
