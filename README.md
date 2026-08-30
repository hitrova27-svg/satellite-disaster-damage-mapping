# Mapping Natural-Disaster Damage from Satellite Imagery

**Cross-disaster transfer on xBD: pretrained versus from-scratch change-detection CNNs**

Artur Zavistovskyi · Bhuvanesh Dinesh Wadhwani · Anastasiia Khitrova · Yitian Zhou

*Deep Learning for Social Sciences*
Department of Politics and Public Administration, University of Konstanz
Summer Semester 2026 — Final Project

---

## 1. Overview

This repository contains the complete implementation, data-preparation pipeline and analysis code
for a study of automated building-damage classification on the xBD / xView2 dataset. The project
asks three questions, corresponding to the three parts of the assignment:

1. **(Part 1)** How are damage classes distributed across disaster types, and what does the
   distribution imply for the cross-disaster transfer problem?
2. **(Part 2)** How much of a change-detection model's performance survives evaluation on a disaster
   type withheld entirely from training, and does ImageNet pretraining improve that transfer
   relative to a comparably designed encoder trained from scratch?
3. **(Part 3)** Are damage and recovery distributed unequally across communities, and what changes
   if a relief programme is keyed to a model's *predicted* damage map rather than to verified
   labels?

The task is framed as building-level change detection. Given two co-registered 64×64 crops of the
same structure, one before and one after the event, a shared-weight siamese encoder produces
`[f_pre, f_post, |f_pre − f_post|]`, and a small head predicts `no-damage`, `minor-damage`,
`major-damage` or `destroyed`.

The experimental design deliberately avoids a random split. Five wildfire events
(`pinery-bushfire`, `portugal-wildfire`, `santa-rosa-wildfire`, `socal-fire`, `woolsey-fire`) are
withheld from training entirely and used as a zero-shot out-of-domain test set, while the official
xView2 `test` and `hold` directories provide an in-distribution reference measured on the same
trained model.

The written report is `report.pdf`. Every figure, table and number in it is generated from a CSV or
PNG artefact written by the notebooks and scripts documented below; nothing is hand-computed.

---

## 2. Results at a glance

Both models are trained on identical patches, splits, class weights and evaluation code, and differ
only in the encoder, which isolates the effect of pretraining from every other design choice.

| Model | Parameters | Epochs run (best) | `val` | `test_id` (seen types) | `test_ood` (wildfire, unseen) |
|---|---:|---:|---|---|---|
| **A** — Siamese ResNet-18, ImageNet-pretrained | 11,571,524 | 43 (35) | 0.830 / 0.666 / 0.910 | 0.808 / 0.650 / 0.908 | 0.774 / **0.429** / 0.814 |
| **B** — Compact Siamese CNN, from scratch | 1,371,108 | 16 (8) | 0.763 / 0.597 / 0.902 | 0.768 / 0.615 / 0.903 | 0.630 / **0.377** / 0.790 |

Cells report *accuracy / macro-F1 / macro ROC-AUC*. Source: [`results/model_comparison.csv`](results/model_comparison.csv).

Three findings summarise the study:

- **Transfer fails as calibration, not as representation.** Model A loses 33.9% of its macro-F1 out
  of domain, but the loss is confined to the two intermediate damage classes, which make up 19.1% of
  training patches and only 1.7% of wildfire patches. `destroyed` actually scores *higher* out of
  domain than in it (F1 0.758 vs. 0.687), `no-damage` precision reaches 0.966, and threshold-free
  ROC-AUC holds at 0.814.
- **Pretraining wins on every split and every metric**, with the largest margin precisely on the
  unseen disaster type: +0.143 accuracy and +0.052 macro-F1 on wildfire.
- **Damage is not distributed along an income gradient once the disaster event is controlled for**
  (β = +0.225, p = 0.0009 naive; β = +0.057, p = 0.44 with event fixed effects), **but recovery
  capacity is.** Priced in dollars against HAZUS loss ratios and validated against NFIP, CAL FIRE
  and FEMA records, the same damage map is roughly 7× steeper across income, and 61% of the poorest
  quartile's flood loss is covered by neither insurance nor federal aid, against 34% in the richest.
  Auditing the two models' own predictions shows that an automated relief rule would misdirect
  $30–66M of every $1B on seen disaster types and $107–188M on wildfire, the type the models never
  trained on.

---

## 3. Repository layout

```text
.
├── notebooks/                          Parts 1 and 2
│   ├── data_cleaning_eda.ipynb           EDA, cleaning ledger, patch extraction
│   ├── pretrained_resnet18_train.ipynb   Model A: ImageNet ResNet-18, fine-tuned
│   ├── train_scratch_cnn.ipynb           Model B: compact CNN, random initialisation
│   └── socio_analysis.ipynb              Part 3 baseline, earlier in-place version
├── socio_analysis_final/               Part 3, final version
│   ├── 1_baseline_socio_analysis.ipynb   damage vs. income, block-group regression
│   ├── 2_damage_cost_analysis.ipynb      damage in dollars, recovery, model audit
│   ├── fetch_open_data.py                re-downloads the federal records
│   ├── data/                             every input, already downloaded
│   ├── outputs/                          every figure and table, already produced
│   └── README.md                         section-by-section guide to both notebooks
├── scripts/                            data acquisition and split construction
│   ├── download_xbd.py                   resumable ranged downloader for xBD
│   ├── download_acs.py                   ACS 2017 5-year block-group extract
│   ├── download_tiger.py                 TIGER/Line 2020 block-group boundaries
│   ├── prepare_splits.py                 scene-level split, leakage and integrity checks
│   └── setup_windows.ps1                 end-to-end Windows bootstrap
├── data/                               committed definitions, no imagery
│   ├── splits.csv                        77,238 file rows → scene, event, origin, split
│   ├── splits_summary.csv                11,034 scene rows, for split inspection
│   ├── acs.csv                           ACS 2017 block-group socioeconomics
│   └── hazard_exposure.csv               hazard-exposure proxy (unpopulated, see §9)
├── results/                            committed metrics, tables and plots
│   ├── eda/                              Part 1 artefacts
│   ├── pretrained_resnet18/              Model A metrics and curves
│   ├── scratch_cnn/                      Model B metrics, curves, confusion matrices
│   ├── socio_analysis/                   Part 3 baseline artefacts
│   └── model_comparison.csv              the single two-model summary table
├── docs/                               specification and long-form guides
├── figures/                            figures used by the LaTeX report
├── report.tex / report.pdf             the written report
├── requirements.txt                    Python dependencies (see §4)
├── README.md                           this file
└── LICENSE                             MIT
```

---

## 4. Environment

### 4.1 Python

Python 3.12 or newer. From the repository root:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip wheel setuptools
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m ipykernel install --user --name xbd-local
```

On Linux or macOS, substitute `python3 -m venv .venv` and `.venv/bin/python`.

### 4.2 GPU

`requirements.txt` installs the CPU build of PyTorch, which runs every notebook but trains the two
CNNs slowly. For GPU training, reinstall the two packages from the PyTorch index matching the
installed CUDA runtime:

```powershell
.\.venv\Scripts\python.exe -m pip install --force-reinstall torch torchvision `
  --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Parts 1 and 3 need neither `torch` nor a GPU.

### 4.3 Reference hardware

The committed results were produced on a Windows laptop with 16 GB RAM and an **NVIDIA GeForce
RTX 4070 Laptop GPU (8 GB VRAM)**, recorded in
[`results/scratch_cnn/run_config.json`](results/scratch_cnn/run_config.json). Batch size 128 and
mixed precision are sized for that VRAM budget; reduce `BATCH_SIZE` in the configuration cell on a
smaller card.

---

## 5. Data

### 5.1 xBD imagery

The full xBD release is **not committed**: 11,034 scenes over 19 events, each contributing a
1024×1024 RGB pre-disaster tile, the matching post-disaster tile, and a GeoJSON label file —
22,068 images and 425,368 annotated building polygons.

Obtain it from [xview2.org](https://xview2.org/) or, as used here, the public Kaggle mirror
`qianlanzz/xbd-dataset`. Place the Kaggle API token at `%USERPROFILE%\.kaggle\kaggle.json`, then:

```powershell
.\.venv\Scripts\python.exe .\scripts\download_xbd.py --parts 8 --workers 1
tar.exe -xf .\xbd-dataset.zip
```

The downloader is resumable: re-run the same command after an interruption and completed byte
ranges in `.download-parts\` are reused. The expected layout after extraction is:

```text
<repository root>/
    data/splits.csv
    xbd/{hold,test,tier1,tier3}/{images,labels,masks}/
```

`masks/` (called `targets/` in the xView2 tars) holds *segmentation* labels. This per-building
*classification* task never reads it.

Validate the data and the split assignment without overwriting the committed CSVs:

```powershell
.\.venv\Scripts\python.exe .\scripts\prepare_splits.py --src .\xbd --out .\data --dry-run
```

### 5.2 The experimental split

The split is **scene-level** and is produced by `scripts/prepare_splits.py`, not inside a training
notebook, so every model in the project trains on an identical partition. Scene-level assignment
guarantees that the pre- and post-disaster image of a location cannot separate and that no building
can appear in two splits.

| Split | Rule | Scenes | Polygons |
|---|---|---:|---:|
| `train` | tier1 + tier3, non-fire | 3,135 | 225,506 |
| `val` | tier1 + tier3, non-fire | 392 | 30,794 |
| `test_id` | official test + hold, non-fire | 1,135 | 91,755 |
| `test_ood` | wildfire, all origins | 6,372 | 77,313 |
| **Total** | | **11,034** | **425,368** |

Only `train` and `val` depend on the random seed; both test sets are defined by the data itself. An
automated check confirms zero wildfire scenes in `train` or `val`, and zero event overlap between
the training pool and `test_ood`.

### 5.3 Socioeconomic and federal records

Already committed, so Part 3 runs without any download:

| File | Source |
|---|---|
| `data/acs.csv` | American Community Survey 2017 5-year estimates, block-group level |
| `socio_analysis_final/data/buildings.csv`, `tracts.csv` | xBD labels joined to census block groups and ACS |
| `socio_analysis_final/data/nfip_claims.csv` | OpenFEMA NFIP redacted claims, filtered to the analysis block groups |
| `socio_analysis_final/data/fema_registrations.csv` | OpenFEMA Individuals and Households Program valid registrations |
| `socio_analysis_final/data/fema_disaster_totals.csv`, `fema_declared_counties.csv` | OpenFEMA declaration summaries |
| `socio_analysis_final/data/calfire_dins.csv` | CAL FIRE Damage Inspection (DINS): Tubbs, Thomas, Woolsey |
| `socio_analysis_final/data/noaa_event_costs.csv` | NOAA NCEI billion-dollar disaster events |
| `socio_analysis_final/data/predictions.csv` | the two Part 2 CNNs, evaluated on their held-out splits |

To re-download the federal records (no API key required):

```powershell
.\.venv\Scripts\python.exe .\socio_analysis_final\fetch_open_data.py
```

Two inputs must be regenerated if the *baseline* Part 3 notebook is re-run from raw geometry:

```powershell
.\.venv\Scripts\python.exe .\scripts\download_tiger.py      # → data/census_tracts.geojson
$env:CENSUS_API_KEY = "<key>"
.\.venv\Scripts\python.exe .\scripts\download_acs.py        # → data/acs.csv (already committed)
```

---

## 6. Running the experiments

Start JupyterLab **from the repository root** so that relative paths resolve:

```powershell
.\.venv\Scripts\jupyter-lab.exe
```

Every notebook resolves the project root itself and writes only into its own output directory, so
the four steps below are the entire pipeline. Run them in order; both training notebooks consume the
patch arrays written in step 6.1.

### 6.1 Part 1 — EDA, cleaning and patch extraction

**Notebook:** [`notebooks/data_cleaning_eda.ipynb`](notebooks/data_cleaning_eda.ipynb)
**Requires:** `xbd/`, `data/splits.csv`
**Writes:** `results/eda/` and `work-pretrained/patches/{pre.npy, post.npy, manifest.csv}`

Sections 1–6 perform the split check, the full 425,368-polygon label scan, the exploratory analysis
and the cleaning ledger. Nothing is removed silently: every rule states its threshold, its cost and
its per-split removal rate, so a rule that quietly favours one split can be caught. Two rules cost
data — the `un-classified` subtype (3.29% of polygons) and sub-8-pixel footprints (3.51% of the
remainder) — leaving 396,899 usable polygons, 93.3% of the original. Neither rate is concentrated in
`test_ood`, so neither flatters the transfer result.

Section 7 extracts the patch arrays. Scenes are sampled under a compute budget
(`SCENE_BUDGET = {train: 800, val: 200, test_id: 300, test_ood: 400}`), restricted to scenes with at
least `MIN_BUILDINGS_PER_SCENE = 5` usable buildings, and cropped at `PATCH_SIZE = 64` with
`BBOX_PAD = 10`. This yields **135,262 patch pairs from 1,698 scenes**:

| Split | `no-damage` | `minor` | `major` | `destroyed` | Total |
|---|---:|---:|---:|---:|---:|
| `train` | 55,284 | 7,661 | 6,265 | 3,706 | 72,916 |
| `val` | 15,701 | 2,571 | 1,164 | 1,010 | 20,446 |
| `test_id` | 20,500 | 2,840 | 2,512 | 1,421 | 27,273 |
| `test_ood` | 12,056 | 116 | 138 | 2,317 | 14,627 |

Patches are written to memory-mapped `uint8` arrays (≈3.3 GB), which keeps the working set on disk
and the loader fast. The expensive stages — the label scan, the image-quality sample, the occlusion
statistics and the perceptual hashes — are cached under `results/eda/cache/`, so a re-run skips
them. Delete the corresponding cache file to force a re-computation.

> The standalone notebooks `notebooks/01_eda.ipynb` and `notebooks/02_data_split.ipynb` cover the
> same sections without requiring `torch`, and write to `results/01_eda/` and
> `results/02_data_split/`. They are useful for inspecting Part 1 in isolation.

### 6.2 Part 2 — the two change-detection CNNs

Both notebooks consume `work-pretrained/patches/` and fail fast with an explicit message if it is
absent. Both use AdamW with cosine annealing, weight decay 1e-4, weighted cross-entropy with
inverse-frequency class weights computed on `train` only (0.330 / 2.379 / 2.910 / 4.919), batch
size 128, mixed precision, **seed 42**, early stopping on validation macro-F1 with patience 8, and
checkpointing on the same criterion.

Macro-F1 rather than loss is the stopping criterion by design: under a weighted loss on imbalanced
data, validation loss bottoms out at epoch 11 and then climbs while macro-F1 keeps improving to
epoch 35. The same decoupling appears in both models, so it is a property of the objective rather
than of one run.

**Model A** — [`notebooks/pretrained_resnet18_train.ipynb`](notebooks/pretrained_resnet18_train.ipynb)
→ writes `work-pretrained/` (checkpoints, metrics, curves)

A linear-probe-then-fine-tune schedule: phase 1 trains only the head on a frozen ImageNet ResNet-18
backbone for 5 epochs at 1e-3, so that the randomly initialised head does not push large gradients
into good weights; phase 2 unfreezes `layer3` and `layer4` for up to 45 further epochs at 1e-4.
Unfreezing lifts validation macro-F1 from 0.499 to 0.580 in a single epoch — the clearest evidence
in the project that a frozen ImageNet backbone is not sufficient for satellite imagery.

**Model B** — [`notebooks/train_scratch_cnn.ipynb`](notebooks/train_scratch_cnn.ipynb)
→ writes `work-scratch/` (checkpoints, metrics, curves, confusion matrices)

Four blocks of two 3×3 convolutions with BatchNorm and ReLU followed by 2×2 max-pooling, at channel
widths 32 → 64 → 128 → 256, then global average pooling. Everything downstream of the encoder is
identical to Model A. The whole network trains from step one at 1e-3 over the same 50-epoch budget.

Two switches in the configuration cell:

- `RESUME = True` — continue after an interrupted full run.
- `SMOKE_TEST = True` — exercise checkpointing and evaluation on a few batches without overwriting
  full-run artefacts.

Copy the final metric CSVs from `work-pretrained/` and `work-scratch/` into
`results/pretrained_resnet18/` and `results/scratch_cnn/` to refresh the committed tables; the
`work-*` directories themselves are gitignored.

### 6.3 Part 3 — impact and recovery analysis

**Baseline** — [`socio_analysis_final/1_baseline_socio_analysis.ipynb`](socio_analysis_final/1_baseline_socio_analysis.ipynb)
**Run from:** the repository root
**Requires:** `xbd/`, `data/splits.csv`, `data/acs.csv`, `data/census_tracts.geojson`, `geopandas`
**Writes:** `results/socio_analysis/`

Takes the per-building WGS84 centroids in the xBD labels, restricts to the ten US events for which
census geography is available, and spatially joins them to 2020 TIGER block groups. Each building
contributes an ordinal damage score (`no-damage` = 0 … `destroyed` = 3); the unit of analysis is the
block-group mean, and block groups with fewer than 20 buildings are dropped. That gives **185,045
buildings in 308 block groups**, joined to ACS 2017 covariates.

**Damage cost, recovery and model audit** — [`socio_analysis_final/2_damage_cost_analysis.ipynb`](socio_analysis_final/2_damage_cost_analysis.ipynb)
**Run from:** `socio_analysis_final/`
**Requires:** CSVs only — no imagery, no GPU, no model training
**Writes:** `socio_analysis_final/outputs/damage_cost/`

```powershell
cd socio_analysis_final
..\.venv\Scripts\jupyter-lab.exe 2_damage_cost_analysis.ipynb   # run all, about ten seconds
```

Ten self-contained sections: pricing every building's damage class in dollars via the HAZUS-MH
per-damage-state loss ratios (0% / 5% / 30% / 100%) applied to its block group's ACS median home
value (§3); three independent price checks, against 6,525 NFIP settlements, 9,395 CAL FIRE post-fire
structure inspections and 133,240 FEMA household loss verifications (§4–5); damage and cost across
income (§6); the repair bill split into insurance, federal aid and nothing (§7); insurance take-up
across all four disaster types (§8); and the relief-allocation audit of the two Part 2 models'
predicted damage maps (§9).

The notebook drops the two degenerate ACS columns that the baseline output still carries —
`pov_rate_2017`, which is empty for every block group, and `exposure_proxy`, which is identically
zero — and rebuilds its own sample of 291 clean block groups after removing the 17 that carry the
Census sentinel value `-666666666`.

### 6.4 Non-interactive execution

Each notebook can be executed headlessly, which is the form used to produce the committed artefacts:

```powershell
.\.venv\Scripts\jupyter.exe nbconvert --to notebook --execute `
  .\notebooks\pretrained_resnet18_train.ipynb `
  --output pretrained_resnet18_train.executed.ipynb `
  --output-dir .\notebooks `
  --ExecutePreprocessor.kernel_name=xbd-local `
  --ExecutePreprocessor.timeout=-1
```

Substitute the notebook name for the other steps.

### 6.5 Cost of a full run

| Step | Dominant cost | Disk | Wall clock |
|---|---|---|---|
| §6.1 EDA and patch extraction | decoding ~3,400 1024×1024 PNGs | ≈3.3 GB of patches | dominated by extraction; all diagnostics cached |
| §6.2 Model A | 43 epochs, 11.57M parameters | ≈3.5 GB under `work-pretrained/` | not recorded |
| §6.2 Model B | 16 epochs, 1.37M parameters | ≈50 MB under `work-scratch/` | **14.2 min**, 53 s/epoch |
| §6.3 baseline | spatial join of 185k centroids | < 50 MB | minutes |
| §6.3 damage cost | CSV arithmetic only | < 5 MB | **≈10 s** |

---

## 7. Where every reported number comes from

| Report item | Artefact |
|---|---|
| Table 1 — split design | `data/splits_summary.csv` |
| Table 2 — cleaning ledger | `results/eda/exclusion_ledger.csv`, `cleaning_summary.csv` |
| Table 3 — patch counts | `results/02_data_split/patch_class_balance.csv` |
| Table 4 — headline results | `results/model_comparison.csv` |
| Table 5 — block-group OLS | re-run regression on the 291 clean block groups (see §8) |
| Fig. 1 — class distribution per split | `results/01_eda/damage_class_per_split.png` |
| Fig. 2 — composition per event | `results/01_eda/damage_composition_per_event_heatmap.png` |
| Fig. 3 — scene occupancy | `results/01_eda/empty_scenes_per_event.png` |
| Fig. 4 — representative patches | `results/01_eda/training_examples_by_class.png`, `ood_examples_by_class.png` |
| Fig. 5 — Model A training history | `results/pretrained_resnet18/training_curves.png` |
| Fig. 6 — per-class F1 | both `results_per_class.csv` files |
| Fig. 7 — out-of-domain prior shift | `results_per_class.csv` + `results/scratch_cnn/confusion_matrices.csv` |
| Fig. 8 — confusion matrices, Model B | `results/scratch_cnn/confusion_matrices.csv` |
| Fig. 9 — damage vs. income | `results/socio_analysis/tracts.csv`, cleaned |
| Fig. 10 — standardised coefficients | re-run regression (see §8) |
| Fig. 11 — damage share vs. dollar loss | `socio_analysis_final/outputs/damage_cost/damage_vs_dollars.png`, `income_quintiles.csv` |
| Fig. 12 — who pays the repair bill | `socio_analysis_final/outputs/damage_cost/who_pays.png`, `who_pays.csv` |
| Fig. 13 — relief misallocation | `socio_analysis_final/outputs/damage_cost/model_misallocation.png`, `allocation_audit.csv` |
| Price checks (report §8.3) | `price_check_vs_nfip.csv`, `price_check_vs_calfire.csv`, `price_check_vs_fema.csv` |
| Insurance take-up (report §8.4) | `insurance_by_income_band.csv`, `insurance_by_disaster_type.csv` |

---

## 8. Reproducibility statement

- **Seed 42 throughout.** Both training runs, the scene sampling and the split construction are
  seeded; `set_seed` covers `random`, `numpy`, `torch` and CUDA.
- **The split is constructed once, outside the notebooks**, by `scripts/prepare_splits.py`, so both
  models and the Part 3 join operate on a byte-identical partition.
- **Model selection uses `val` only**, never `test_ood`; selecting on the out-of-domain set would
  leak the test set.
- **Caches are keyed on dataset coverage**, so a scan over a partially extracted dataset can never
  be silently reused as if it were the full one.

Three caveats a reader should know about the committed artefacts:

1. The event-fixed-effects specification reported in Table 5 and Figure 10 is a **re-run** on the 291
   clean block groups. `results/socio_analysis/regression_results.csv`, produced by the earlier
   in-place notebook, is superseded and should not be read as a result: it never filtered the Census
   sentinel values, it reports `pov_rate_2017` with a coefficient of 1.4e-17 and p = 0.0 — a
   rank-deficiency artefact of the pseudo-inverse fallback rather than a finding — and its
   `exposure_proxy` control is identically zero.
2. `socio_analysis_final/outputs/damage_cost/` was regenerated after the report PDF was compiled, so
   a few Part 3 totals in the report are one execution behind the artefacts. Total modelled loss is
   $7.01B in the current artefacts against $6.76B quoted in the report, over 110,147 buildings
   against 108,491; the relief-misallocation range is $30–66M / $107–188M per $1B against
   $30–78M / $112–200M in the report. The disaster-type shares, the three price checks, the income
   gradients and every qualitative conclusion are unchanged. Where the two disagree, the CSVs under
   `socio_analysis_final/outputs/damage_cost/` are authoritative.
3. The model checkpoints in `checkpoints/` do **not** correspond to the run in `results/`. Their
   architectures differ from the reported ones and they predict a single class for everything. They
   cannot be used to reproduce or extend the reported numbers; re-run §6.2 instead.

---

## 9. Known limitations

Stated in full in Section 9 of the report. In brief:

- **Thin out-of-domain support.** `minor` and `major` rest on 116 and 138 true examples in
  `test_ood`. These two classes drive the macro-F1 story, so their per-class figures carry wide
  uncertainty.
- **Confounded acquisition geometry.** Sun elevation differs sharply between domains (64.4° ± 7.5
  non-fire against 55.0° ± 15.9 wildfire), but every scene within an event shares a single
  acquisition, so the effective sample size is 19 events, not 11,034 scenes. Part of the transfer gap
  may be attributable to shadow length rather than to fire damage, and we cannot bound how much.
- **Not a pure pretraining ablation.** The two encoders differ in initialisation *and* in size
  (11.57M against 1.37M parameters). A randomly initialised ResNet-18 would separate the two effects.
- **Annotator selection.** `un-classified` polygons are excluded, so true zero-shot difficulty is
  somewhat higher than reported.
- **Ecological inference in Part 3.** Block-group averages cannot support claims about individual
  households; no true hazard-exposure layer was available (`data/hazard_exposure.csv` is
  unpopulated, and event fixed effects stand in for it); and ten events is a small basis for fixed
  effects.
- **Recovery-analysis assumptions.** One median home value stands in for every building in a block
  group, and four HAZUS loss ratios stand in for a continuous damage distribution. Tornado has no
  inspection dataset comparable to CAL FIRE's, so its loss ratios rest on the HAZUS assumption alone.
  The predicted-damage audit covers seven of the ten events, because all three tornado events sit
  entirely inside the Part 2 training split and so have no held-out predicted map.

---

## 10. References

1. R. Gupta et al., *xBD: A Dataset for Assessing Building Damage from Satellite Imagery*,
   arXiv:1911.09296, 2019.
2. E. Weber and H. Kané, *Building Disaster Damage Assessment in Satellite Imagery with
   Multi-Temporal Fusion*, arXiv:2004.05525, 2020.
3. K. He, X. Zhang, S. Ren and J. Sun, *Deep Residual Learning for Image Recognition*, CVPR, 2016.
4. J. Deng et al., *ImageNet: A Large-Scale Hierarchical Image Database*, CVPR, 2009.
5. A. Kumar, A. Raghunathan, R. Jones, T. Ma and P. Liang, *Fine-Tuning can Distort Pretrained
   Features and Underperform Out-of-Distribution*, ICLR, 2022.
6. M. Saerens, P. Latinne and C. Decaestecker, *Adjusting the Outputs of a Classifier to New a
   Priori Probabilities*, Neural Computation 14(1):21–41, 2002.
7. A. Paszke et al., *PyTorch: An Imperative Style, High-Performance Deep Learning Library*,
   NeurIPS, 2019.
8. U.S. Census Bureau, *American Community Survey 5-Year Estimates (2017)* and *TIGER/Line
   block-group boundaries (2020)*.
9. xView2 Challenge, https://xview2.org.
10. FEMA, *Hazus Multi-Hazard Loss Estimation Methodology (Hazus-MH), Technical Manual*.
11. FEMA, OpenFEMA API: *FIMA NFIP Redacted Claims*, *Individuals and Households Program Valid
    Registrations*, and disaster declaration summaries,
    https://www.fema.gov/about/openfema/data-sets.
12. CAL FIRE, *Damage Inspection (DINS) Data*,
    https://www.fire.ca.gov/programs/fire-prevention/wildfire-preparedness/damage-inspection-dins-data.

---

## 11. License

Released under the MIT License; see [`LICENSE`](LICENSE). The xBD dataset is subject to its own
terms and is not redistributed here. ACS, TIGER/Line, OpenFEMA, NOAA NCEI and CAL FIRE records are
public-domain U.S. government works.
