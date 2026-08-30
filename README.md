# Mapping Natural-Disaster Damage from Satellite Imagery

**Cross-disaster transfer on xBD: pretrained versus from-scratch change-detection CNNs**

Artur Zavistovskyi · Bhuvanesh Dinesh Wadhwani · Anastasiia Khitrova · Yitian Zhou

*Deep Learning for Social Sciences*
Department of Politics and Public Administration, University of Konstanz
Summer Semester 2026 — Final Project

---

## 1. Overview

This repository contains the complete implementation, data-preparation pipeline and analysis code
for a study of automated building-damage classification on the xBD / xView2 dataset, addressing the
three parts of the assignment:

1. **(Part 1)** How are damage classes distributed across disaster types?
2. **(Part 2)** How much of a change-detection model's performance survives evaluation on a disaster
   type withheld entirely from training, and does ImageNet pretraining improve that transfer
   relative to a comparably designed encoder trained from scratch?
3. **(Part 3)** Are damage and recovery distributed unequally across communities, and what changes
   if a relief programme is keyed to a model's *predicted* damage map rather than to verified
   labels?

The task is framed as building-level change detection. Given two co-registered 64×64 crops of the
same structure, one before and one after the event, a shared-weight siamese encoder produces
`[f_pre, f_post, |f_pre − f_post|]`, and a small head predicts `no-damage`, `minor-damage`,
`major-damage` or `destroyed`. Five wildfire events are withheld from training entirely and used as
a zero-shot out-of-domain test set; the official xView2 `test`/`hold` directories give an
in-distribution reference on the same trained model.

The written report is `report.pdf`. Its figures, tables and numbers come from the CSV and PNG
artefacts written by the notebooks and scripts below.

---

## 2. How to Run

### 2.1 Environment

Python 3.12 or newer. From the repository root:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip wheel setuptools
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m ipykernel install --user --name xbd-local
```

`requirements.txt` installs the CPU build of PyTorch, which runs every notebook but trains the two
CNNs slowly. For GPU training, reinstall from the PyTorch index matching your CUDA runtime, e.g.:

```powershell
.\.venv\Scripts\python.exe -m pip install --force-reinstall torch torchvision `
  --index-url https://download.pytorch.org/whl/cu128
```

Parts 1 and 3 need neither `torch` nor a GPU. The committed results were produced on an NVIDIA
RTX 4070 Laptop GPU (8 GB VRAM); reduce `BATCH_SIZE` in the configuration cell on a smaller card.

### 2.2 Data

**xBD imagery** (not committed — 11,034 scenes, 22,068 images, 425,368 building polygons): obtain
from [xview2.org](https://xview2.org/) or the public Kaggle mirror `qianlanzz/xbd-dataset`, then:

```powershell
.\.venv\Scripts\python.exe .\scripts\download_xbd.py --parts 8 --workers 1
tar.exe -xf .\xbd-dataset.zip
```

giving `xbd/{hold,test,tier1,tier3}/{images,labels,masks}/` alongside `data/splits.csv`.

**Git LFS is required.** `data/splits.csv`, `data/acs.csv`, `data/census_tracts.geojson`,
`results/socio_analysis/buildings.csv` and the other paths listed in `.gitattributes` are stored via
LFS; after cloning, run `git lfs install && git lfs pull` or they will be pointer stubs.

**The split** is scene-level, produced once by `scripts/prepare_splits.py` (not inside a notebook),
so every model trains on an identical, seeded partition. Five wildfire events are held out entirely
(`test_ood`); the official xView2 `test`+`hold` directories give the in-distribution reference
(`test_id`).

**Socioeconomic and federal records** for Part 3 are committed (ACS income/housing data and the
CAL FIRE / NFIP / FEMA records used to price and validate damage in dollars). Two inputs are not:
the baseline notebook additionally needs the xBD label files and `data/census_tracts.geojson`, so
run `.\.venv\Scripts\python.exe .\scripts\download_tiger.py` first. To refresh the CAL FIRE and
FEMA files from source: `.\.venv\Scripts\python.exe .\scripts\download_socio_extras.py`.

### 2.3 Running the experiments

Start JupyterLab from the repository root — `.\.venv\Scripts\jupyter-lab.exe` — then run the
notebooks in order. Each resolves its own project root and writes only into its own output
directory; both training notebooks consume the patches written in step 2.3.1.

Two manual steps join the stages, because the notebooks do not agree on paths. The training
notebooks write to `work-pretrained/` and `work-scratch/` and the cost notebook to
`outputs/damage_cost/`, while the committed copies live under `results/`. And the cost notebook
reads `data/buildings.csv`, `data/tracts.csv` and `data/predictions.csv`: the first two must be
copied there from `results/socio_analysis/`, and `predictions.csv` (the two Part 2 models'
building-level output on their held-out splits, joined to census geography) is not written by any
notebook and must be produced separately.

**2.3.1 Part 1** — [`notebooks/data_cleaning_eda.ipynb`](notebooks/data_cleaning_eda.ipynb)

EDA, the full label scan, and a cleaning ledger where every rule states its threshold, cost and
per-split removal rate (nothing is removed silently). Extracts 135,262 patch pairs from 1,698
scenes into memory-mapped arrays under `work-pretrained/patches/`, sampled under a compute budget
and restricted to scenes with at least 5 usable buildings.

**2.3.2 Part 2** — the two change-detection CNNs

Both consume `work-pretrained/patches/`, use AdamW, weighted cross-entropy on inverse-frequency
class weights, batch size 128, mixed precision, **seed 42**, and early stopping on validation
macro-F1, not loss — under class imbalance the two decouple: Model A's validation loss bottoms out
at epoch 11 and rises while macro-F1 keeps improving to epoch 35, and Model B's loss is still
falling at its final epoch while macro-F1 peaked at epoch 8.

- **Model A** — [`notebooks/pretrained_resnet18_train.ipynb`](notebooks/pretrained_resnet18_train.ipynb): ImageNet ResNet-18, linear-probe then fine-tune.
- **Model B** — [`notebooks/train_scratch_cnn.ipynb`](notebooks/train_scratch_cnn.ipynb): compact CNN trained from scratch, same schedule budget.

To exercise the pipeline without a full run, point `OUTPUT_DIR_OVERRIDE` in the configuration cell
at a scratch directory and lower the epoch budget (`EPOCHS_HEAD`/`EPOCHS_FT` in Model A, `EPOCHS` in
Model B), so full-run artefacts are not overwritten.

**2.3.3 Part 3** — impact and recovery analysis

- **Baseline** — [`notebooks/socio_analysis.ipynb`](notebooks/socio_analysis.ipynb): joins 185,045 xBD buildings to census block groups and ACS covariates, then fits the income/damage regression on the 291 block groups (108,491 buildings) left after the 20-building threshold and the census-sentinel drop.
- **Damage cost, recovery and audit** — [`notebooks/damage_cost_analysis.ipynb`](notebooks/damage_cost_analysis.ipynb): prices damage in dollars via HAZUS loss ratios, validates against NFIP/CAL FIRE/FEMA records, and audits the Part 2 models' predicted damage maps for relief-allocation bias.

---

## 3. Repository structure

```text
.
├── notebooks/                     Parts 1, 2 and 3
│   ├── data_cleaning_eda.ipynb      Part 1: EDA, cleaning ledger, patch extraction
│   ├── pretrained_resnet18_train.ipynb   Part 2, Model A: ImageNet ResNet-18
│   ├── train_scratch_cnn.ipynb     Part 2, Model B: compact CNN from scratch
│   ├── socio_analysis.ipynb        Part 3: damage vs. income regression
│   └── damage_cost_analysis.ipynb  Part 3: damage in dollars, recovery, model audit
├── scripts/                      data acquisition and split construction
│   ├── download_xbd.py             resumable xBD downloader
│   ├── download_acs.py             ACS 2017 block-group extract
│   ├── download_tiger.py           TIGER/Line block-group boundaries
│   ├── download_socio_extras.py    CAL FIRE DINS and FEMA IHP records
│   ├── prepare_splits.py           scene-level split + integrity checks
│   └── setup_windows.ps1           end-to-end Windows bootstrap
├── data/                          committed inputs, no imagery
│   ├── splits.csv, splits_summary.csv
│   ├── acs.csv, hazard_exposure.csv
│   └── calfire_dins.csv, fema_*.csv, nfip_claims.csv, noaa_event_costs.csv
├── results/                       committed metrics, tables and plots
│   ├── eda/                        Part 1 artefacts
│   ├── pretrained_resnet18/        Model A metrics and curves
│   ├── scratch_cnn/                Model B metrics, curves, confusion matrices
│   ├── socio_analysis/             Part 3 baseline artefacts
│   ├── damage_cost_analysis/       Part 3 cost/recovery/audit artefacts
│   └── model_comparison.csv        the single two-model summary table
├── figures/                       figures used by the LaTeX report
├── docs/                          specification and long-form guides
├── report.tex / report.pdf        the written report
├── requirements.txt               Python dependencies
└── LICENSE                        MIT
```

---

## 4. Results at a glance

| Model | Parameters | `val` | `test_id` (seen types) | `test_ood` (wildfire, unseen) |
|---|---:|---|---|---|
| **A** — Siamese ResNet-18, ImageNet-pretrained | 11.57M | 0.830 / 0.666 / 0.910 | 0.808 / 0.650 / 0.908 | 0.774 / **0.429** / 0.814 |
| **B** — Compact Siamese CNN, from scratch | 1.37M | 0.763 / 0.597 / 0.902 | 0.768 / 0.615 / 0.903 | 0.630 / **0.377** / 0.790 |

Cells report *accuracy / macro-F1 / macro ROC-AUC*. Source: [`results/model_comparison.csv`](results/model_comparison.csv).

- **Transfer fails as calibration, not as representation.** Model A loses 33.9% of its macro-F1 out
  of domain, but the loss is confined to the two intermediate damage classes (19.1% of training
  patches, only 1.7% of wildfire patches). `destroyed` still scores well out of domain (F1 0.758 vs.
  0.687 — partly its 3× higher prevalence there; recall 0.670 vs. 0.749), and ROC-AUC holds at 0.814.
- **Pretraining wins on every split and every metric**, most where it matters: +0.143 accuracy and
  +0.052 macro-F1 on wildfire.
- **Damage is not distributed along an income gradient once the disaster event is controlled for**
  (β = +0.211, p = 0.002 naive; β = +0.225, p = 0.0009 with built-environment controls →
  β = +0.057, p = 0.44 with event fixed effects), **but recovery
  capacity is**: priced in dollars and validated against NFIP/CAL FIRE/FEMA records, 60% of the
  poorest quartile's flood loss is covered by neither insurance nor federal aid, against 33% in the
  richest. Auditing the models' own predictions shows an automated relief rule would misdirect
  $30–78M of every $1B on seen disaster types and $112–200M on wildfire.

---

## 5. Limitations

Full discussion in Section 9 of `report.pdf`. Briefly: `minor`/`major` rest on only 116/138
out-of-domain examples; sun elevation is confounded with wildfire vs. non-fire acquisition, so part
of the transfer gap may be shadow length rather than fire damage; the two encoders differ in size as
well as pretraining, so this is not a pure ablation; and the socioeconomic analysis is ecological
(block-group averages, not individual households) with no true hazard-exposure layer available.

---

## 6. License

Released under the MIT License; see [`LICENSE`](LICENSE). The xBD dataset is subject to its own
terms and is not redistributed here. ACS, TIGER/Line, OpenFEMA, NOAA NCEI and CAL FIRE records are
public-domain U.S. government works.
