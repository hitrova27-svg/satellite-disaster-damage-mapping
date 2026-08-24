# Mapping Natural-Disaster Damage from Satellite Imagery

Per-building damage classification from paired pre/post satellite imagery on
xBD. The central question is cross-disaster transfer: both models train without
wildfire and are evaluated on wildfire as an unseen test-OOD domain.

**Team:** Bhuvanesh Dinesh Wadhwani · Artur Zavistovskyi · Anastasiia Khitrova · Yitian Zhou

## Experiment

Both systems use the same 135,262 paired 64x64 patches and the same scene-level
train/val/test-ID/test-OOD split.

1. A compact shared-weight Siamese CNN trained from random initialization.
2. A shared-weight Siamese ResNet-18 initialized from ImageNet, then fine-tuned.

Each branch encodes one of the pre/post patches. The classifier receives
`[f_pre, f_post, abs(f_pre - f_post)]` and predicts no damage, minor damage,
major damage, or destroyed.

## Results

Fifty epochs was the maximum; both runs used early stopping on validation
macro-F1 with patience 8.

| Model | Epochs run | Best epoch | Val macro-F1 | Test-ID macro-F1 | Test-OOD macro-F1 |
|---|---:|---:|---:|---:|---:|
| ImageNet-pretrained Siamese ResNet-18 | 43 | 35 | 0.6656 | 0.6498 | 0.4292 |
| From-scratch Siamese CNN | 16 | 8 | 0.5967 | 0.6148 | 0.3768 |

See [`docs/experiment_results.md`](docs/experiment_results.md) and
[`results/model_comparison.csv`](results/model_comparison.csv) for the full
comparison and reproducibility notes.

## Trained model downloads

The best checkpoints from both experiments are published in the
[`v1.0-trained-models` GitHub Release](https://github.com/artyz1200-hub/satellite-disaster-damage-mapping/releases/tag/v1.0-trained-models):

- [`pretrained_siamese_resnet18.pt`](https://github.com/artyz1200-hub/satellite-disaster-damage-mapping/releases/download/v1.0-trained-models/pretrained_siamese_resnet18.pt)
  (44.23 MB): ImageNet-pretrained Siamese ResNet-18, best epoch 35.
- [`from_scratch_siamese_cnn.pt`](https://github.com/artyz1200-hub/satellite-disaster-damage-mapping/releases/download/v1.0-trained-models/from_scratch_siamese_cnn.pt)
  (5.25 MB): compact Siamese CNN trained from scratch, best epoch 8.

## Repository layout

```text
.
├── notebooks/                 training notebooks
│   └── archive/               preserved handoff version
├── scripts/                   download, split, and setup scripts
├── data/                      committed split definitions (not xBD imagery)
├── docs/                      project specification and experiment guides
├── results/                   final lightweight metrics and plots
├── figures/                   figures used by the existing LaTeX report
├── report.tex                 existing report source
├── report.pdf                 existing compiled report
├── requirements-local.txt     local Python dependencies
├── README.md                  project entry point
└── LICENSE                    MIT license
```

### Root files

| Path | Purpose |
|---|---|
| `README.md` | Project overview, headline results, file guide, model downloads, and run commands. |
| `requirements-local.txt` | Python packages required by the local Windows environment. CUDA-enabled PyTorch is installed separately. |
| `.gitignore` | Excludes xBD imagery, archives, virtual environments, extracted patches, logs, executed notebooks, and checkpoints. |
| `LICENSE` | MIT open-source license. |
| `report.tex` | Existing LaTeX report source maintained by the team. |
| `report.pdf` | Existing compiled report. |

### Notebooks

| Path | Purpose |
|---|---|
| `notebooks/pretrained_resnet18_train.ipynb` | Formal local experiment for the ImageNet-pretrained Siamese ResNet-18. It performs EDA, cleaning, patch extraction, training, and evaluation. |
| `notebooks/train_scratch_cnn.ipynb` | Trains, validates, checkpoints, and evaluates the compact Siamese CNN from random initialization on the pretrained notebook's patch arrays. |
| `notebooks/pretrained_resnet18_smoke_test.ipynb` | Small end-to-end environment check. It uses fewer scenes and short training phases; its metrics are not final results. |
| `notebooks/pretrained_resnet18_kaggle_reference.ipynb` | Executed Kaggle T4 reference notebook with the earlier pretrained-model run and embedded outputs. It is preserved for comparison, not used as the final two-model result. |
| `notebooks/archive/pretrained_resnet18_local_handoff.ipynb` | Earlier local handoff version preserved for history. Use the formal training notebook above for current runs. |

### Scripts

| Path | Purpose |
|---|---|
| `scripts/download_xbd.py` | Resumable range downloader for the Kaggle xBD archive. |
| `scripts/prepare_splits.py` | Scans xBD, creates the scene-level train/val/test-ID/test-OOD assignment, and checks leakage and file integrity. |
| `scripts/setup_windows.ps1` | Waits for a download, validates and extracts the archive, checks the split, and runs the smoke notebook on Windows. |

### Data definitions

| Path | Purpose |
|---|---|
| `data/splits.csv` | File-level xBD index: 77,238 rows mapping each image, label, and mask file to a scene, disaster, origin, and split. |
| `data/splits_summary.csv` | Scene-level summary: 11,034 rows used to inspect event composition and split counts. |

These CSVs define the shared experiment split. They are not the xBD image
dataset and are safe to keep in Git.

### Documentation

| Path | Purpose |
|---|---|
| `docs/project_specification.md` | Original project description, required steps, and grading criteria. |
| `docs/experiment_results.md` | Final two-model comparison, interpretation, limitations, and reproduction commands. |
| `docs/local_setup.md` | Detailed Windows, Kaggle, CUDA, directory-layout, and local-run instructions. |

### Final results

| Path | Purpose |
|---|---|
| `results/model_comparison.csv` | Single summary table comparing both models across validation, test-ID, and wildfire test-OOD. |
| `results/pretrained_resnet18/results_overall.csv` | Overall metrics for the pretrained model. |
| `results/pretrained_resnet18/results_per_class.csv` | Per-class precision, recall, F1, and support for the pretrained model. |
| `results/pretrained_resnet18/training_history.csv` | Epoch-by-epoch pretrained-model loss and validation metrics. |
| `results/pretrained_resnet18/training_curves.png` | Pretrained-model training curves. |
| `results/scratch_cnn/results_overall.csv` | Overall metrics for the from-scratch CNN. |
| `results/scratch_cnn/results_per_class.csv` | Per-class precision, recall, F1, and support for the scratch CNN. |
| `results/scratch_cnn/training_history.csv` | Epoch-by-epoch scratch-CNN loss and validation metrics. |
| `results/scratch_cnn/training_curves.png` | Scratch-CNN training curves. |
| `results/scratch_cnn/confusion_matrices.csv` | Confusion matrices for validation, test-ID, and test-OOD. |
| `results/scratch_cnn/run_config.json` | GPU, seed, batch size, stopping rule, paths, parameter count, and best epoch for the scratch run. |

### Existing report figures

The `figures/` directory belongs to the existing LaTeX report. These files are
separate from the final local-run plots under `results/`.

| Path | Purpose |
|---|---|
| `figures/bbox_geometry.png` | Building bounding-box size and geometry diagnostics. |
| `figures/buildings_per_scene.png` | Distribution of annotated building counts per scene. |
| `figures/capture_metadata.png` | Satellite capture-metadata distributions. |
| `figures/class_dist_event.png` | Damage-class distribution by disaster event. |
| `figures/class_dist_split.png` | Damage-class distribution across train, validation, test-ID, and test-OOD. |
| `figures/confusion_matrices.png` | Confusion matrices from the earlier reference run. |
| `figures/examples_ood.png` | Example wildfire test-OOD scenes. |
| `figures/examples_train.png` | Example training scenes. |
| `figures/occlusion_gallery.png` | Gallery of highly ranked cloud/occlusion candidates. |
| `figures/occlusion_scatter.png` | Relationship between image statistics and unclassified-label rates. |
| `figures/patches_grid.png` | Paired pre/post building-patch examples. |
| `figures/per_class_recall.png` | Per-class recall from the earlier reference evaluation. |
| `figures/training_curves.png` | Training curves from the earlier reference run. |
| `figures/unclassified_rate.png` | Distribution of unclassified annotation rates. |

## Local run

The expected layout is:

```text
<repo>/
  data/splits.csv
  data/splits_summary.csv
  xbd/{hold,test,tier1,tier3}/{images,labels,masks}/
```

Start Jupyter with the project environment:

```powershell
.\.venv\Scripts\jupyter-lab.exe
```

Run the pretrained experiment from Jupyter, or non-interactively:

```powershell
.\.venv\Scripts\jupyter.exe nbconvert --to notebook --execute `
  .\notebooks\pretrained_resnet18_train.ipynb `
  --output pretrained_resnet18_train.executed.ipynb `
  --output-dir .\notebooks `
  --ExecutePreprocessor.kernel_name=xbd-local `
  --ExecutePreprocessor.timeout=-1
```

The pretrained notebook creates the shared patch arrays. Then open
`notebooks/train_scratch_cnn.ipynb` and run all cells from the project root. It
saves the best validation-macro-F1 weights to `work-scratch/checkpoints/best.pt`.

For a non-interactive run, execute the notebook with:

```powershell
.\.venv\Scripts\jupyter.exe nbconvert --to notebook --execute `
  .\notebooks\train_scratch_cnn.ipynb `
  --output train_scratch_cnn.executed.ipynb `
  --output-dir .\notebooks `
  --ExecutePreprocessor.kernel_name=xbd-local `
  --ExecutePreprocessor.timeout=-1
```

Set `RESUME = True` in its configuration cell after an interrupted full run.
Set `SMOKE_TEST = True` to exercise the checkpointing and evaluation procedure
on a few batches without overwriting full-run artifacts.

## Data and generated artifacts

xBD is not committed. Obtain it from the xView2 source or the configured Kaggle
mirror. Do not commit `xbd/`, archives, `.venv/`, extracted patch arrays, or
training work directories.

Model checkpoints are generated locally and the best checkpoints are published
through the GitHub Release linked above. They are not committed to normal Git
history.
