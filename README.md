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

See [EXPERIMENT_RESULTS.md](EXPERIMENT_RESULTS.md) and
[`results/model_comparison.csv`](results/model_comparison.csv) for the full
comparison and reproducibility notes.

## Trained model downloads

The best checkpoints from both experiments are published in the
[`v1.0-trained-models` GitHub Release](https://github.com/artyz1200-hub/satellite-disaster-damage-mapping/releases/tag/v1.0-trained-models):

- [`pretrained_siamese_resnet18.pt`](https://github.com/artyz1200-hub/satellite-disaster-damage-mapping/releases/download/v1.0-trained-models/pretrained_siamese_resnet18.pt)
  (44.23 MB): ImageNet-pretrained Siamese ResNet-18, best epoch 35.
- [`from_scratch_siamese_cnn.pt`](https://github.com/artyz1200-hub/satellite-disaster-damage-mapping/releases/download/v1.0-trained-models/from_scratch_siamese_cnn.pt)
  (5.25 MB): compact Siamese CNN trained from scratch, best epoch 8.

## Repository guide

- `artur_xbd_LOCAL_50E.ipynb`: pretrained ResNet-18 formal experiment.
- `train_from_scratch_50e.py`: from-scratch CNN formal experiment.
- `artur_xbd_LOCAL_SMOKE.ipynb`: short end-to-end local validation.
- `make_splits.py`: builds and validates the scene-level split.
- `splits.csv`: shared file index and split assignment.
- `results/`: lightweight metrics, histories, confusion matrices, and curves.
- `LOCAL_SETUP.md`: Windows environment, data download, and launch commands.

## Local run

The expected layout is:

```text
<repo>/
  splits.csv
  xbd/{hold,test,tier1,tier3}/{images,labels,masks}/
```

Start Jupyter with the project environment:

```powershell
.\.venv\Scripts\jupyter-lab.exe
```

Run the pretrained experiment from Jupyter, or non-interactively:

```powershell
.\.venv\Scripts\jupyter.exe nbconvert --to notebook --execute `
  .\artur_xbd_LOCAL_50E.ipynb `
  --output artur_xbd_LOCAL_50E.executed.ipynb `
  --ExecutePreprocessor.kernel_name=xbd-local `
  --ExecutePreprocessor.timeout=-1
```

The pretrained notebook creates the shared patch arrays. Train the scratch CNN
on those exact arrays with:

```powershell
.\.venv\Scripts\python.exe .\train_from_scratch_50e.py `
  --epochs 50 --batch-size 128 --patience 8
```

Use `--resume` after an interrupted scratch run.

## Data and generated artifacts

xBD is not committed. Obtain it from the xView2 source or the configured Kaggle
mirror. Do not commit `xbd/`, archives, `.venv/`, extracted patch arrays, or
training work directories.

Model checkpoints are generated locally and the best checkpoints are published
through the GitHub Release linked above. They are not committed to normal Git
history.
