# Two-model experiment results

Both models use the same 135,262 paired 64x64 pre/post patches, scene-level
splits, shared geometric and photometric augmentation, inverse-frequency class
weights computed on train only, and the same val/test-ID/test-OOD evaluation.
Wildfire is absent from train and val and is evaluated only as test-OOD.

Training used an NVIDIA GeForce RTX 4070 Laptop GPU. Fifty epochs was an upper
bound, with early stopping on validation macro-F1 after eight stale epochs.

| Model | Parameters | Epochs run | Best epoch | Val macro-F1 | Test-ID macro-F1 | Test-OOD macro-F1 |
|---|---:|---:|---:|---:|---:|---:|
| Siamese ResNet-18, ImageNet pretrained | 11,571,524 | 43 | 35 | 0.6656 | 0.6498 | 0.4292 |
| Compact Siamese CNN, from scratch | 1,371,108 | 16 | 8 | 0.5967 | 0.6148 | 0.3768 |

The pretrained system is better on both held-out regimes. It gains 0.0350
macro-F1 on test-ID and 0.0524 on wildfire test-OOD. Its ID-to-OOD macro-F1
drop is 0.2206 (34.0%), versus 0.2380 (38.7%) for the scratch CNN.

This primary comparison answers the assignment's two-model question, but it is
not a pure pretraining ablation: the pretrained ResNet-18 is also larger than
the compact custom CNN. An optional third experiment using the same ResNet-18
architecture with random initialization would isolate pretraining alone.

## Reproduce

Pretrained experiment:

```powershell
.\.venv\Scripts\jupyter.exe nbconvert --to notebook --execute `
  .\notebooks\pretrained_resnet18_train.ipynb `
  --output pretrained_resnet18_train.executed.ipynb `
  --output-dir .\notebooks `
  --ExecutePreprocessor.kernel_name=xbd-local `
  --ExecutePreprocessor.timeout=-1
```

From-scratch experiment:

```powershell
.\.venv\Scripts\jupyter.exe nbconvert --to notebook --execute `
  .\notebooks\train_scratch_cnn.ipynb `
  --output train_scratch_cnn.executed.ipynb `
  --output-dir .\notebooks `
  --ExecutePreprocessor.kernel_name=xbd-local `
  --ExecutePreprocessor.timeout=-1
```

Set `RESUME = True` in the notebook configuration cell after an interrupted
full run. Set `SMOKE_TEST = True` to run a brief isolated procedure check.

Model checkpoints remain in the ignored local `work-*/checkpoints/`
directories. The best checkpoints are published in the
[`v1.0-trained-models` GitHub Release](https://github.com/artyz1200-hub/satellite-disaster-damage-mapping/releases/tag/v1.0-trained-models)
rather than committed to normal Git history.
