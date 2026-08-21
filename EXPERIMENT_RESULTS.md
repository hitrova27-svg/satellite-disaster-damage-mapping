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
  .\artur_xbd_LOCAL_50E.ipynb `
  --output artur_xbd_LOCAL_50E.executed.ipynb `
  --ExecutePreprocessor.kernel_name=xbd-local `
  --ExecutePreprocessor.timeout=-1
```

From-scratch experiment:

```powershell
.\.venv\Scripts\python.exe .\train_from_scratch_50e.py `
  --epochs 50 --batch-size 128 --patience 8
```

Add `--resume` to the from-scratch command after an interrupted run.

Model checkpoints remain in the ignored local `work-*/checkpoints/`
directories. Publish them through Git LFS, a GitHub Release, or Kaggle rather
than committing generated training data or the xBD dataset to Git.
