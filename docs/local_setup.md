# Local Windows setup

The tested local layout is:

```text
D:\DLSS\satellite-disaster-damage-mapping\
  .venv\
  data\
    splits.csv
    splits_summary.csv
  xbd\
    hold\{images,labels,masks}
    test\{images,labels,masks}
    tier1\{images,labels,masks}
    tier3\{images,labels,masks}
```

## Environment

From PowerShell in the repository root:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip wheel setuptools
.\.venv\Scripts\python.exe -m pip install -r requirements-local.txt
.\.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

Verify the GPU:

```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

The tested machine reports PyTorch `2.11.0+cu128`, CUDA available, and an
NVIDIA GeForce RTX 4070 Laptop GPU.

## Kaggle credentials and xBD download

Put the Kaggle API token at `%USERPROFILE%\.kaggle\kaggle.json`. Do not add it
to this repository.

The official one-stream Kaggle download is slow on the tested connection. The
repository therefore includes an eight-range downloader that resumes each
part independently:

```powershell
.\.venv\Scripts\python.exe .\scripts\download_xbd.py --parts 8 --workers 1
```

Re-run the same command after an interruption. Completed bytes in
`.download-parts\` are reused. When all parts finish, the script creates
`xbd-dataset.zip`.

Extract the archive into the repository root. Its top-level `xbd\` directory
must sit in the repository root, next to the `data\` directory:

```powershell
tar.exe -xf .\xbd-dataset.zip
```

Then validate the data and split assignment without overwriting the supplied
CSV files:

```powershell
.\.venv\Scripts\python.exe .\scripts\prepare_splits.py --src .\xbd --out .\data --dry-run
```

## Run locally

Start Jupyter from the repository root so relative paths resolve correctly:

```powershell
.\.venv\Scripts\jupyter-lab.exe
```

Use `notebooks\pretrained_resnet18_smoke_test.ipynb` first. It is configured for the tested
Windows laptop with 16 GB RAM and 8 GB VRAM: 170 scenes, batch size 32, four
extraction threads, and one epoch in each training phase. Its artifacts go to
`work-smoke\`.

After the smoke notebook succeeds, use
`notebooks\pretrained_resnet18_train.ipynb` for the formal pretrained run. It
creates about 3.5 GB under `work-pretrained\` and takes substantially longer.

Train the from-scratch CNN on the same extracted patches with:

```powershell
.\.venv\Scripts\jupyter.exe nbconvert --to notebook --execute `
  .\notebooks\train_scratch_cnn.ipynb `
  --output train_scratch_cnn.executed.ipynb `
  --output-dir .\notebooks `
  --ExecutePreprocessor.kernel_name=xbd-local `
  --ExecutePreprocessor.timeout=-1
```

Use the notebook configuration cell to set `RESUME = True` after an interrupted
full run, or `SMOKE_TEST = True` for an isolated short procedure check.
