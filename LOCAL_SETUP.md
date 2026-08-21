# Local Windows setup

The tested local layout is:

```text
D:\DLSS\satellite-disaster-damage-mapping\
  .venv\
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
.\.venv\Scripts\python.exe .\download_xbd_parallel.py --parts 8 --workers 1
```

Re-run the same command after an interruption. Completed bytes in
`.download-parts\` are reused. When all parts finish, the script creates
`xbd-dataset.zip`.

Extract the archive into the repository root. Its top-level `xbd\` directory
must sit next to `splits.csv`:

```powershell
tar.exe -xf .\xbd-dataset.zip
```

Then validate the data and split assignment without overwriting the supplied
CSV files:

```powershell
.\.venv\Scripts\python.exe .\make_splits.py --src .\xbd --out . --dry-run
```

## Run locally

Start Jupyter from the repository root so relative paths resolve correctly:

```powershell
.\.venv\Scripts\jupyter-lab.exe
```

Use `artur_xbd_LOCAL_SMOKE.ipynb` first. It is configured for the tested
Windows laptop with 16 GB RAM and 8 GB VRAM: 170 scenes, batch size 32, four
extraction threads, and one epoch in each training phase. Its artifacts go to
`work-smoke\`.

After the smoke notebook succeeds, use `artur_xbd_NESH_CAN_RUN.ipynb` for the
full reference configuration. The default full run creates about 3.5 GB under
`work\` and takes substantially longer.
