"""Shared Dataset + transforms. Reads the manifest produced by notebook 01."""
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from . import config

try:
    import pandas as pd
except ImportError:
    pd = None

_MEAN = [0.485, 0.456, 0.406]
_STD = [0.229, 0.224, 0.225]


def _to_tensor(img: Image.Image) -> torch.Tensor:
    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    arr = (arr - _MEAN) / _STD
    return torch.from_numpy(arr).permute(2, 0, 1).float()


class DamageDataset(Dataset):
    """Yields (pre, post, label). `pre`/`post` are CxHxW tensors.

    Expects data/processed/manifest.csv with columns:
        pre_path, post_path, label, disaster_type, split
    Role A (notebook 01) produces this file.
    """

    def __init__(self, split="train", augment=False, manifest_path=None):
        if pd is None:
            raise ImportError("pandas is required: pip install pandas")
        manifest_path = manifest_path or Path(config.DATA_DIR) / "processed" / "manifest.csv"
        if not Path(manifest_path).exists():
            raise FileNotFoundError(
                f"{manifest_path} not found. Run notebook 01 first to build the dataset."
            )
        df = pd.read_csv(manifest_path)
        self.df = df[df["split"] == split].reset_index(drop=True)
        self.augment = augment

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        pre = Image.open(row["pre_path"]).resize((config.PATCH_SIZE, config.PATCH_SIZE))
        post = Image.open(row["post_path"]).resize((config.PATCH_SIZE, config.PATCH_SIZE))
        if self.augment and np.random.rand() < 0.5:
            pre = pre.transpose(Image.FLIP_LEFT_RIGHT)
            post = post.transpose(Image.FLIP_LEFT_RIGHT)
        return _to_tensor(pre), _to_tensor(post), int(row["label"])
