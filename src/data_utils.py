"""Data layer: synthetic demo data + real xBD loading, behind one Dataset interface.

main.ipynb flips a single switch (USE_SYNTHETIC). Both paths produce datasets that
yield (pre, post, label) normalized CxHxW tensors and expose:
    .labels    -> numpy array of int labels (handy for class weights / EDA)
    .raw(i)    -> (pre_hwc, post_hwc) un-normalized [0,1] arrays, for plotting
"""
from pathlib import Path
import json
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from . import config

try:
    import pandas as pd
except ImportError:
    pd = None

# ImageNet stats (the pretrained ResNet expects this normalization).
_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_STD = np.array([0.229, 0.224, 0.225], np.float32)


def normalize_chw(hwc: np.ndarray) -> torch.Tensor:
    """HxWx3 in [0,1]  ->  normalized 3xHxW tensor."""
    arr = (hwc - _MEAN) / _STD
    return torch.from_numpy(arr.transpose(2, 0, 1).copy()).float()


def denormalize(hwc: np.ndarray) -> np.ndarray:
    """Just clip a [0,1] image for display (raw() already returns un-normalized)."""
    return np.clip(hwc, 0, 1)


# ===========================================================================
# SYNTHETIC DATA  — toy "buildings before/after" so the project runs with no download
# ===========================================================================
# Disaster "signature" = how the damage looks (color). Damage level = how much changes.
DISASTER_COLOR = {
    "hurricane":  [0.50, 0.50, 0.55],   # grey-blue debris
    "flood":      [0.20, 0.30, 0.80],   # water = blue
    "earthquake": [0.45, 0.42, 0.40],   # grey rubble
    "wildfire":   [0.75, 0.35, 0.08],   # burn scar = orange/black
    "tsunami":    [0.25, 0.45, 0.70],
    "tornado":    [0.55, 0.52, 0.50],
    "volcano":    [0.30, 0.25, 0.25],
}
DAMAGE_FRAC = {0: 0.0, 1: 0.20, 2: 0.50, 3: 0.90}   # fraction of building pixels changed
CLASS_P = [0.55, 0.20, 0.15, 0.10]                  # class imbalance (most are fine)


def make_pair(damage, disaster, rng, size=None):
    """Return (pre, post) toy images, HxWx3 in [0,1]."""
    size = size or config.PATCH_SIZE
    base = np.ones((size, size, 3), np.float32) * np.array([0.55, 0.60, 0.45], np.float32)
    base += rng.normal(0, 0.04, base.shape).astype(np.float32)
    h, w = rng.integers(size // 3, size // 2, size=2)
    y, x = int(rng.integers(0, size - h)), int(rng.integers(0, size - w))
    roof = np.array([0.60, 0.60, 0.62], np.float32) + rng.normal(0, 0.03, 3).astype(np.float32)
    pre = base.copy(); pre[y:y+h, x:x+w] = roof
    post = pre.copy()
    mask = rng.random((size, size)) < DAMAGE_FRAC[damage]
    region = np.zeros((size, size), bool); region[y:y+h, x:x+w] = True
    mask &= region
    sig = np.array(DISASTER_COLOR.get(disaster, [0.5, 0.5, 0.5]), np.float32)
    post[mask] = sig + rng.normal(0, 0.05, (int(mask.sum()), 3)).astype(np.float32)
    return np.clip(pre, 0, 1), np.clip(post, 0, 1)


class SyntheticDataset(Dataset):
    def __init__(self, n, disasters, seed):
        rng = np.random.default_rng(seed)
        pre, post, labels = [], [], []
        for _ in range(n):
            d = int(rng.choice(4, p=CLASS_P))
            dis = str(rng.choice(disasters))
            a, b = make_pair(d, dis, rng)
            pre.append(a); post.append(b); labels.append(d)
        self.pre = np.array(pre, np.float32)
        self.post = np.array(post, np.float32)
        self.labels = np.array(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return normalize_chw(self.pre[i]), normalize_chw(self.post[i]), int(self.labels[i])

    def raw(self, i):
        return self.pre[i], self.post[i]


def make_synthetic_datasets(n_train=900, n_val=200, n_test=300, seed=None):
    """Train/val on TRAIN_DISASTERS, test on the held-out disaster only."""
    seed = seed or config.SEED
    train = SyntheticDataset(n_train, config.TRAIN_DISASTERS, seed)
    val = SyntheticDataset(n_val, config.TRAIN_DISASTERS, seed + 1)
    test = SyntheticDataset(n_test, [config.HELDOUT_DISASTER], seed + 2)
    return train, val, test


# ===========================================================================
# REAL xBD DATA  — parse downloaded tiles into building patches + a manifest
# ===========================================================================
import re
_WKT_NUM = re.compile(r"-?\d+\.?\d*")


def _bbox(wkt_str):
    """Bounding box from a WKT polygon, e.g. 'POLYGON ((x1 y1, x2 y2, ...))'.

    Parsed directly (no shapely dependency): numbers alternate x, y.
    """
    nums = [float(v) for v in _WKT_NUM.findall(wkt_str)]
    xs, ys = nums[0::2], nums[1::2]
    return int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))


def _crop(img, bbox, pad=10, size=None):
    size = size or config.PATCH_SIZE
    minx, miny, maxx, maxy = bbox
    return img.crop((minx - pad, miny - pad, maxx + pad, maxy + pad)).resize((size, size))


def find_xbd_sources(data_dir=None):
    """Auto-discover xView2 source roots. Returns dirs that contain labels/ + images/.

    Works with the official layout (data/train, data/test) and a flat data/raw/xbd.
    """
    base = Path(data_dir or config.DATA_DIR)
    candidates = [base / "train", base / "test", base / "tier3", base / "raw" / "xbd"]
    return [c for c in candidates if (c / "labels").is_dir() and (c / "images").is_dir()]


def build_xbd_manifest(source_dirs=None, out_dir=None, val_frac=0.15, limit_tiles=None):
    """Turn xView2 tiles into building patches + data/processed/manifest.csv.

    Scans the official layout: data/train/{images,labels}/, data/test/..., data/tier3/...
    (each label is *_post_disaster.json with matching *_pre/post_disaster.png images).
    Produces data/processed/patches/*.png + manifest.csv with columns
    (pre_path, post_path, label, disaster_type, split). We re-split BY DISASTER TYPE:
    the held-out disaster -> 'test', everything else -> 'train' (a 'val' slice is carved out).
    """
    if pd is None:
        raise ImportError("pandas is required: pip install pandas")
    out = Path(out_dir or Path(config.DATA_DIR) / "processed")
    (out / "patches").mkdir(parents=True, exist_ok=True)
    roots = source_dirs or find_xbd_sources()
    if not roots:
        raise FileNotFoundError(
            "No xView2 data found. Expected data/train/{images,labels} (extract the .tar first)."
        )
    # collect (post_label_json, its images/ dir) across all source roots
    label_files = []
    keep = set(config.TRAIN_DISASTERS) | {config.HELDOUT_DISASTER}   # ignore other disaster types
    for root in map(Path, roots):
        for jp in sorted((root / "labels").glob("*post_disaster.json")):
            if config.event_to_disaster_type(jp.stem.split("_")[0]) in keep:
                label_files.append((jp, root / "images"))
    if limit_tiles:
        # shuffle first so a small subset still covers ALL disaster types (incl. held-out)
        rng_pick = np.random.default_rng(config.SEED)
        rng_pick.shuffle(label_files)
        label_files = label_files[:limit_tiles]

    rows = []
    for jp, images_dir in label_files:
        event = jp.stem.split("_")[0]                       # e.g. 'hurricane-florence'
        dtype = config.event_to_disaster_type(event)
        post_png = images_dir / (jp.stem + ".png")
        pre_png = images_dir / (jp.stem.replace("post", "pre") + ".png")
        if not post_png.exists() or not pre_png.exists():
            continue
        pre_img = Image.open(pre_png).convert("RGB")
        post_img = Image.open(post_png).convert("RGB")
        data = json.loads(jp.read_text())
        for k, feat in enumerate(data["features"].get("xy", [])):
            sub = feat["properties"].get("subtype", "un-classified")
            if sub not in config.NAME_TO_IDX:               # skip 'un-classified'
                continue
            bbox = _bbox(feat["wkt"])
            pre_p = _crop(pre_img, bbox); post_p = _crop(post_img, bbox)
            if _is_bad(np.asarray(post_p)):
                continue
            stem = f"{jp.stem}_{k}"
            pre_path = out / "patches" / f"{stem}_pre.png"
            post_path = out / "patches" / f"{stem}_post.png"
            pre_p.save(pre_path); post_p.save(post_path)
            rows.append({"pre_path": str(pre_path), "post_path": str(post_path),
                         "label": config.NAME_TO_IDX[sub], "disaster_type": dtype,
                         "split": "test" if dtype == config.HELDOUT_DISASTER else "train"})

    df = pd.DataFrame(rows)
    # carve a stratified val set out of train (sample per class so val keeps the class mix)
    train_part = df[df["split"] == "train"]
    val_idx = (train_part.groupby("label", group_keys=False)
               .sample(frac=val_frac, random_state=config.SEED).index)
    df.loc[val_idx, "split"] = "val"
    df.to_csv(out / "manifest.csv", index=False)
    print(f"Built manifest: {len(df)} patches ->", out / "manifest.csv")
    print(df["split"].value_counts())
    return df


def _is_bad(patch_np, white_frac=0.5):
    """Drop near-white (cloud) or near-black (missing) patches."""
    gray = patch_np.mean(axis=-1)
    return (gray > 240).mean() > white_frac or (gray < 15).mean() > white_frac


class DamageDataset(Dataset):
    """Real-data Dataset reading data/processed/manifest.csv (one split)."""

    def __init__(self, split="train", augment=False, manifest_path=None):
        if pd is None:
            raise ImportError("pandas is required: pip install pandas")
        manifest_path = manifest_path or Path(config.DATA_DIR) / "processed" / "manifest.csv"
        if not Path(manifest_path).exists():
            raise FileNotFoundError(
                f"{manifest_path} not found. Run data_utils.build_xbd_manifest() first."
            )
        df = pd.read_csv(manifest_path)
        self.df = df[df["split"] == split].reset_index(drop=True)
        self.labels = self.df["label"].to_numpy()
        self.augment = augment

    def __len__(self):
        return len(self.df)

    def _load(self, path):
        return np.asarray(Image.open(path).convert("RGB").resize(
            (config.PATCH_SIZE, config.PATCH_SIZE)), np.float32) / 255.0

    def __getitem__(self, i):
        row = self.df.iloc[i]
        pre, post = self._load(row["pre_path"]), self._load(row["post_path"])
        if self.augment and np.random.rand() < 0.5:
            pre, post = pre[:, ::-1], post[:, ::-1]
        return normalize_chw(pre), normalize_chw(post), int(row["label"])

    def raw(self, i):
        row = self.df.iloc[i]
        return self._load(row["pre_path"]), self._load(row["post_path"])
