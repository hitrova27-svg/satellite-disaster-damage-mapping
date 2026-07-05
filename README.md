# 🛰️ Mapping Natural-Disaster Damage from Satellite Imagery

Single-notebook project (built for **Kaggle**): classify **per-building damage** by comparing
satellite imagery **before vs after** a disaster (*change detection*), on the xBD / xView2 dataset.

**Team:** Bhuvanesh Dinesh Wadhwani · Artur Zavistovskyi · Anastasiia Khitrova · Yitian Zhou

## What it does
- **Part 1 — data + EDA:** class imbalance, per-disaster breakdown, co-registered pre/post patches.
- **Part 2 — models:** two siamese change-detection CNNs (from-scratch vs pretrained ResNet18),
  evaluated on a **held-out disaster type (wildfire)** the model never trained on — the transfer
  question that is the heart of the project.
- **Part 3 — impact analysis:** overlay predicted damage with socioeconomic data (open / freeform).

Everything — code, data loading, training — lives in **[`main.ipynb`](main.ipynb)**. Kaggle opens
only one notebook, so all helper functions are inlined (no `src/` imports).

## Run it

### On Kaggle
1. **Add Data** ▸ attach the xBD dataset (the 4 `.tar` files below).
2. Open `main.ipynb`, set `USE_SYNTHETIC = False`, confirm the path in the **Data path** cell.
3. **Run All**. (Enable *Settings ▸ Internet* for the pretrained ResNet weights.)

### The one switch: `USE_SYNTHETIC`
| Value | What happens |
|-------|--------------|
| `True` (default) | Toy data generated in-notebook. Full pipeline in ~3–6 min, **no download**. Numbers are fake. |
| `False` | Real **xBD / xView2** data. Real results. |

## Data
xBD / xView2 — request access at <https://xview2.org> and upload these as a Kaggle Dataset:

| File | Why |
|------|-----|
| `train_images_labels_targets.tar` | main training data (PNG + JSON) |
| `tier3.tar` | more disaster types incl. **wildfires** (our held-out type) |
| `test_images_labels_targets.tar` | final evaluation |
| `xview_geotransforms.json.tgz` | lat/lon for Part 3 |

Kaggle auto-extracts the archives; the notebook auto-detects `train/ test/ tier3/` under the
attached dataset.
