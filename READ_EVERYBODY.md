# 📖 READ EVERYBODY — start here!

Hey team 👋 (Bhuvanesh ·  · Anastasiia · Yitian) — this is the one-page map of our
project. Read it before touching any notebook. It takes 3 minutes.

---

## 🎯 What are we actually building?

We take **satellite photos taken before and after a natural disaster** (hurricane, flood,
wildfire…) and train a neural network to look at the *same building in both photos* and say
**how badly it got damaged**, on a 4-level scale:

```
no-damage  →  minor-damage  →  major-damage  →  destroyed
```

This "compare before vs after" idea has a name: **change detection**.

### The interesting research question 🔬
> *If a model learns what hurricane damage looks like, can it also recognise wildfire damage
> it has never seen?*

Damage looks **very different** across disasters (floods = water everywhere, wildfires = burn
scars, hurricanes = ripped-off roofs). So we deliberately **train on some disaster types and
test on a completely different one** (we hold out wildfires). This is the heart of the grade —
*not* a random train/test split.

We also build **two** models and compare them:
1. a small CNN **from scratch**, and
2. one on a **pretrained backbone** (ResNet18) — transfer learning.

---

## 🗺️ The pipeline (how data flows)

```
 ┌──────────────────────┐
 │  xBD / xView2 data    │   raw satellite tiles + building labels
 │  (download yourself)  │
 └──────────┬───────────┘
            │   notebook 01  (Role A)
            ▼
 ┌──────────────────────┐
 │  data/processed/      │   tiny pre/post patches, one per building
 │  manifest.csv         │   columns: pre_path, post_path, label, disaster_type, split
 └──────────┬───────────┘
            │
   ┌────────┴─────────┬──────────────────────┐
   │ notebook 02 (B)  │  notebooks 03 & 04    │
   ▼                  ▼  (C & D)              ▼
 ┌──────────┐   ┌───────────────┐    ┌────────────────────┐
 │   EDA     │   │ Model: scratch │    │ Model: pretrained   │
 │  charts   │   │  (siamese CNN) │    │  (ResNet18 + head)  │
 └──────────┘   └───────┬────────┘    └─────────┬──────────┘
                        └──────────┬────────────┘
                                   │  notebook 04 (Role D)
                                   ▼
                 ┌─────────────────────────────────┐
                 │  Held-out-disaster evaluation     │  ← the key result!
                 │  + scratch-vs-pretrained compare  │
                 │  + damage map of a real event     │
                 └────────────────┬─────────────────┘
                                  │  notebook 05 (Role B)
                                  ▼
                 ┌─────────────────────────────────┐
                 │  Impact analysis + report         │
                 │  damage × income/population data   │
                 └─────────────────────────────────┘
```

**Key idea:** notebook 01 produces ONE file — `manifest.csv`. Everything downstream reads it.
So as soon as Role A has even a tiny manifest, everyone else can start. 🚀

---

## 👥 Who does what

Pick the role you like — names aren't fixed. Full details are in
[`notebooks/00_overview_and_tasks.ipynb`](notebooks/00_overview_and_tasks.ipynb).

| Role | Notebook(s) | In one sentence |
|------|-------------|-----------------|
| **A · Data** | `01_data_preprocessing` | Turn raw tiles into clean `(pre, post, label)` patches + `manifest.csv`. |
| **B · EDA + Impact** | `02_eda_visualization`, `05_impact_recovery` | Charts of the data + Part 3 social-impact analysis + report. |
| **C · Scratch model** | `03_model_from_scratch` | Build & train the from-scratch siamese CNN. |
| **D · Pretrained + eval** | `04_model_pretrained_and_eval` | Transfer-learning model + the held-out evaluation + damage map. |

Everyone shares the code in [`src/`](src/) — change it together, don't fork it.

---

## 🏁 How to get started (everyone, once)

```bash
# 1. install the libraries
pip install -r requirements.txt

# 2. open Jupyter
jupyter notebook            # or use the VS Code notebook UI

# 3. open notebooks/00_overview_and_tasks.ipynb and run the last cell
#    -> if config prints, your setup works ✅
```

Then go to **your** notebook and look for the `TODO(Role X)` markers — that's where you write code.

> 💡 **Blocked because there's no data yet?** Ping Artur — we can generate a tiny *fake* dataset
> so notebooks 03/04 run end-to-end before the real download finishes.

---

## 📊 How we're graded (30 points)

| Part | Pts | The thing graders care about most |
|------|-----|-----------------------------------|
| 1 · Data & EDA | 10 | co-registration, handling the **class imbalance**, per-disaster charts |
| 2 · Models | 10 | both CNNs, a **fair** comparison, and the **held-out-disaster** test |
| 3 · Impact | 10 | damage × socioeconomic overlay + clear report writing |

---

## 🧠 Mini-glossary (for when a word feels scary)

- **Co-registration** — lining up the pre & post images so pixel (x,y) is the same spot on the ground.
- **Patch** — a small square crop around one building (we classify these, not whole images).
- **Siamese / two-branch network** — same CNN runs on both pre & post images, then we compare the two feature vectors.
- **Class imbalance** — most buildings are undamaged, so the model is lazy and predicts "no-damage" for everything unless we fix it (class weights).
- **Held-out disaster** — a disaster type the model never saw in training; testing on it = does knowledge *transfer*.
- **Transfer learning** — starting from a model pretrained on millions of generic images instead of from zero.

Let's go! 🛰️
