# 📖 READ EVERYBODY — start here!

Hey team 👋 (Bhuvanesh · Artur · Anastasiia · Yitian) — this is the one-page map of our
project. Read it before opening `main.ipynb`. Takes 3 minutes.

---

## 🎯 What are we building?

We take **satellite photos before and after a natural disaster** (hurricane, flood, wildfire…)
and train a neural network to look at the *same building in both photos* and say **how badly it
was damaged**, on a 4-level scale:

```
no-damage  →  minor-damage  →  major-damage  →  destroyed
```

This "compare before vs after" idea is called **change detection**.

### The research question 🔬
> *If the model learns hurricane damage, can it recognise wildfire damage it has never seen?*

Damage looks very different across disasters, so we **train on some disaster types and test on a
completely different one** (we hold out wildfires). This is the heart of the grade — *not* a
random train/test split. We also build **two** models (one from scratch, one pretrained) and
compare them.

---

## 🗂️ How the project is organized (simple!)

There is **ONE notebook**: [`main.ipynb`](main.ipynb). It runs the whole project top to bottom.
The heavy code lives in [`src/`](src/) (plain `.py` files) so we can edit it without notebook
merge conflicts.

```
main.ipynb        ← THE notebook: Part 1 → Part 2 → Part 3, run top to bottom
src/
  config.py       ← settings: classes, disaster split, patch size
  data_utils.py   ← data: synthetic generator + real xBD loader
  models.py       ← both CNNs (from-scratch + pretrained)
  train_utils.py  ← training loop + metrics
READ_EVERYBODY.md ← this file
requirements.txt
```

> 🧪 **There's a built-in test inside `main.ipynb`.** At the top there's one switch,
> `USE_SYNTHETIC`. Leave it `True` and the notebook **generates toy data and runs the entire
> pipeline in ~3–6 min with no download** — so you can see how everything works immediately.
> Flip it to `False` for the real xBD data. Same code, only the data source changes.

---

## 🔌 The one switch: `USE_SYNTHETIC`

| Value | What happens | When to use |
|-------|--------------|-------------|
| `True`  (default) | Toy "buildings before/after" generated in-notebook. Full pipeline runs, **no download**. Numbers are fake. | Learning the code, developing, quick checks. |
| `False` | Loads real **xBD / xView2** data from `data/raw/xbd/`. Real results. | For the actual submission. |

---

## 🗺️ The pipeline (what `main.ipynb` does)

```
  data  ──►  EDA charts  ──►  train 2 models  ──►  test on a disaster
 (synthetic                    (scratch +          they NEVER saw   ──►  damage map  ──►  impact
  OR real xBD)                  pretrained)         = the key result                      analysis
```

---

## 👥 How we split the work (freeform — grab what you want)

Honest version: the core (data pipeline + both models + evaluation) is already wired up in
`main.ipynb` + `src/`. The parts that are **open for anyone to extend** without breaking the core:

| Open task | Where | Notes |
|-----------|-------|-------|
| **Part 3 — impact analysis** | bottom of `main.ipynb` | overlay damage with income/population data; the section is a guided TODO |
| **Better EDA** | Part 1 of `main.ipynb` | extra charts, per-disaster breakdowns |
| **The report** | `report.md` (create it) | write up findings, limitations, ethics |
| **Real-data polish** | `src/data_utils.build_xbd_manifest` | tune patch cropping / cleaning once data is downloaded |

These only *read* the model's outputs, so you can't break Parts 1–2. 🙂

---

## 🏁 How to run (everyone, once)

```bash
pip install -r requirements.txt
jupyter notebook        # or open main.ipynb in VS Code
# open main.ipynb, leave USE_SYNTHETIC = True, click "Run All"
```

If it runs to the bottom and shows a damage map — your setup works ✅.

---

## 📥 Getting the real data (xBD / xView2)

Needed only when you set `USE_SYNTHETIC = False`. From https://xview2.org/ (request access),
download the **Challenge** files (NOT the 51 GB GEOTIFF version):

| Download | Size | Why |
|----------|------|-----|
| Challenge **training set** | 7.8 GB | main data (PNG + JSON), enough to start |
| **Tier3** training data | 17.3 GB | more disaster types incl. wildfires (our held-out) |
| Challenge **test set** | 2.6 GB | optional, for final eval |
| **Geotransforms** | 500 KB | lat/lon for Part 3 |

Unpack into:
```
data/raw/xbd/
  images/   ...pre_disaster.png / ...post_disaster.png
  labels/   ...pre_disaster.json / ...post_disaster.json
```
Then `main.ipynb` (with `USE_SYNTHETIC = False`) builds the patches automatically on first run.

---

## 📊 How we're graded (30 points)

| Part | Pts | What graders care about most |
|------|-----|------------------------------|
| 1 · Data & EDA | 10 | co-registration, handling **class imbalance**, per-disaster charts |
| 2 · Models | 10 | both CNNs, a **fair** comparison, the **held-out-disaster** test |
| 3 · Impact | 10 | damage × socioeconomic overlay + clear report writing |

---

## 🧠 Mini-glossary

- **Change detection** — comparing before/after images to find what changed.
- **Patch** — a small crop around one building (we classify these, not whole images).
- **Siamese / two-branch network** — same CNN looks at both pre & post, then we compare the features.
- **Class imbalance** — most buildings are fine, so the model gets lazy unless we use class weights.
- **Held-out disaster** — a disaster type the model never trained on; testing on it = does knowledge *transfer*.
- **Transfer learning** — starting from a model pretrained on millions of images instead of from zero.

Let's go! 🛰️
