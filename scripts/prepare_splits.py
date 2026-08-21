#!/usr/bin/env python3
"""
prepare_splits.py — index the four xBD directories and assign train/val/test splits.

PURPOSE
-------
xBD ships as four directories (hold, test, tier1, tier3) that reflect the
structure of the original xView2 competition.
This script does both steps in one pass:

    1. MERGE  — scan all four directories into one unified index, checking
                for filename collisions and incomplete scenes.
    2. SPLIT  — assign every scene to train / val / test_id / test_ood.

The output is a CSV. By default NO image files are moved, copied or linked:
each row records the file's original path, and downstream code reads paths from the CSV.

INPUT   xbd/{hold,test,tier1,tier3}/{images,labels,masks}/
OUTPUT  data/splits.csv          one row per file: path, scene, disaster, split
        data/splits_summary.csv  one row per scene (convenient for plots)

SPLIT DESIGN
------------
    train     tier1 + tier3, non-wildfire
    val       tier1 + tier3, non-wildfire
    test_id   official test + hold, non-wildfire   (in-distribution)
    test_ood  every wildfire scene, any origin     (zero-shot)

The research question is whether a damage-assessment model generalises to a
disaster type it has never seen. `test_ood` measures that. But an OOD score
in isolation is uninterpretable — a mediocre number could mean domain shift
or simply a weak model — so `test_id` provides a same-model, same-training-run
reference on data the model also never trained on. The ID -> OOD gap is the
actual result; neither number means much alone.

`test_id` is drawn from the official test/hold directories because those were
never part of any training pool and, unlike `val`, were never used for tuning.
That makes them the more honest basis for a headline number.

Background on the source directories: in the xView2 competition `test` was
the public leaderboard set and `hold` was withheld for final scoring. Both
are labelled now and both are disjoint from tier1/tier3, so both function
here as clean evaluation data. `tier1` was the original training pool and
`tier3` was released later as additional training data.
"""

import argparse
import os
import re
import shutil
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


# ==========================================================================
# CONSTANTS
#
# These were established by inspecting the data interactively, then frozen
# here. Hard-coding them (rather than re-deriving them at runtime) means a
# change in the data causes a loud failure instead of a silently different
# split.
# ==========================================================================

# The four directories xBD ships as. `origin` in the output refers to these.
ORIGINS = ("hold", "test", "tier1", "tier3")
SUBDIRS = ("images", "labels", "masks")

TRAIN_ORIGINS = ("tier1", "tier3")      # original training data
OFFICIAL_ORIGINS = ("test", "hold")     # official evaluation sets

FIRE_EVENTS = {
    "socal-fire",
    "woolsey-fire",
    "pinery-bushfire",
    "portugal-wildfire",
    "santa-rosa-wildfire",
}

# Used to detect fire-like names MISSING from FIRE_EVENTS above.
FIRE_KEYWORDS = ("fire", "bushfire", "wildfire")

# If the in-distribution reference set falls below this many scenes, the
# ID -> OOD comparison rests on too little data to support a claim.
MIN_TEST_ID_SCENES = 300

# Filename grammar: the disaster name, a zero-padded scene index, then a
# "kind" suffix.
NAME_PAT = re.compile(
    r"^(?P<disaster>[a-z\-]+?)_"     # e.g. santa-rosa-wildfire
    r"(?P<idx>\d+)_"                 # e.g. 00000123
    r"(?P<kind>.+)$"                 # e.g. post_disaster
)


def fail(msg: str) -> None:
    """Abort with a nonzero exit code.

    Used for conditions that would invalidate every downstream result.
    Exiting rather than warning
    """
    sys.exit(f"ERROR: {msg}")


# ==========================================================================
# STEP 1 — MERGE: scan the four directories into one index
# ==========================================================================

def scan(src: Path) -> pd.DataFrame:
    """Walk src/{origin}/{subdir}/ and build one row per file.

    Args:
        src: dataset root containing the four origin directories.

    Returns:
        DataFrame with columns [origin, subdir, name, path].

    Exits:
        If no files are found at all (usually a wrong --src).
    """
    rows = []
    for origin in ORIGINS:
        for sub in SUBDIRS:
            d = src / origin / sub
            if not d.is_dir():
                print(f"  ! missing directory, skipping: {d}", file=sys.stderr)
                continue
            for p in sorted(d.iterdir()):
                # Skip dotfiles: macOS .DS_Store and similar would otherwise
                # reach the filename parser and abort the run.
                if p.is_file() and not p.name.startswith("."):
                    rows.append({"origin": origin, "subdir": sub,
                                 "name": p.name, "path": str(p)})

    if not rows:
        fail(f"no files found under {src} — check the path and directory layout")

    df = pd.DataFrame(rows)
    print(f"scanned {len(df)} files from {src}")
    print("\nfiles per origin and subdirectory")
    print(df.pivot_table(index="origin", columns="subdir", values="name",
                         aggfunc="count", fill_value=0)
            .reindex([o for o in ORIGINS if o in set(df.origin)]).to_string())
    return df


def check_collisions(df: pd.DataFrame) -> None:
    """Verify no filename is reused across origin directories.

    xBD filenames encode disaster name and index, so they are expected to
    be globally unique.

    A collision matters here even without a physical merge: two different
    files sharing a name would collapse into one scene_id, so a tier1 scene
    and a tier3 scene would be treated as the same location and could pull
    each other across the train/val boundary.

    Args:
        df: scanned file index.

    Exits:
        If the same filename appears in more than one origin.
    """
    dupes = df.groupby(["subdir", "name"]).filter(lambda g: len(g) > 1)
    if len(dupes):
        print(f"\n! {dupes.name.nunique()} filename(s) appear in multiple origins:",
              file=sys.stderr)
        print(dupes.sort_values("name").head(10).to_string(index=False),
              file=sys.stderr)
        fail("filename collisions across origins; scene ids would merge "
             "unrelated locations")
    print("\nno filename collisions across origins")


# ==========================================================================
# STEP 2 — PARSE AND VALIDATE STRUCTURE
# ==========================================================================

def parse_names(df: pd.DataFrame) -> pd.DataFrame:
    """Decompose each filename into disaster, scene_id and kind.

    `scene_id` is the primary key.

    Args:
        df: scanned file index, requires a `name` column.

    Returns:
        The same frame with `disaster`, `scene_id` and `kind` columns added.

    Exits:
        If any filename fails to match NAME_PAT. An unparsed file would
        receive no scene_id, hence no split, and would vanish silently
        from the dataset.
    """
    def one(name):
        m = NAME_PAT.match(Path(name).stem)
        if not m:
            return pd.Series({"disaster": None, "scene_id": None, "kind": None})
        g = m.groupdict()
        return pd.Series({
            "disaster": g["disaster"],
            # Note the origin is intentionally NOT part of the scene id: a
            # scene must resolve identically regardless of which directory
            # it came from, otherwise pre and post could split apart.
            "scene_id": f"{g['disaster']}_{g['idx']}",
            "kind": g["kind"],
        })

    df = df.join(df.name.apply(one))

    bad = df[df.scene_id.isna()]
    if len(bad):
        print(bad[["origin", "subdir", "name"]].head(10).to_string(index=False),
              file=sys.stderr)
        fail(f"{len(bad)} filename(s) did not parse; fix NAME_PAT")
    return df


def check_structure(df: pd.DataFrame) -> None:
    """Print the file-kind inventory and verify every scene is complete.

    The masks directory holds roughly 1.5x as many files as images (e.g.
    2,799 masks against 1,866 images in `hold`). This function derives the expected
    file count per scene empirically (the mode) and flags any scene that
    deviates from it.

    Args:
        df: parsed index with `subdir`, `scene_id` and `kind` columns.

    Exits:
        If a subdirectory is empty, or if any scene has an unexpected
        number of files.
    """
    inv = df.groupby("subdir").kind.value_counts().unstack(fill_value=0).T
    print("\nfile kinds per subdirectory")
    print(inv.to_string())

    for sub in SUBDIRS:
        s = df[df.subdir == sub]
        if s.empty:
            fail(f"no files found for subdir '{sub}'")

        sizes = s.groupby("scene_id").kind.nunique()
        expected = sizes.mode().iat[0]      # the overwhelmingly common case
        odd = sizes[sizes != expected]
        if len(odd):
            print(f"\n! {len(odd)} scene(s) in '{sub}' have an unusual number "
                  f"of files (expected {expected}):", file=sys.stderr)
            print(odd.head(10).to_string(), file=sys.stderr)
            fail(f"incomplete scenes in '{sub}'; inspect before splitting")

        print(f"  {sub:7s} {expected} file(s) per scene, "
              f"{s.scene_id.nunique()} scenes")


def check_fire_events(scenes: pd.DataFrame) -> None:
    """Guard the hard-coded FIRE_EVENTS list against drift in the data.

    Args:
        scenes: one row per scene, requires a `disaster` column.

    Exits:
        If a fire-like disaster name is not in FIRE_EVENTS.
    """
    present = set(scenes.disaster.unique())

    # Informational only — a listed event may legitimately be absent if
    # the script is run on a partial download.
    missing = FIRE_EVENTS - present
    if missing:
        print(f"! listed fire events absent from data: {sorted(missing)}",
              file=sys.stderr)

    suspicious = {
        d for d in present - FIRE_EVENTS
        if any(k in d.lower() for k in FIRE_KEYWORDS)
    }
    if suspicious:
        fail(f"disaster(s) look like fire but are not in FIRE_EVENTS: "
             f"{sorted(suspicious)}. Add them, or the holdout leaks.")


# ==========================================================================
# STEP 3 — ASSIGNMENT
# ==========================================================================

def assign(scenes: pd.DataFrame, val_frac: float, seed: int,
           include_wildfire: bool) -> pd.DataFrame:
    """Map every scene to exactly one split.

    Only train and val are drawn randomly. The two test sets are defined
    by the data itself — test_id is whatever the official evaluation
    directories contribute, test_ood is every wildfire scene — so neither
    depends on the seed.

    Args:
        scenes: one row per scene, with `origin` and `is_fire` columns.
        val_frac: validation fraction OF THE TRAINABLE POOL, not of the
            whole dataset. With wildfire removed the pool is smaller than
            the full dataset, so this figure is tuned to land val near
            10% overall; always read the printed summary rather than
            assuming the final percentage.
        seed: passed to train_test_split for reproducibility.
        include_wildfire: if True, use the standard xBD protocol instead
            (wildfire stays in training, test_ood is empty). Published
            baselines train on all disaster types, so this mode exists to
            produce numbers that are actually comparable to them.

    Returns:
        DataFrame with columns [scene_id, split].
    """
    trainable = scenes[scenes.origin.isin(TRAIN_ORIGINS)]
    official = scenes[scenes.origin.isin(OFFICIAL_ORIGINS)]

    if include_wildfire:
        # Standard protocol: nothing is held out by disaster type.
        pool, ood = trainable.copy(), scenes.iloc[0:0]
        test_id = official.copy()
    else:
        # Held-out protocol: fire is removed from BOTH the training pool
        # and the in-distribution test set, so test_id stays strictly
        # in-distribution and test_ood collects every fire scene
        # regardless of which directory it came from.
        pool = trainable[~trainable.is_fire].copy()
        test_id = official[~official.is_fire].copy()
        ood = scenes[scenes.is_fire].copy()

    if len(pool) < 10:
        fail(f"training pool has only {len(pool)} scenes")

    # Stratify by disaster so train and val see a comparable event mix.
    # Without this, a rare event could land entirely in val, making
    # validation loss jump for reasons unrelated to model quality.
    #
    # train_test_split cannot stratify a class with only one member, and
    # xBD does contain events represented by a single scene. Those are
    # routed directly to train rather than bucketed together: a singleton
    # cannot be split at all, and the extra example is worth more in
    # training than in a validation set it would distort.
    counts = pool.disaster.value_counts()
    singleton = pool[pool.disaster.map(counts) < 2]
    splittable = pool[pool.disaster.map(counts) >= 2].copy()

    if len(singleton):
        print(f"\n{len(singleton)} scene(s) from single-scene events sent "
              f"directly to train: {sorted(singleton.disaster.unique())}")

    # StratifiedShuffleSplit additionally requires the validation set to be
    # at least as large as the number of classes, otherwise some class gets
    # zero validation examples. If val_frac is too small for the number of
    # distinct events, stratification is dropped.
    n_classes = splittable.disaster.nunique()
    n_val = int(round(len(splittable) * val_frac))
    strat = splittable.disaster
    if n_val < n_classes:
        print(f"! val would hold {n_val} scenes but there are {n_classes} "
              f"events; splitting without stratification. Raise --val-frac "
              f"to restore a balanced event mix.", file=sys.stderr)
        strat = None

    tr, va = train_test_split(
        splittable, test_size=val_frac, random_state=seed, stratify=strat
    )

    return pd.concat([
        tr.assign(split="train"),
        singleton.assign(split="train"),
        va.assign(split="val"),
        test_id.assign(split="test_id"),
        ood.assign(split="test_ood"),
    ])[["scene_id", "split"]]


# ==========================================================================
# STEP 4 — VERIFICATION
# ==========================================================================

def verify(df: pd.DataFrame, scenes: pd.DataFrame, include_wildfire: bool) -> None:
    """Assert the partition is sound before anything is written.

    Each check corresponds to a failure mode that is invisible at training
    time and only shows up as an implausibly good score later.

    Args:
        df: file-level frame, already merged with the split assignment.
        scenes: scene-level frame carrying `is_fire`.
        include_wildfire: relaxes the wildfire check under the standard
            protocol, where fire in training is intended.

    Exits:
        On unassigned files, scene-level leakage, or wildfire found in
        train/val when it should have been held out.
    """
    # 1. Every file must have a split. A missing assignment means a scene
    #    fell through the origin filters and would be dropped from the
    #    dataset without warning.
    n = df.split.isna().sum()
    if n:
        fail(f"{n} file(s) received no split")

    # 2. The central leakage check: no scene may appear in two splits.
    #    This is what guarantees pre and post images stayed together.
    straddle = df.groupby("scene_id").split.nunique()
    if (straddle > 1).any():
        fail(f"{(straddle > 1).sum()} scene(s) appear in more than one split")

    if not include_wildfire:
        # 3. The experiment's core premise: zero wildfire exposure during
        #    training. Checked on images only, since one row per file
        #    would otherwise triple-count via the masks.
        img = df[df.subdir == "images"].merge(
            scenes[["scene_id", "is_fire"]], on="scene_id")
        leaked = img[img.split.isin(["train", "val"]) & img.is_fire]
        if len(leaked):
            fail(f"{leaked.scene_id.nunique()} wildfire scene(s) in train/val")

        # 4. Not an error — the split is valid, but a small reference set
        #    makes the ID -> OOD gap too noisy to draw conclusions from.
        n_id = df[df.split == "test_id"].scene_id.nunique()
        if n_id < MIN_TEST_ID_SCENES:
            print(f"\n! test_id has only {n_id} scenes (< {MIN_TEST_ID_SCENES}). "
                  f"The ID -> OOD comparison will be noisy; consider moving "
                  f"some non-fire tier1/tier3 scenes into it.", file=sys.stderr)

    print("\nall integrity checks passed")


# ==========================================================================
# ENTRY POINT
# ==========================================================================

def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, default=Path("xbd"),
                    help="dataset root holding hold/test/tier1/tier3 "
                         "(default: xbd)")
    ap.add_argument("--out", type=Path, default=Path("data"),
                    help="where to write splits.csv (default: data)")
    ap.add_argument("--seed", type=int, default=42,
                    help="controls the train/val draw only (default: 42)")
    ap.add_argument("--val-frac", type=float, default=0.111,
                    help="val fraction of the trainable pool (default: 0.111)")
    ap.add_argument("--include-wildfire", action="store_true",
                    help="standard xBD protocol: wildfire stays in training")
    ap.add_argument("--link-into", type=Path, default=None, metavar="DIR",
                    help="also build a {split}/{subdir}/ tree of hard links "
                         "here (only needed for ImageFolder-style loaders)")
    ap.add_argument("--dry-run", action="store_true",
                    help="run every check and print the summary, write nothing")
    args = ap.parse_args()

    if not args.src.is_dir():
        fail(f"not found: {args.src}")

    # ---- merge -----------------------------------------------------------
    df = scan(args.src)
    check_collisions(df)
    df = parse_names(df)
    check_structure(df)

    # Collapse to one row per scene. Restricting to images (rather than
    # dropping duplicates over all subdirs) keeps this unambiguous: images
    # have exactly one row per phase and no derived variants.
    scenes = (df[df.subdir == "images"][["scene_id", "disaster", "origin"]]
              .drop_duplicates("scene_id").reset_index(drop=True))

    # Validate the fire list BEFORE using it to classify anything.
    check_fire_events(scenes)
    scenes["is_fire"] = scenes.disaster.isin(FIRE_EVENTS)

    print(f"\n{len(scenes)} scenes, {scenes.is_fire.sum()} wildfire "
          f"({scenes.is_fire.mean():.1%})")
    print("\nscenes by disaster and origin")
    print(pd.crosstab(scenes.disaster, scenes.origin, margins=True).to_string())

    # ---- split -----------------------------------------------------------
    mapping = assign(scenes, args.val_frac, args.seed, args.include_wildfire)
    df = df.merge(mapping, on="scene_id", how="left")

    verify(df, scenes, args.include_wildfire)

    # ---- reporting: these tables belong in the report's dataset section --
    summary = scenes.merge(mapping, on="scene_id")
    print("\nscenes per split")
    print(summary.split.value_counts().to_string())
    print("\ndisaster mix per split")
    print(pd.crosstab(summary.disaster, summary.split).to_string())

    if not args.include_wildfire:
        # Retaining `origin` allows test_id to be reported separately for
        # `test` and `hold`. Since these were independent evaluation sets
        # in the original competition, agreement between them is a free
        # robustness check on a single trained model.
        sub = summary[summary.split == "test_id"]
        print("\ntest_id by official origin (report these separately)")
        print(pd.crosstab(sub.disaster, sub.origin).to_string())

    print("\nfiles per split")
    print(df.groupby(["split", "subdir"]).size()
            .unstack(fill_value=0).to_string())

    if args.dry_run:
        print("\ndry run — nothing written")
        return

    # ---- outputs ---------------------------------------------------------
    # File-level CSV drives dataloaders; scene-level CSV is easier to plot.
    args.out.mkdir(parents=True, exist_ok=True)
    out = df[["name", "path", "subdir", "scene_id", "disaster", "kind",
              "origin", "split"]].sort_values(["split", "name"])
    out.to_csv(args.out / "splits.csv", index=False)
    summary.to_csv(args.out / "splits_summary.csv", index=False)
    print(f"\nwrote {args.out / 'splits.csv'} ({len(out)} rows)")
    print(f"wrote {args.out / 'splits_summary.csv'} ({len(summary)} rows)")

    # Echo the configuration so the run is self-documenting in a log.
    print(f"\nseed={args.seed}  val_frac={args.val_frac}  "
          f"include_wildfire={args.include_wildfire}")


if __name__ == "__main__":
    main()
