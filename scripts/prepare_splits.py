"""
Prepare train, validation and test splits for the xBD dataset.

The dataset is stored in four folders: hold, test, tier1 and tier3.
This script scans these folders, checks the files and creates a CSV
with the split assigned to each scene.

The main split setup is:

    train     tier1 + tier3, without wildfire scenes
    val       tier1 + tier3, without wildfire scenes
    test_id   test + hold, without wildfire scenes
    test_ood  wildfire scenes

Wildfire scenes are kept out of training because we want to test how
well the model performs on a disaster type that it has not seen before.

The script does not move the original files. It only saves their paths
and split information to CSV files.
"""

import argparse
import os
import re
import shutil
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


# CONSTANTS FOR THE DATASET
# These were established by inspecting the data interactively, then frozen here. 
# Hard-coding them (rather than re-deriving them at runtime) means a change in the data causes a loud failure instead of a silently different split.


# Main xBD folders
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

# Extra check for possible wildfire event names
FIRE_KEYWORDS = ("fire", "bushfire", "wildfire")

# Warn if the ID test set is too small
MIN_TEST_ID_SCENES = 300

# Filename grammar: the disaster name, a zero-padded scene index, then a "kind" suffix
NAME_PAT = re.compile(
    r"^(?P<disaster>[a-z\-]+?)_"     # e.g. santa-rosa-wildfire
    r"(?P<idx>\d+)_"                 # e.g. 00000123
    r"(?P<kind>.+)$"                 # e.g. post_disaster
)


def fail(msg: str) -> None:
    """Print an error message and stop the script."""
    sys.exit(f"ERROR: {msg}")


# STEP 1 — MERGE: Scan dataset folders

def scan(src: Path) -> pd.DataFrame:
    """Scan the dataset folders and collect information about every file.

    Args:
    src: Root folder of the xBD dataset.

    Returns:
        DataFrame containing file paths and basic metadata.
    """

    rows = []
    for origin in ORIGINS:
        for sub in SUBDIRS:
            d = src / origin / sub
            if not d.is_dir():
                print(f"  ! missing directory, skipping: {d}", file=sys.stderr)
                continue
            for p in sorted(d.iterdir()):
                # Ignore hidden system files such as .DS_Store
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
    """Check that the same filename does not appear in different folders.

    Args:
        df: DataFrame returned by scan().

    Raises:
        Stops the script if duplicate filenames are found.
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


# STEP 2 — Parse filenames and check dataset


def parse_names(df: pd.DataFrame) -> pd.DataFrame:
    """Extract disaster name, scene ID and file type from each filename.

    Args:
        df: DataFrame with file names.

    Returns:
        DataFrame with additional parsed columns.
    """
    def one(name):
        m = NAME_PAT.match(Path(name).stem)
        if not m:
            return pd.Series({"disaster": None, "scene_id": None, "kind": None})
        g = m.groupdict()
        return pd.Series({
            "disaster": g["disaster"],
            # Scene ID is based only on the filename.
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
    """Check that every scene has the expected file structure.

    The function counts how many file types belong to each scene and
    reports scenes that are missing files or contain unexpected ones.

    Args:
        df: Parsed dataset index with scene IDs and file types.
    """

    inv = df.groupby("subdir").kind.value_counts().unstack(fill_value=0).T
    print("\nfile kinds per subdirectory")
    print(inv.to_string())

    for sub in SUBDIRS:
        s = df[df.subdir == sub]
        if s.empty:
            fail(f"no files found for subdir '{sub}'")

        sizes = s.groupby("scene_id").kind.nunique()
        expected = sizes.mode().iat[0]      # most common number of files per scene
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

    # Just print a warning if an expected event is missing
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


# STEP 3 — Create dataset splits

def assign(scenes: pd.DataFrame, val_frac: float, seed: int,
           include_wildfire: bool) -> pd.DataFrame:
    """Assign each scene to train, validation or test.

    Train and validation splits are created randomly.
    The test sets are determined by the dataset structure:
    official test/hold scenes become test_id and wildfire
    scenes become test_ood.

    Args:
        scenes: DataFrame with one row per scene.
        val_frac: Fraction of training data used for validation.
        seed: Random seed for reproducibility.
    include_wildfire: If True, wildfire scenes stay in training.

    Returns:
        DataFrame containing scene IDs and assigned splits.
    """
    trainable = scenes[scenes.origin.isin(TRAIN_ORIGINS)]
    official = scenes[scenes.origin.isin(OFFICIAL_ORIGINS)]

    if include_wildfire:
        # Standard protocol: nothing is held out by disaster type
        pool, ood = trainable.copy(), scenes.iloc[0:0]
        test_id = official.copy()
    else:
        # Hold out all wildfire scenes for OOD evaluation
        pool = trainable[~trainable.is_fire].copy()
        test_id = official[~official.is_fire].copy()
        ood = scenes[scenes.is_fire].copy()

    if len(pool) < 10:
        fail(f"training pool has only {len(pool)} scenes")

    # Keep a similar disaster distribution in train and validation
    # Events with only one scene cannot be stratified, so they are added directly to the training set
    counts = pool.disaster.value_counts()
    singleton = pool[pool.disaster.map(counts) < 2]
    splittable = pool[pool.disaster.map(counts) >= 2].copy()

    if len(singleton):
        print(f"\n{len(singleton)} scene(s) from single-scene events sent "
              f"directly to train: {sorted(singleton.disaster.unique())}")

    # Disable stratification if the validation set is too small
    n_classes = splittable.disaster.nunique()
    n_val = int(round(len(splittable) * val_frac))
    strat = splittable.disaster
    if n_val < n_classes:
        print(f"! val would hold {n_val} scenes but there are {n_classes} "
              f"events; splitting without stratification. Raise --val-frac "
              f"to restore a balanced event mix.", file=sys.stderr)
        strat = None

    tr, va = train_test_split(splittable, test_size=val_frac, random_state=seed, stratify=strat)

    return pd.concat([
        tr.assign(split="train"),
        singleton.assign(split="train"),
        va.assign(split="val"),
        test_id.assign(split="test_id"),
        ood.assign(split="test_ood"),
    ])[["scene_id", "split"]]


# STEP 4 — Verify the splits

def verify(df: pd.DataFrame, scenes: pd.DataFrame, include_wildfire: bool) -> None:
    """Run basic checks on the generated splits.

    Args:
        df: File-level DataFrame with assigned splits.
        scenes: Scene-level DataFrame.
        include_wildfire: Whether wildfire is included in training.

    Exits:
        On unassigned files, scene-level leakage, or wildfire found in
        train/val when it should have been held out.
    """
    # 1. Every file must have a split
    n = df.split.isna().sum()
    if n:
        fail(f"{n} file(s) received no split")

    # 2. Make sure the same scene is not assigned to multiple splits
    straddle = df.groupby("scene_id").split.nunique()
    if (straddle > 1).any():
        fail(f"{(straddle > 1).sum()} scene(s) appear in more than one split")

    if not include_wildfire:
        # 3. The experiment's core premise: wildfire scenes must not be in train/val
        img = df[df.subdir == "images"].merge(
            scenes[["scene_id", "is_fire"]], on="scene_id")
        leaked = img[img.split.isin(["train", "val"]) & img.is_fire]
        if len(leaked):
            fail(f"{leaked.scene_id.nunique()} wildfire scene(s) in train/val")

        # 4. Warn if the ID test set is very small.
        n_id = df[df.split == "test_id"].scene_id.nunique()
        if n_id < MIN_TEST_ID_SCENES:
            print(f"\n! test_id has only {n_id} scenes (< {MIN_TEST_ID_SCENES}). "
                  f"The ID -> OOD comparison will be noisy; consider moving "
                  f"some non-fire tier1/tier3 scenes into it.", file=sys.stderr)

    print("\nall integrity checks passed")


# ENTRY POINT

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

    # ---- merge ----
    df = scan(args.src)
    check_collisions(df)
    df = parse_names(df)
    check_structure(df)

    # Use image files to create one record for each scene
    scenes = (df[df.subdir == "images"][["scene_id", "disaster", "origin"]]
              .drop_duplicates("scene_id").reset_index(drop=True))

    # Check that the wildfire event list is complete
    check_fire_events(scenes)
    scenes["is_fire"] = scenes.disaster.isin(FIRE_EVENTS)

    print(f"\n{len(scenes)} scenes, {scenes.is_fire.sum()} wildfire "
          f"({scenes.is_fire.mean():.1%})")
    print("\nscenes by disaster and origin")
    print(pd.crosstab(scenes.disaster, scenes.origin, margins=True).to_string())

    # ---- split ----
    mapping = assign(scenes, args.val_frac, args.seed, args.include_wildfire)
    df = df.merge(mapping, on="scene_id", how="left")

    verify(df, scenes, args.include_wildfire)

    # ---- print split statistics ----
    summary = scenes.merge(mapping, on="scene_id")
    print("\nscenes per split")
    print(summary.split.value_counts().to_string())
    print("\ndisaster mix per split")
    print(pd.crosstab(summary.disaster, summary.split).to_string())

    if not args.include_wildfire:
        # Show the distribution of test_id scenes by original folder
        sub = summary[summary.split == "test_id"]
        print("\ntest_id by official origin (report these separately)")
        print(pd.crosstab(sub.disaster, sub.origin).to_string())

    print("\nfiles per split")
    print(df.groupby(["split", "subdir"]).size()
            .unstack(fill_value=0).to_string())

    if args.dry_run:
        print("\ndry run — nothing written")
        return

    # ---- outputs ----
    # File-level CSV drives dataloaders; scene-level CSV is easier to plot
    args.out.mkdir(parents=True, exist_ok=True)
    out = df[["name", "path", "subdir", "scene_id", "disaster", "kind",
              "origin", "split"]].sort_values(["split", "name"])
    out.to_csv(args.out / "splits.csv", index=False)
    summary.to_csv(args.out / "splits_summary.csv", index=False)
    print(f"\nwrote {args.out / 'splits.csv'} ({len(out)} rows)")
    print(f"wrote {args.out / 'splits_summary.csv'} ({len(summary)} rows)")

    # Print the settings used for this run
    print(f"\nseed={args.seed}  val_frac={args.val_frac}  "
          f"include_wildfire={args.include_wildfire}")


if __name__ == "__main__":
    main()
