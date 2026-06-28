"""Shared project configuration. Imported by main.ipynb (and any helper code)."""
from pathlib import Path

# --- paths -----------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = str(PROJECT_ROOT / "data")
CKPT_DIR = str(PROJECT_ROOT / "checkpoints")

# --- damage labels (xBD 4-level scale) -------------------------------------
DAMAGE_CLASSES = ["no-damage", "minor-damage", "major-damage", "destroyed"]
NAME_TO_IDX = {name: i for i, name in enumerate(DAMAGE_CLASSES)}
IDX_TO_NAME = {i: name for i, name in enumerate(DAMAGE_CLASSES)}
NUM_CLASSES = len(DAMAGE_CLASSES)

# --- cross-disaster split (NOT random — this is the graded design) ----------
# Train/val on these disaster types ...
TRAIN_DISASTERS = ["hurricane", "flood", "earthquake"]
# ... and hold out this one entirely for testing (the transfer question).
HELDOUT_DISASTER = "wildfire"

# --- patches & training ----------------------------------------------------
PATCH_SIZE = 64       # px; small keeps CPU training fast
BATCH_SIZE = 64
EPOCHS = 5
LR = 1e-3
SEED = 42

# xBD event names encode the disaster type by keyword, e.g.
#   "hurricane-florence", "midwest-flooding", "santa-rosa-wildfire", "socal-fire".
# We can't just take the first token, so map by keyword instead.
_DISASTER_KEYWORDS = {
    "hurricane": "hurricane",
    "flood": "flood",
    "fire": "wildfire", "wildfire": "wildfire", "bushfire": "wildfire",
    "earthquake": "earthquake",
    "tsunami": "tsunami",
    "volcano": "volcano",
    "tornado": "tornado",
}


def event_to_disaster_type(event_name: str) -> str:
    """'santa-rosa-wildfire' -> 'wildfire', 'hurricane-florence' -> 'hurricane'."""
    name = event_name.lower()
    for keyword, dtype in _DISASTER_KEYWORDS.items():
        if keyword in name:
            return dtype
    return "other"
