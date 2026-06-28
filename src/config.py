"""Shared project configuration. Import from every notebook so settings stay consistent."""
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

# --- cross-disaster split (NOT a random split — this is the graded design) --
# Train/val on these disaster types ...
TRAIN_DISASTERS = ["hurricane", "flooding", "earthquake"]
# ... and hold out this one entirely for testing (transfer question).
HELDOUT_DISASTER = "wildfire"

# --- patches & training ----------------------------------------------------
PATCH_SIZE = 128      # px, square patches resized from building bbox crops
BATCH_SIZE = 64
EPOCHS = 20
LR = 1e-3
SEED = 42
