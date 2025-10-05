from pathlib import Path

# Paths
SCRIPT_DIR = Path(__file__).resolve().parents[2]
BASE_DIR = SCRIPT_DIR / "data" / "CVC-ClinicDB"
META_DATA = SCRIPT_DIR / "data" / "metadata.csv"
CHECKPOINT_PATH = SCRIPT_DIR / "checkpoint" / "best_model_weights.pth"
HISTORY_PLOT_PATH = SCRIPT_DIR / "history.png"
OUTPUT_DIR = SCRIPT_DIR / "prediction"

# Loader Directory
NEW_SIZE = (256, 256)

# Train Test Split
TEST_SIZE = 0.15
RANDOM_STATE = 0

# Model
NUM_SEQUENCES = 29
BATCH_SIZE = 16
FILM_LAYER = -1
EMB_DIM = 32

# Training
EPOCHS = 35

# Optimizer
WEIGHT_DECAY = 1e-4
LR = 1e-4

# Scheduler
MODE = "min"
FACTOR = 0.5
PATIENCE = 5

# Plot
FIG_SIZE = (25, 7)
