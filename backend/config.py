"""Central configuration. Change values here; every script imports them."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
SEQ_DIR = DATA_DIR / "sequences"      # data/sequences/<sign>/<n>.npy  (30 x FEATURE_DIM)
VIDEO_DIR = DATA_DIR / "videos"       # data/videos/<sign>/<clip>.mp4  (for WLASL / ISL)
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "sign_model.pt"
FRONTEND_DIR = ROOT / "frontend"

# ---- Sequence / feature settings (training and inference MUST match) ----
SEQ_LEN = 30            # frames per gesture window
TARGET_FPS = 15         # capture + live rate (30 frames = 2 seconds)
USE_FACE = True         # include 468 face landmarks (1404 values)

POSE_DIM = 33 * 4       # x, y, z, visibility
FACE_DIM = 468 * 3
HAND_DIM = 21 * 3
FEATURE_DIM = POSE_DIM + (FACE_DIM if USE_FACE else 0) + 2 * HAND_DIM

# ---- Live inference settings ----
CONF_THRESHOLD = 0.80   # minimum softmax confidence to accept a prediction
VOTE_WINDOW = 5         # same label must win this many consecutive predictions
IDLE_RESET_FRAMES = 10  # frames without hands before the window resets
MODEL_COMPLEXITY = 1    # MediaPipe Holistic: 0 (fast), 1 (balanced), 2 (accurate)
