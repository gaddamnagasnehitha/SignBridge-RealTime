"""Load saved keypoint sequences from data/sequences/<label>/*.npy"""
from pathlib import Path

import numpy as np

from . import config


def load_dataset(seq_dir: Path = config.SEQ_DIR, seq_len: int = config.SEQ_LEN,
                 feature_dim: int = config.FEATURE_DIM):
    seq_dir = Path(seq_dir)
    if not seq_dir.exists():
        raise FileNotFoundError(f"{seq_dir} does not exist")
    labels = sorted(d.name for d in seq_dir.iterdir() if d.is_dir() and any(d.glob("*.npy")))
    if len(labels) < 2:
        raise RuntimeError(
            f"Need at least 2 sign folders with .npy files in {seq_dir}, found {len(labels)}. "
            "Collect data (backend.collect_data) or convert videos (backend.process_videos) first."
        )
    X, y = [], []
    for idx, label in enumerate(labels):
        for f in sorted((seq_dir / label).glob("*.npy")):
            arr = np.load(f)
            if arr.shape != (seq_len, feature_dim):
                print(f"[skip] {f} has shape {arr.shape}, expected {(seq_len, feature_dim)} "
                      "(was it recorded with a different USE_FACE / SEQ_LEN setting?)")
                continue
            X.append(arr.astype(np.float32))
            y.append(idx)
    if not X:
        raise RuntimeError("No valid sequences were loaded.")
    return np.stack(X), np.asarray(y, dtype=np.int64), labels
