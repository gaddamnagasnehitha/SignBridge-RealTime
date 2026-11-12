"""Quick self-check:  python -m backend.smoke_test
1) trains LSTM/GRU/TCN on synthetic data and verifies save/load
2) if MediaPipe is installed, verifies the keypoint vector size on a blank frame
"""
import shutil
import tempfile
from pathlib import Path

import numpy as np

from . import config
from .model import load_checkpoint
from .train import run_training


def main():
    tmp = Path(tempfile.mkdtemp())
    rng = np.random.default_rng(0)
    for c in range(3):
        d = tmp / "seq" / f"sign{c}"
        d.mkdir(parents=True)
        base = rng.normal(size=(config.SEQ_LEN, config.FEATURE_DIM)).astype(np.float32)
        for i in range(12):
            np.save(d / f"{i}.npy", base + 0.1 * rng.normal(size=base.shape).astype(np.float32))
    try:
        for arch in ("lstm", "gru", "tcn"):
            mp = tmp / f"{arch}.pt"
            acc = run_training(tmp / "seq", arch=arch, epochs=8, batch=8, model_path=mp, verbose=False)
            model, ckpt = load_checkpoint(mp)
            assert ckpt["labels"] == ["sign0", "sign1", "sign2"]
            print(f"[ok] {arch}: trained, saved, reloaded (val acc {acc:.2f})")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    try:
        from .keypoints import create_holistic, process_bgr_frame
        h = create_holistic(0)
        vec, hands = process_bgr_frame(h, np.zeros((240, 320, 3), np.uint8))
        h.close()
        assert vec.shape[0] == config.FEATURE_DIM
        print(f"[ok] MediaPipe Holistic works, feature size = {vec.shape[0]}")
    except ImportError:
        print("[warn] mediapipe not installed; skipped keypoint check")
    print("All checks passed.")


if __name__ == "__main__":
    main()
