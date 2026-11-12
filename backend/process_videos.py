"""Convert sign videos (WLASL / ISL) into keypoint sequences.

Expected input:  data/videos/<sign_name>/<clip>.mp4   (one folder per sign)
Output:          data/sequences/<sign_name>/<clip>.npy
Usage:           python -m backend.process_videos
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

from . import config
from .keypoints import create_holistic, process_bgr_frame

EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def video_to_sequence(path: Path):
    holistic = create_holistic()
    cap = cv2.VideoCapture(str(path))
    kps = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            h, w = frame.shape[:2]
            if w > 640:                      # speed up very large videos
                frame = cv2.resize(frame, (640, int(h * 640 / w)))
            kp, _ = process_bgr_frame(holistic, frame)
            kps.append(kp)
    finally:
        cap.release()
        holistic.close()
    if len(kps) < 5:
        return None
    idx = np.linspace(0, len(kps) - 1, config.SEQ_LEN).round().astype(int)
    return np.stack(kps)[idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(config.VIDEO_DIR))
    ap.add_argument("--dst", default=str(config.SEQ_DIR))
    a = ap.parse_args()
    src, dst = Path(a.src), Path(a.dst)
    if not src.exists():
        raise SystemExit(f"{src} not found")
    total = 0
    for sign_dir in sorted(d for d in src.iterdir() if d.is_dir()):
        out = dst / sign_dir.name
        out.mkdir(parents=True, exist_ok=True)
        for video in sorted(p for p in sign_dir.iterdir() if p.suffix.lower() in EXTS):
            target = out / f"{video.stem}.npy"
            if target.exists():
                continue
            seq = video_to_sequence(video)
            if seq is None:
                print(f"[skip] {video} (too few frames)")
                continue
            np.save(target, seq)
            total += 1
            print(f"saved {target}")
    print(f"Converted {total} videos. Now train: python -m backend.train")


if __name__ == "__main__":
    main()
