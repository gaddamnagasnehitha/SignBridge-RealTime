"""Record your own sign data from the webcam.

python -m backend.collect_data --signs hello thanks yes no --num 30
A window opens: 'GET READY' (1.5 s) then 'RECORDING' (2 s, 30 frames). Press q to quit.
"""
import argparse
import sys
import time

import cv2
import numpy as np

from . import config
from .keypoints import create_holistic, process_bgr_frame


def put(frame, text, color=(0, 255, 0), y=30):
    cv2.putText(frame, text, (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--signs", nargs="+", required=True, help="sign names, e.g. hello thanks")
    ap.add_argument("--num", type=int, default=30, help="sequences to record per sign")
    ap.add_argument("--camera", type=int, default=0)
    a = ap.parse_args()

    cap = cv2.VideoCapture(a.camera)
    if not cap.isOpened():
        sys.exit("Cannot open camera. Try --camera 1 or close other apps using it.")
    holistic = create_holistic()
    interval = 1.0 / config.TARGET_FPS

    try:
        for sign in a.signs:
            out = config.SEQ_DIR / sign
            out.mkdir(parents=True, exist_ok=True)
            start = len(list(out.glob("*.npy")))
            for n in range(a.num):
                t0 = time.time()
                while time.time() - t0 < 1.5:
                    ok, frame = cap.read()
                    if not ok:
                        continue
                    put(frame, f"GET READY: '{sign}'  ({n + 1}/{a.num})", (0, 200, 255))
                    cv2.imshow("Collect", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        return
                seq = []
                while len(seq) < config.SEQ_LEN:
                    t_frame = time.time()
                    ok, frame = cap.read()
                    if not ok:
                        continue
                    kp, hands = process_bgr_frame(holistic, frame)
                    seq.append(kp)
                    put(frame, f"RECORDING '{sign}' {len(seq)}/{config.SEQ_LEN}", (0, 0, 255))
                    put(frame, "hands: yes" if hands else "hands: NO", (255, 255, 255), 65)
                    cv2.imshow("Collect", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        return
                    wait = interval - (time.time() - t_frame)
                    if wait > 0:
                        time.sleep(wait)
                np.save(out / f"{start + n}.npy", np.stack(seq))
        print("Done. Now train: python -m backend.train")
    finally:
        holistic.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
