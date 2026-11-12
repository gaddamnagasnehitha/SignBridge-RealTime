"""Pick the top-N WLASL glosses and arrange videos into data/videos/<gloss>/.

python -m backend.prepare_wlasl --json path/WLASL_v0.3.json --videos path/videos --top 20
"""
import argparse
import json
import shutil
from pathlib import Path

from . import config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True, help="WLASL_v0.3.json (or nslt-style json with gloss/instances)")
    ap.add_argument("--videos", required=True, help="folder containing <video_id>.mp4 files")
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--min-clips", type=int, default=5)
    ap.add_argument("--dst", default=str(config.VIDEO_DIR))
    a = ap.parse_args()

    with open(a.json, "r", encoding="utf-8") as f:
        entries = json.load(f)
    vid_dir = Path(a.videos)
    available = {p.stem: p for p in vid_dir.glob("*.mp4")}
    if not available:
        raise SystemExit(f"No .mp4 files found in {vid_dir}")

    scored = []
    for e in entries:
        clips = [available[i["video_id"]] for i in e["instances"] if i["video_id"] in available]
        if len(clips) >= a.min_clips:
            scored.append((e["gloss"], clips))
    scored.sort(key=lambda t: len(t[1]), reverse=True)
    chosen = scored[:a.top]
    if not chosen:
        raise SystemExit("No glosses matched. Check --videos path and --min-clips.")

    for gloss, clips in chosen:
        out = Path(a.dst) / gloss.replace(" ", "_")
        out.mkdir(parents=True, exist_ok=True)
        for c in clips:
            shutil.copy2(c, out / c.name)
        print(f"{gloss}: {len(clips)} clips")
    print("Next: python -m backend.process_videos")


if __name__ == "__main__":
    main()
