
---
title: SignBridge
sdk: docker
app_port: 7860
---

# Real-Time Sign Language to Text Translator

MediaPipe Holistic keypoints (pose + face + both hands = 1662 values/frame) -> 30-frame window ->
LSTM / GRU / TCN classifier (PyTorch) -> live text + browser speech synthesis.

```
sign-language-translator/
  backend/    config, keypoints, model, train, predictor, FastAPI server, data tools
  frontend/   index.html, style.css, app.js (webcam -> WebSocket -> text + speech)
  data/       sequences/ (training .npy) and videos/ (raw WLASL / ISL clips)
  models/     sign_model.pt is created by training
```

## 1. Setup (Python 3.9 - 3.12, 64-bit)

Windows (PowerShell):
```
cd sign-language-translator
python -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```
macOS / Linux:
```
cd sign-language-translator
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```
(Smaller CPU-only PyTorch on Linux: `pip install torch --index-url https://download.pytorch.org/whl/cpu` before the last command.)

Verify everything:
```
python -m backend.smoke_test
```
Expected last line: `All checks passed.`

## 2. Get training data (choose A, B, or both)

### A. Record your own signs (fastest, works for a demo)
```
python -m backend.collect_data --signs hello thanks yes no please --num 30
```
Do each sign within the 2-second RECORDING window. Vary position/speed slightly. Press `q` to quit.
Use the same camera setup later for best accuracy. 30+ sequences per sign is recommended.

### B. WLASL
1. Download WLASL (`WLASL_v0.3.json` and the videos) from the official WLASL repository.
2. Select the top glosses and arrange them by folder:
```
python -m backend.prepare_wlasl --json path/to/WLASL_v0.3.json --videos path/to/videos --top 20
```
3. Convert videos to keypoints:
```
python -m backend.process_videos
```

### C. Indian Sign Language (ISL)
Put clips in `data/videos/<sign_name>/*.mp4` (one folder per sign; image-only datasets
must be turned into short videos first), then run `python -m backend.process_videos`.

## 3. Train
```
python -m backend.train --arch lstm --epochs 80
python -m backend.train --arch gru
python -m backend.train --arch tcn
```
The best model is saved to `models/sign_model.pt` (labels are stored inside it).
You need at least 2 signs. If validation accuracy is low, record more samples per sign.

## 4. Run the app
```
python run_server.py
```
Open **http://localhost:8000** (use localhost, not an IP, so the browser allows the camera),
click **Start camera**, allow permission, and sign. Recognized words appear as text and are spoken.

## Tuning (`backend/config.py`)
- `CONF_THRESHOLD` (0.80) and `VOTE_WINDOW` (5): raise for fewer false words, lower for faster response.
- `USE_FACE = False`: lighter/faster model (258 features). You must re-record/re-convert data and retrain after changing it.
- `MODEL_COMPLEXITY = 0`: faster MediaPipe on slow computers.
- Lower hands out of frame for ~1 second between words so the same sign can be recognised again.

## Troubleshooting
- `ModuleNotFoundError: backend`: run commands from the project root with `python -m backend.<name>`.
- `No trained model found`: run training first (step 3).
- Camera not opening in collect_data: try `--camera 1`.
- Model/feature mismatch error: data and model must use the same `USE_FACE` and `SEQ_LEN`.
- `mediapipe` has no `solutions`: you installed a newer mediapipe; run `pip install mediapipe==0.10.14`.
- Port busy: edit the port in `run_server.py`.
