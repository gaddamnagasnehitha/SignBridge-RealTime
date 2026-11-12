"""FastAPI server: serves the frontend and a WebSocket that turns webcam frames into text.

Run from the project root:   python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000
"""
import asyncio
import json
from collections import deque

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from . import config
from .keypoints import create_holistic, process_bgr_frame
from .predictor import SignPredictor

app = FastAPI(title="Real-Time Sign Language Translator")
predictor = SignPredictor()
app.mount("/static", StaticFiles(directory=str(config.FRONTEND_DIR)), name="static")


@app.get("/")
def index():
    return FileResponse(config.FRONTEND_DIR / "index.html")


@app.get("/api/status")
def status():
    return {"model_ready": predictor.ready, "labels": predictor.labels, "error": predictor.error,
            "seq_len": config.SEQ_LEN, "target_fps": config.TARGET_FPS}


class Session:
    """Per-connection state: keypoint window, vote buffer, sentence."""

    def __init__(self):
        self.holistic = create_holistic()
        self.buffer = deque(maxlen=config.SEQ_LEN)
        self.votes = deque(maxlen=config.VOTE_WINDOW)
        self.sentence = []
        self.no_hand = 0
        self.last_emitted = None

    def reset_window(self):
        self.buffer.clear()
        self.votes.clear()

    def clear(self):
        self.reset_window()
        self.sentence = []
        self.last_emitted = None

    def process(self, data: bytes) -> dict:
        frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            return {"status": "bad_frame", "sentence": " ".join(self.sentence)}
        keypoints, hands = process_bgr_frame(self.holistic, frame)

        if hands:
            self.no_hand = 0
        else:
            self.no_hand += 1
        if self.no_hand >= config.IDLE_RESET_FRAMES:
            self.reset_window()
            self.last_emitted = None          # allow the same sign again after a pause
            return {"status": "idle", "sentence": " ".join(self.sentence)}

        self.buffer.append(keypoints)
        if len(self.buffer) < config.SEQ_LEN:
            return {"status": "collecting", "progress": len(self.buffer) / config.SEQ_LEN,
                    "sentence": " ".join(self.sentence)}

        label, conf = predictor.predict(np.stack(self.buffer))
        self.votes.append(label if conf >= config.CONF_THRESHOLD else None)
        out = {"status": "predicting", "label": label, "confidence": round(conf, 3), "new": False}

        if (len(self.votes) == config.VOTE_WINDOW and self.votes[0] is not None
                and len(set(self.votes)) == 1 and label != self.last_emitted):
            self.sentence.append(label.replace("_", " "))
            self.last_emitted = label
            self.votes.clear()
            out["new"] = True
            out["word"] = label.replace("_", " ")
        out["sentence"] = " ".join(self.sentence)
        return out

    def close(self):
        try:
            self.holistic.close()
        except Exception:
            pass


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    if not predictor.ready:
        await ws.send_text(json.dumps({"status": "no_model", "message": predictor.error}))
        await ws.close()
        return
    session = await asyncio.to_thread(Session)
    try:
        while True:
            msg = await ws.receive()
            if msg["type"] == "websocket.disconnect":
                break
            if msg.get("bytes"):
                result = await asyncio.to_thread(session.process, msg["bytes"])
                await ws.send_text(json.dumps(result))
            elif msg.get("text") == "clear":
                session.clear()
                await ws.send_text(json.dumps({"status": "cleared", "sentence": ""}))
    except WebSocketDisconnect:
        pass
    finally:
        session.close()
