"""Loads the trained model and classifies a (SEQ_LEN, FEATURE_DIM) window."""
import numpy as np
import torch

from . import config
from .model import load_checkpoint


class SignPredictor:
    def __init__(self, model_path=config.MODEL_PATH):
        self.ready = False
        self.labels = []
        self.error = None
        try:
            if not model_path.exists():
                self.error = "No trained model found. Run: python -m backend.train"
                return
            self.model, ckpt = load_checkpoint(model_path)
            self.labels = ckpt["labels"]
            if ckpt["input_dim"] != config.FEATURE_DIM or ckpt["seq_len"] != config.SEQ_LEN:
                self.error = ("Model was trained with different SEQ_LEN/USE_FACE settings "
                              "than backend/config.py. Retrain or restore the settings.")
                return
            torch.set_num_threads(2)
            self.ready = True
        except Exception as exc:  # corrupted file, version mismatch, ...
            self.error = f"Could not load model: {exc}"

    @torch.no_grad()
    def predict(self, seq: np.ndarray):
        x = torch.from_numpy(seq.astype(np.float32)).unsqueeze(0)
        probs = torch.softmax(self.model(x), dim=1)[0]
        conf, idx = probs.max(0)
        return self.labels[int(idx)], float(conf)
