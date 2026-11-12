"""Train the sign classifier.   python -m backend.train --arch lstm --epochs 80"""
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from . import config
from .dataset import load_dataset
from .model import ARCHS, build_model, save_checkpoint


def stratified_split(y, val_split, rng):
    train_idx, val_idx = [], []
    for c in np.unique(y):
        idx = rng.permutation(np.where(y == c)[0])
        if len(idx) >= 2:
            n_val = min(max(1, int(round(len(idx) * val_split))), len(idx) - 1)
        else:
            n_val = 0
        val_idx += idx[:n_val].tolist()
        train_idx += idx[n_val:].tolist()
    return np.array(train_idx), np.array(val_idx, dtype=int)


def evaluate(model, X, y, device):
    model.eval()
    with torch.no_grad():
        pred = model(X.to(device)).argmax(1).cpu()
    return (pred == y).float().mean().item()


def run_training(seq_dir=config.SEQ_DIR, arch="lstm", epochs=80, batch=32, lr=1e-3,
                 val_split=0.2, seed=42, patience=20, model_path=config.MODEL_PATH, verbose=True):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    X, y, labels = load_dataset(seq_dir)
    if verbose:
        print(f"Loaded {len(X)} sequences, {len(labels)} classes: {labels}")
    tr, va = stratified_split(y, val_split, rng)
    Xtr, ytr = torch.from_numpy(X[tr]), torch.from_numpy(y[tr])
    if len(va):
        Xva, yva = torch.from_numpy(X[va]), torch.from_numpy(y[va])
    else:
        Xva, yva = Xtr, ytr

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(arch, X.shape[2], len(labels)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=6)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05)

    best_acc, best_state, bad = -1.0, None, 0
    n = len(Xtr)
    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n)
        total = 0.0
        for i in range(0, n, batch):
            b = perm[i:i + batch]
            if len(b) < 2:          # BatchNorm (TCN) needs >1 sample
                continue
            xb = Xtr[b].to(device)
            xb = xb + 0.005 * torch.randn_like(xb)      # light augmentation
            yb = ytr[b].to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            total += loss.item() * len(b)
        acc = evaluate(model, Xva, yva, device)
        sched.step(total / n)
        if verbose and (ep == 1 or ep % 5 == 0):
            print(f"epoch {ep:3d}  loss {total / n:.4f}  val_acc {acc:.3f}")
        if acc > best_acc:
            best_acc, bad = acc, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                if verbose:
                    print(f"Early stopping at epoch {ep}")
                break

    model.load_state_dict(best_state)
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint(model_path, model.cpu(), arch, X.shape[2], X.shape[1], labels)
    if verbose:
        print(f"Best val accuracy: {best_acc:.3f}\nSaved model to {model_path}")
    return best_acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="lstm", choices=ARCHS)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-split", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--data", default=str(config.SEQ_DIR))
    a = ap.parse_args()
    run_training(a.data, a.arch, a.epochs, a.batch, a.lr, a.val_split, a.seed, a.patience)


if __name__ == "__main__":
    main()
