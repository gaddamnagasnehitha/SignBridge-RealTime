"""Lightweight temporal models: LSTM, GRU, TCN (PyTorch)."""
import torch
import torch.nn as nn

ARCHS = ("lstm", "gru", "tcn")


class SignRNN(nn.Module):
    def __init__(self, input_dim, num_classes, rnn="lstm", hidden=128, layers=2, dropout=0.3):
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        cell = nn.LSTM if rnn == "lstm" else nn.GRU
        self.rnn = cell(input_dim, hidden, num_layers=layers, batch_first=True,
                        dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, 64),
                                  nn.ReLU(), nn.Linear(64, num_classes))

    def forward(self, x):            # x: (B, T, F)
        out, _ = self.rnn(self.norm(x))
        return self.head(out[:, -1])


class _TCNBlock(nn.Module):
    def __init__(self, cin, cout, k, dilation, dropout):
        super().__init__()
        pad = (k - 1) * dilation // 2    # keeps the time length unchanged
        self.net = nn.Sequential(
            nn.Conv1d(cin, cout, k, padding=pad, dilation=dilation), nn.BatchNorm1d(cout),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Conv1d(cout, cout, k, padding=pad, dilation=dilation), nn.BatchNorm1d(cout),
            nn.ReLU(), nn.Dropout(dropout),
        )
        self.res = nn.Conv1d(cin, cout, 1) if cin != cout else nn.Identity()
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(self.net(x) + self.res(x))


class SignTCN(nn.Module):
    def __init__(self, input_dim, num_classes, channels=(128, 128, 128), k=3, dropout=0.3):
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        blocks, cin = [], input_dim
        for i, c in enumerate(channels):
            blocks.append(_TCNBlock(cin, c, k, 2 ** i, dropout))
            cin = c
        self.blocks = nn.Sequential(*blocks)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(cin, num_classes))

    def forward(self, x):            # x: (B, T, F)
        x = self.norm(x).transpose(1, 2)      # (B, F, T)
        x = self.blocks(x).mean(dim=2)        # global average over time
        return self.head(x)


def build_model(arch, input_dim, num_classes):
    arch = arch.lower()
    if arch in ("lstm", "gru"):
        return SignRNN(input_dim, num_classes, rnn=arch)
    if arch == "tcn":
        return SignTCN(input_dim, num_classes)
    raise ValueError(f"Unknown arch '{arch}'. Choose from {ARCHS}")


def save_checkpoint(path, model, arch, input_dim, seq_len, labels):
    path = str(path)
    torch.save({
        "state_dict": model.state_dict(),
        "arch": arch,
        "input_dim": int(input_dim),
        "seq_len": int(seq_len),
        "labels": list(labels),
    }, path)


def load_checkpoint(path, device="cpu"):
    ckpt = torch.load(str(path), map_location=device, weights_only=True)
    model = build_model(ckpt["arch"], ckpt["input_dim"], len(ckpt["labels"]))
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model, ckpt
