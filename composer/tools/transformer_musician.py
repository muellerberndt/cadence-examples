"""A second conventional control for the musician: a small transformer over the same
event streams.

Same encoded information per event as the GRU control and the musician's clamped ports
(the sixteen-event window, the sense row, the mood), the same five softmax targets, the
same held-out streams and the same evaluation. Each of the sixteen heard events is one
token; the sense row and the mood make a seventeenth token. Two pre-norm encoder layers
attend over the seventeen tokens and the last position is read by the five heads. There
is no recurrent state: unlike the GRU, the transformer sees only its window, as the
musician's ear does (the musician additionally carries its working memory and form record
between events). The parameter count is set to exceed the GRU control's 7,640,370.
"""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from composer.musician import EVENT, MOOD, SENSE, SIZES, WINDOW, encode_events, encode_mood, encode_sense
from tools.train_musician import Streams

ROOT = Path(__file__).resolve().parents[1]


class TransformerControl(nn.Module):
    def __init__(self, width=512, layers=2, heads=8, feedforward=2816):
        super().__init__()
        self.width = width
        self.embed = nn.Linear(EVENT, width)
        self.context = nn.Linear(SENSE + MOOD, width)
        self.position = nn.Parameter(torch.zeros(WINDOW + 1, width))
        nn.init.normal_(self.position, std=0.02)
        layer = nn.TransformerEncoderLayer(width, heads, feedforward, dropout=0.0, batch_first=True, norm_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, layers)
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, EVENT)

    def forward(self, context, sense, mood, h=None):
        e = self.embed(context.view(len(context), WINDOW, EVENT))
        c = self.context(torch.cat([sense, mood], 1))[:, None, :]
        x = torch.cat([e, c], 1) + self.position
        x = self.encoder(x)
        return self.head(self.norm(x[:, -1])), h


def inputs(context, sense, mood, device):
    return (
        torch.as_tensor(encode_events(context), device=device),
        torch.as_tensor(encode_sense(sense), device=device),
        torch.as_tensor(encode_mood(mood), device=device),
    )


def losses(logits, labels):
    labels = torch.as_tensor(labels, device=logits.device)
    out, offset = [], 0
    for k, width in enumerate(SIZES):
        out.append(nn.functional.cross_entropy(logits[:, offset : offset + width], labels[:, k], reduction="none"))
        offset += width
    return torch.stack(out, 1)


def evaluate(model, streams, device, updates=192):
    model.eval()
    total = np.zeros(len(SIZES))
    correct = np.zeros(len(SIZES))
    count = 0
    with torch.no_grad():
        for _ in range(updates):
            context, sense, mood, labels, _fresh, _conditioning = streams.batch_rows()
            logits, _ = model(*inputs(context, sense, mood, device))
            total += losses(logits, labels).sum(0).cpu().numpy()
            offset = 0
            for k, width in enumerate(SIZES):
                correct[k] += (logits[:, offset : offset + width].argmax(1).cpu().numpy() == labels[:, k]).sum()
                offset += width
            count += len(labels)
            streams.advance()
    model.train()
    return {
        "nll": (total / count).tolist(),
        "mean_nll": float(total.sum() / count / len(SIZES)),
        "accuracy": (correct / count).tolist(),
        "events": int(count),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="musician")
    p.add_argument("--name", default="transformer-control")
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--feedforward", type=int, default=2816)
    p.add_argument("--updates", type=int, default=60000)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup", type=int, default=1000)
    p.add_argument("--evaluate-every", type=int, default=5000)
    p.add_argument("--max-seconds", type=int, default=1800, help="bounded training budget")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=41)
    a = p.parse_args()
    torch.manual_seed(a.seed)
    device = torch.device(a.device)
    model = TransformerControl(a.width, a.layers, a.heads, a.feedforward).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda u: min(1.0, (u + 1) / a.warmup) * max(0.05, 1 - u / a.updates))
    training = Streams(ROOT / "data" / a.dataset, "train", a.batch, np.random.default_rng(a.seed))
    folder = ROOT / "runs" / a.name
    folder.mkdir(parents=True, exist_ok=True)
    receipt = {
        "model": "transformer control: per-event embedding of the sixteen heard events plus one sense-and-mood token, learned positions, %d pre-norm encoder layers, five softmax heads read at the last position; no recurrent state" % a.layers,
        "width": a.width,
        "layers": a.layers,
        "heads": a.heads,
        "feedforward": a.feedforward,
        "parameters": sum(p.numel() for p in model.parameters()),
        "dataset": a.dataset,
        "dataset_manifest_sha256": hashlib.sha256((ROOT / "data" / a.dataset / "manifest.json").read_bytes()).hexdigest(),
        "batch_streams": a.batch,
        "lr": a.lr,
        "warmup": a.warmup,
        "max_seconds": a.max_seconds,
        "sources": {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in [Path(__file__), ROOT / "composer/musician.py", ROOT / "tools/train_musician.py"]
        },
        "history": [],
    }
    print(json.dumps({"parameters": receipt["parameters"]}), flush=True)
    started = time.monotonic()
    best = float("inf")
    for update in range(a.updates):
        context, sense, mood, labels, _fresh, _conditioning = training.batch_rows()
        logits, _ = model(*inputs(context, sense, mood, device))
        loss = losses(logits, labels).mean()
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        schedule.step()
        training.advance()
        out_of_time = time.monotonic() - started > a.max_seconds
        if (update + 1) % a.evaluate_every == 0 or update == a.updates - 1 or out_of_time:
            measured = evaluate(model, Streams(ROOT / "data" / a.dataset, "validation", 64, np.random.default_rng(7), cap=256), device)
            row = {"update": update + 1, "seconds": time.monotonic() - started, "events_seen": training.events_seen, "train_loss": float(loss), **measured}
            receipt["history"].append(row)
            if row["mean_nll"] < best:
                best = row["mean_nll"]
                torch.save(model.state_dict(), folder / "control.pt")
            receipt.update(best_validation_nll=best, updates=update + 1, seconds=time.monotonic() - started)
            (folder / "receipt.json").write_text(json.dumps(receipt, indent=2))
            print(json.dumps(row), flush=True)
        if out_of_time:
            receipt["stopped"] = "time budget"
            break
    receipt["complete"] = True
    (folder / "receipt.json").write_text(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
