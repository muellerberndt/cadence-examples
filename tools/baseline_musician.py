"""The conventional control for the musician: a GRU over the same event streams.

Same encoded information per event (the sixteen-event window, the sense row, the mood),
the same five softmax targets, the same held-out streams and the same evaluation. It is
trained by backpropagation through time over spans of the stream. A recurrent
conventional model is the right control for a brain that carries working memory
between events; an MLP without state would be an easier target.
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


class Control(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.embed = nn.Linear(EVENT, 64)
        self.inputs = nn.Linear(WINDOW * 64 + SENSE + MOOD, hidden)
        self.gru = nn.GRUCell(hidden, hidden)
        self.head = nn.Linear(hidden, EVENT)

    def forward(self, context, sense, mood, h):
        e = self.embed(context.view(len(context), WINDOW, EVENT)).flatten(1)
        x = torch.relu(self.inputs(torch.cat([e, sense, mood], 1)))
        h = self.gru(x, h)
        return self.head(h), h


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
    h = torch.zeros(streams.batch, model.gru.hidden_size, device=device)
    total = np.zeros(len(SIZES))
    correct = np.zeros(len(SIZES))
    count = 0
    with torch.no_grad():
        for _ in range(updates):
            context, sense, mood, labels, fresh = streams.batch_rows()
            h[torch.as_tensor(fresh, device=device)] = 0
            logits, h = model(*inputs(context, sense, mood, device), h)
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
    p.add_argument("--name", default="gru-control")
    p.add_argument("--hidden", type=int, default=1024)
    p.add_argument("--updates", type=int, default=30000)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--span", type=int, default=32, help="events per backpropagation-through-time segment")
    p.add_argument("--evaluate-every", type=int, default=1000)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=41)
    a = p.parse_args()
    torch.manual_seed(a.seed)
    device = torch.device(a.device)
    model = Control(a.hidden).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    training = Streams(ROOT / "data" / a.dataset, "train", a.batch, np.random.default_rng(a.seed))
    folder = ROOT / "runs" / a.name
    folder.mkdir(parents=True, exist_ok=True)
    receipt = {
        "model": "GRU control: shared event embedding, one linear input layer, one GRU cell, five softmax heads",
        "hidden": a.hidden,
        "parameters": sum(p.numel() for p in model.parameters()),
        "dataset": a.dataset,
        "dataset_manifest_sha256": hashlib.sha256((ROOT / "data" / a.dataset / "manifest.json").read_bytes()).hexdigest(),
        "batch_streams": a.batch,
        "span": a.span,
        "sources": {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in [Path(__file__), ROOT / "composer/musician.py", ROOT / "tools/train_musician.py"]
        },
        "history": [],
    }
    h = torch.zeros(a.batch, a.hidden, device=device)
    started = time.monotonic()
    best = float("inf")
    accumulated, seen = 0.0, 0
    for update in range(a.updates):
        context, sense, mood, labels, fresh = training.batch_rows()
        h = h.detach() if update % a.span == 0 else h
        h = h * (~torch.as_tensor(fresh, device=device)).float()[:, None]
        logits, h = model(*inputs(context, sense, mood, device), h)
        loss = losses(logits, labels).mean()
        accumulated = accumulated + loss
        seen += 1
        if seen == a.span or update == a.updates - 1:
            optimizer.zero_grad()
            (accumulated / seen).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            accumulated, seen = 0.0, 0
            h = h.detach()
        training.advance()
        if (update + 1) % a.evaluate_every == 0 or update == a.updates - 1:
            measured = evaluate(model, Streams(ROOT / "data" / a.dataset, "validation", 64, np.random.default_rng(7), cap=256), device)
            row = {"update": update + 1, "seconds": time.monotonic() - started, "events_seen": training.events_seen, **measured}
            receipt["history"].append(row)
            if row["mean_nll"] < best:
                best = row["mean_nll"]
                torch.save(model.state_dict(), folder / "control.pt")
            receipt.update(best_validation_nll=best, updates=update + 1, seconds=time.monotonic() - started)
            (folder / "receipt.json").write_text(json.dumps(receipt, indent=2))
            print(json.dumps(row), flush=True)
    receipt["complete"] = True
    (folder / "receipt.json").write_text(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
