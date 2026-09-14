import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from composer.compose import Composer

p = argparse.ArgumentParser()
p.add_argument("prompt", nargs="?", default="A sad gentle piano piece")
p.add_argument("--checkpoint")
p.add_argument("--seed", type=int, default=17)
p.add_argument("--variants", type=int, default=6)
p.add_argument("--no-audio", action="store_true")
a = p.parse_args()
r = Composer(a.checkpoint).compose(
    a.prompt,
    seed=a.seed,
    variants=a.variants,
    render_audio=not a.no_audio,
    progress=lambda e: print(
        json.dumps({k: v for k, v in e.items() if k != "trace"}), flush=True
    ),
)
print(
    json.dumps(
        {k: r[k] for k in ["id", "before", "after", "files", "brain", "seconds"]},
        indent=2,
    )
)
