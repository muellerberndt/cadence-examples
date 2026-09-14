# Cadence examples

**Five interactive websites for state, embodiment and learning.**

[Cadence](https://github.com/muellerberndt/cadence) builds observer-like software
patches: bounded local state, declared ports, readback, retained records and local
feedback. These demos make the loop visible and let you change its world.

## Launch any demo

From this checkout, run one command. Python 3.11+ is sufficient: no packages,
account, GPU or training run required. The browser opens automatically and the
server picks an available local port. Ctrl-C stops it.

| Website | Command | Try it |
|---|---|---|
| **Teachable mouse** | `python serve.py mouse` | Teach a new destination, revise the lesson, then transfer it to a new maze |
| **Eye & arm** | `python serve.py eye-arm` | Upload an image, watch an outline emerge, and disturb a joint |
| **Fly-inspired forager** | `python serve.py fly` | Move flowers and change nectar while each encounter updates memory |
| **C. elegans habitat** | `python serve.py worm` | Place food, draw walls, erase a passage; switch to Circuit to inspect neurons |
| **Changing memory** | `python serve.py memory` | Teach once, replace a record, and compare online MLP updates |

`python serve.py` opens the mouse. Use `--no-browser` to print the address, or
`--port 8765` to select a fixed port. Each demo has its own direct URL fragment
and title; the tabs let you move between them. Assets and computation stay local.

## Live composite brains

![The teachable mouse: task memory, a spatial field and a moving body](showcase/preview.png)

The mouse comes with three supplied demonstrations and saves additional lessons
in this browser. The forager and memory demo learn live from a fresh state. The
arm uses a supplied feedback controller. C. elegans includes its public chemical
graph, an engineered habitat/body adapter and a trained MLP circuit comparator.

Each website includes an MRI-inspired view of its actual patch circuit: live
activity, slow repair replay, transient input-release probes where recurrence
exists, and recent associative-weight changes. This is a software visualization,
not an anatomical scan. [Read the circuit view](showcase/README.md#read-the-brain-view).

Every website explains its starting state, how to observe learning or feedback,
and **why Cadence fits the task**, including the relevant conventional controls.
See the [two-minute guide](showcase/README.md#a-two-minute-demonstration),
[starting states](showcase/README.md#ready-to-run-and-watch-learning) and
[comparison methods](showcase/README.md#results-and-comparison-contract).

The habitat follows local food cues around walls and consumes food patches on
contact. Diffusion, gradient heading, movement and consumption are supplied
rules. The chemical circuit gates body movement; this is not validated worm
locomotion or digestion. The fly and mouse bodies are simplified too. The
comparisons measure specific memory, circuit and control tasks.

## Reproduce and test

Serving the websites needs no dependencies. For numerical reproduction:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-reproduce.txt pytest playwright
python showcase/verify.py
node showcase/habitat_benchmark.mjs
python -m pytest -q tests
python -m playwright install chromium
python tools/showcase_pages.py
```

Use Node 20+ for the browser-kernel tests. On Windows, activate the environment
with `.venv\Scripts\Activate.ps1`. Full model training additionally needs PyTorch;
see [reproduction commands](showcase/README.md#reproduce). The evidence retains
sources, budgets, controls and comparison limits. `python tools/build_showcase.py`
regenerates both website entry pages from the same authored shell.

MIT licensed. Public worm data attribution is in [the methods](showcase/README.md#biological-sources-and-data-attribution).
