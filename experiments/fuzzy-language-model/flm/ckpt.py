"""Checkpoint / resume and Weights & Biases logging for `flm.train` (`PLAN_PRECISION.md`).

A training checkpoint holds everything needed to continue a run so that it ends exactly where an uninterrupted run would have: the model, the optimiser (moments), the next step, the numpy generator that draws the batches, the torch generators, the EMA copy, the loss-scaler of fp16 training, the curves so far and the training seconds so far.
It is written atomically (a temporary file, then a rename) every ``--ckpt-every`` steps into ``--ckpt-dir`` as ``step_<n>.ckpt`` (the newest ``--ckpt-keep`` are kept), and ``--resume`` continues from the newest one (``--resume auto``), from a given file, or from a Weights & Biases artifact (``--resume wandb:<entity>/<project>/<name>:latest``).
A resume refuses to start if the run's arguments differ from the ones the checkpoint was written with (anything that changes the model, the data order or the schedule), so a run cannot silently continue as a different experiment.
Weights & Biases is optional and imported lazily: ``--wandb <project>`` logs the loss, learning rate, speed and validation curve of the run, a resumed run continues the same W&B run (its id is stored in the checkpoint), and ``--wandb-artifacts`` also uploads every checkpoint as an artifact.  ``WANDB_API_KEY`` (or ``wandb login``) and ``WANDB_MODE=offline`` are the usual W&B settings; this module never handles credentials.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import torch

# arguments that may differ between the original run and its resume (where it runs, what it writes, how it is watched)
IGNORE = {
    "out",
    "device",
    "threads",
    "save_model",
    "resume",
    "ckpt_dir",
    "ckpt_every",
    "ckpt_keep",
    "wandb",
    "wandb_name",
    "wandb_artifacts",
    "eval_every",
    "eval_chars",
    "eval_at_start",
}


def run_key(args) -> dict:
    return {k: v for k, v in vars(args).items() if k not in IGNORE}


def check_compatible(saved: dict, args) -> None:
    now = run_key(args)
    diff = {
        k: (saved.get(k), now.get(k))
        for k in set(saved) | set(now)
        if saved.get(k) != now.get(k)
    }
    if diff:
        lines = "\n".join(
            f"  {k}: checkpoint {a!r}, now {b!r}" for k, (a, b) in sorted(diff.items())
        )
        raise SystemExit(
            "refusing to resume: the arguments differ from the run that wrote the checkpoint\n"
            + lines
        )


def step_of(path: Path) -> int:
    m = re.search(r"step_(\d+)\.ckpt$", str(path))
    return int(m.group(1)) if m else -1


def latest(ckpt_dir: str | Path):
    files = sorted(Path(ckpt_dir).glob("step_*.ckpt"), key=step_of)
    return files[-1] if files else None


def save(ckpt_dir: str | Path, step: int, state: dict, keep: int = 2) -> Path:
    d = Path(ckpt_dir)
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"step_{step}.ckpt"
    tmp = d / f".step_{step}.ckpt.tmp"
    torch.save(state, tmp)
    os.replace(
        tmp, path
    )  # atomic on one filesystem: a killed run leaves the previous checkpoint intact
    for old in sorted(d.glob("step_*.ckpt"), key=step_of)[:-keep] if keep > 0 else []:
        old.unlink()
    return path


def load(path: str | Path, ckpt_dir: str = ""):
    """Resolve ``--resume`` (``auto``, a file, a directory, or ``wandb:...``) to a loaded checkpoint dict, or None for ``auto`` with nothing to resume."""
    spec = str(path)
    if spec.startswith("wandb:"):
        import wandb

        art = wandb.Api().artifact(spec[len("wandb:") :])
        d = Path(art.download())
        f = latest(d)
        if f is None:
            raise SystemExit(f"no step_*.ckpt in artifact {spec}")
        return torch.load(f, map_location="cpu", weights_only=False)
    if spec == "auto":
        f = latest(ckpt_dir) if ckpt_dir else None
        return torch.load(f, map_location="cpu", weights_only=False) if f else None
    p = Path(spec)
    if p.is_dir():
        p = latest(p)
    if p is None or not p.exists():
        raise SystemExit(f"--resume: no checkpoint at {spec}")
    return torch.load(p, map_location="cpu", weights_only=False)


class Logger:
    """Weights & Biases when ``--wandb`` is given, otherwise a no-op with the same interface."""

    def __init__(self, project: str, name: str, config: dict, run_id: str | None):
        self.run = None
        if not project:
            return
        import wandb

        self._wandb = wandb
        self.run = wandb.init(
            project=project,
            name=name or None,
            config=config,
            id=run_id,
            resume="allow",
        )

    @property
    def id(self):
        return self.run.id if self.run is not None else None

    def log(self, step: int, **values):
        if self.run is not None:
            self.run.log(values, step=step)

    def artifact(self, path: Path, step: int):
        if self.run is not None:
            art = self._wandb.Artifact(f"ckpt-{self.run.id}", type="checkpoint")
            art.add_file(str(path))
            self.run.log_artifact(art, aliases=["latest", f"step_{step}"])

    def finish(self, **summary):
        if self.run is not None:
            for k, v in summary.items():
                self.run.summary[k] = v
            self.run.finish()
