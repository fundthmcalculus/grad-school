"""Tests of checkpoint / resume, the precision flags and the Weights & Biases hook of `flm.train` (`PLAN_PRECISION.md`), on a tiny CPU model: ``pytest experiments/fuzzy-language-model/test_resume.py``.

They need the TinyStories character stream (``data/tinystories/train.bin``); without it they are skipped.  W&B is exercised through a stand-in module (no network, no account).
"""

import json
import sys
import types
from pathlib import Path

import pytest

import flm.ckpt as CK
import flm.train as T
from flm.data import data_dir

pytestmark = pytest.mark.skipif(
    not (data_dir() / "train.bin").exists(), reason="TinyStories stream not present"
)

BASE = [
    "--d", "16", "--layers", "1", "--heads", "2", "--mixer", "softmax", "--ffn", "mlp",
    "--ctx", "64", "--batch", "8", "--chars", "0.3e6", "--lr", "3e-3", "--warmup", "5",
    "--eval-every", "10", "--eval-chars", "20000", "--seed", "3", "--device", "cpu",
]  # fmt: skip


def run(tmp, name, *extra):
    out = tmp / f"{name}.json"
    T.main(BASE + ["--out", str(out)] + list(extra))
    return json.loads(out.read_text())


def test_resume_reproduces_an_uninterrupted_run(tmp_path):
    ref = run(tmp_path, "ref")
    ck = tmp_path / "ck"
    first = run(tmp_path, "first", "--ckpt-dir", str(ck), "--ckpt-every", "20")
    assert (
        first["evals"] == ref["evals"]
    )  # writing checkpoints does not perturb the run
    saved = sorted(p.name for p in ck.glob("step_*.ckpt"))
    assert saved and not list(ck.glob(".*.tmp"))
    # a restart picks up the newest checkpoint and ends where the uninterrupted run ended
    second = run(
        tmp_path,
        "second",
        "--ckpt-dir",
        str(ck),
        "--ckpt-every",
        "20",
        "--resume",
        "auto",
    )
    assert second["evals"] == ref["evals"]
    assert second["val_bpc"] == ref["val_bpc"]
    assert second["train_curve"] == ref["train_curve"]


def test_resume_auto_with_nothing_saved_starts_fresh(tmp_path):
    ref = run(tmp_path, "ref")
    fresh = run(
        tmp_path, "fresh", "--ckpt-dir", str(tmp_path / "empty"), "--resume", "auto"
    )
    assert fresh["val_bpc"] == ref["val_bpc"]


def test_resume_refuses_a_different_experiment(tmp_path):
    ck = tmp_path / "ck"
    run(tmp_path, "a", "--ckpt-dir", str(ck), "--ckpt-every", "20")
    with pytest.raises(SystemExit) as e:
        run(tmp_path, "b", "--ckpt-dir", str(ck), "--resume", "auto", "--lr", "1e-3")
    assert "lr" in str(e.value)


def test_checkpoint_pruning_and_atomic_write(tmp_path):
    for s in (10, 20, 30, 40):
        CK.save(tmp_path, s, {"step": s}, keep=2)
    assert sorted(CK.step_of(p) for p in tmp_path.glob("step_*.ckpt")) == [30, 40]
    assert CK.latest(tmp_path).name == "step_40.ckpt"
    assert not list(tmp_path.glob(".*.tmp"))


def test_bf16_autocast_runs_and_is_rescored_in_fp32(tmp_path):
    r = run(tmp_path, "bf16", "--precision", "bf16")
    assert r["val_bpc_fp32"] is not None and r["val_bpc"] == r["val_bpc"]
    assert (
        abs(r["val_bpc"] - r["val_bpc_fp32"]) < 0.05
    )  # the two scorings agree on this model
    assert run(tmp_path, "fp32")["val_bpc_fp32"] is None


def test_wandb_hook_logs_resumes_and_finishes(tmp_path, monkeypatch):
    calls = []

    class Run:
        id = "abc123"
        summary = {}

        def log(self, values, step=None):
            calls.append(("log", step, sorted(values)))

        def log_artifact(self, art, aliases=None):
            calls.append(("artifact", aliases))

        def finish(self):
            calls.append(("finish",))

    fake = types.ModuleType("wandb")
    fake.init = (
        lambda **kw: calls.append(("init", kw["project"], kw["id"], kw["resume"]))
        or Run()
    )
    fake.Artifact = lambda name, type=None: types.SimpleNamespace(
        add_file=lambda p: None
    )
    monkeypatch.setitem(sys.modules, "wandb", fake)
    ck = tmp_path / "ck"
    run(
        tmp_path, "w1", "--wandb", "proj", "--ckpt-dir", str(ck), "--ckpt-every", "20",
        "--wandb-artifacts",
    )  # fmt: skip
    kinds = [c[0] for c in calls]
    assert kinds[0] == "init" and calls[0][1:] == ("proj", None, "allow")
    assert "artifact" in kinds and kinds[-1] == "finish"
    assert any(c[0] == "log" and "val_bpc" in c[2] for c in calls)
    # the id travels in the checkpoint, so a resumed run continues the same W&B run
    saved = CK.load(CK.latest(ck), "")
    assert saved["wandb_id"] == "abc123"
    calls.clear()
    run(tmp_path, "w2", "--wandb", "proj", "--ckpt-dir", str(ck), "--resume", "auto")
    assert calls[0] == ("init", "proj", "abc123", "allow")


def test_real_wandb_offline_logs_and_resumes_the_same_run(tmp_path, monkeypatch):
    wandb = pytest.importorskip("wandb")  # the real package, no network: offline mode
    monkeypatch.setenv("WANDB_MODE", "offline")
    monkeypatch.setenv("WANDB_DIR", str(tmp_path))
    monkeypatch.setenv("WANDB_SILENT", "true")
    ck = tmp_path / "ck"
    run(
        tmp_path,
        "r1",
        "--wandb",
        "flm-test",
        "--ckpt-dir",
        str(ck),
        "--ckpt-every",
        "20",
    )
    first_id = CK.load(CK.latest(ck), "")["wandb_id"]
    assert first_id and list((tmp_path / "wandb").glob(f"offline-run-*-{first_id}"))
    run(
        tmp_path,
        "r2",
        "--wandb",
        "flm-test",
        "--ckpt-dir",
        str(ck),
        "--ckpt-every",
        "20",
        "--resume",
        "auto",
    )
    runs = list((tmp_path / "wandb").glob("offline-run-*"))
    assert len(runs) == 2 and all(
        first_id in p.name for p in runs
    )  # the resume reused the id
