"""CPU-only tests for the bounded example using a tiny locally saved model."""

import argparse
import json

import numpy as np
import pytest
import torch

from .finetune import WindowDataset, run
from .training_test import tiny_model


def data_dir(path, offset=0):
  path.mkdir()
  t = np.arange(56, dtype=np.float32)
  np.save(path / "target.npy", (10 + np.sin(t / 4))[:, None])
  np.save(path / "past_covariates.npy", np.cos(t / 4)[:, None])
  np.save(path / "timestamps.npy", (np.arange(56) + offset).astype("int64"))
  return path


def test_window_samples_use_history_and_separate_future_labels(tmp_path):
  ds = WindowDataset(data_dir(tmp_path / "data"), context=16, horizon=8)
  sample = ds[3]
  raw = np.load(tmp_path / "data/target.npy")
  np.testing.assert_array_equal(sample["context"].numpy(), raw[3:19].T)
  np.testing.assert_array_equal(sample["future_values"].numpy(), raw[19:27].T)
  assert sample["past_only_covariates"].shape == (1, 16)
  assert len(ds) == 33


def test_incomplete_clock_or_short_data_not_silently_skipped(tmp_path):
  p = data_dir(tmp_path / "data")
  with pytest.raises(ValueError):
    WindowDataset(p, context=64, horizon=8)
  timestamps = np.load(p / "timestamps.npy")
  timestamps[4] = timestamps[3]
  np.save(p / "timestamps.npy", timestamps)
  with pytest.raises(ValueError, match="timestamps"):
    WindowDataset(p, 16, 8)


def args(tmp_path, output, epochs=1, resume=None):
  return argparse.Namespace(
    checkpoint=str(tmp_path / "base"),
    train_data=str(tmp_path / "train"),
    valid_data=str(tmp_path / "valid"),
    output=str(tmp_path / output),
    epochs=epochs,
    context=16,
    horizon=8,
    batch_size=8,
    lr=0.002,
    rank=2,
    alpha=4.0,
    dropout=0.0,
    seed=42,
    device="cpu",
    resume=resume,
  )


def test_train_save_resume_equals_uninterrupted(tmp_path):
  tiny_model().save_pretrained(tmp_path / "base")
  data_dir(tmp_path / "train")
  data_dir(tmp_path / "valid", 100)
  a = args(tmp_path, "one")
  run(a)
  saved = tmp_path / "one/epoch-0001"
  assert (saved / "adapter/adapter_model.safetensors").exists()
  assert (saved / "trainer.pt").exists()
  first = json.loads((tmp_path / "one/result.json").read_text())
  assert first["status"] == "completed" and first["optimizer_steps"] > 0
  run(args(tmp_path, "resumed", 2, str(saved)))
  run(args(tmp_path, "continuous", 2))
  from safetensors.torch import load_file

  resumed = load_file(tmp_path / "resumed/epoch-0002/adapter/adapter_model.safetensors")
  continuous = load_file(
    tmp_path / "continuous/epoch-0002/adapter/adapter_model.safetensors"
  )
  assert resumed.keys() == continuous.keys()
  for key in resumed:
    torch.testing.assert_close(resumed[key], continuous[key], rtol=0, atol=0)


def test_reject_overlapping_train_validation_before_model_load(tmp_path):
  data_dir(tmp_path / "train")
  data_dir(tmp_path / "valid")
  with pytest.raises(ValueError, match="overlap"):
    run(args(tmp_path, "out"))
  assert not (tmp_path / "out").exists()


def test_resume_rejects_changed_data_or_base(tmp_path):
  tiny_model().save_pretrained(tmp_path / "base")
  data_dir(tmp_path / "train")
  data_dir(tmp_path / "valid", 100)
  run(args(tmp_path, "first"))
  values = np.load(tmp_path / "train/target.npy")
  values[0] += 10
  np.save(tmp_path / "train/target.npy", values)
  with pytest.raises(ValueError, match="resume"):
    run(args(tmp_path, "changed", 2, str(tmp_path / "first/epoch-0001")))


def test_training_error_leaves_failure_receipt(tmp_path):
  tiny_model().save_pretrained(tmp_path / "base")
  p = data_dir(tmp_path / "train")
  data_dir(tmp_path / "valid", 100)
  values = np.load(p / "target.npy")
  values[:] = np.nan
  np.save(p / "target.npy", values)
  with pytest.raises(ValueError, match="nonfinite"):
    run(args(tmp_path, "failed"))
  assert json.loads((tmp_path / "failed/result.json").read_text())["status"] == "failed"
