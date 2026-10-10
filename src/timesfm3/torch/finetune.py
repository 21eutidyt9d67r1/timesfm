"""Minimal local-checkpoint LoRA fine-tuning CLI for TimesFM-3.

Input: separate chronological train/validation directories of memory-mapped
NPY time series, not pre-materialized overlapping windows. No network download,
data interpolation, test-set tuning, trading, or RL is performed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .model import TimesFM3Torch
from .training import TimesFM3ForTraining, add_lora


class WindowDataset(Dataset):
  """Uniform chronological series; all windows stay inside one data split.

  target.npy and optional past_covariates.npy are (time, variates).
  timestamps.npy is a strictly increasing uniform int64 or datetime64 clock.
  Optional masks (target_mask.npy, past_mask.npy) use True=unavailable.
  Masked labels never enter the loss, and there is no automatic gap filling.
  """

  def __init__(self, directory, context, horizon):
    if (
      type(context) is not int
      or not 1 <= context <= 15360
      or type(horizon) is not int
      or horizon < 1
    ):
      raise ValueError("positive context/horizon required")
    self.directory = Path(directory)
    self.context, self.horizon = context, horizon
    self.arrays = {}
    for name in ["target", "timestamps", "past_covariates", "target_mask", "past_mask"]:
      path = self.directory / (name + ".npy")
      if path.exists():
        self.arrays[name] = np.load(path, mmap_mode="r", allow_pickle=False)
    if not {"target", "timestamps"} <= self.arrays.keys():
      raise ValueError("target.npy and timestamps.npy required")
    target, times = self.arrays["target"], self.arrays["timestamps"]
    if target.ndim != 2 or target.dtype.kind != "f" or target.shape[1] < 1:
      raise ValueError("target must be floating (time, variates)")
    if len(target) < context + horizon:
      raise ValueError("data too short for context+horizon")
    if (
      times.shape != (len(target),)
      or times.dtype.kind not in "iM"
      or times.dtype.itemsize != 8
    ):
      raise ValueError("timestamps must be int64 or datetime64 with one entry per row")
    if times.dtype.kind == "M" and np.isnat(times).any():
      raise ValueError("timestamps cannot contain NaT")
    deltas = np.diff(times.astype("int64"))
    if not (deltas > 0).all() or not (deltas == deltas[0]).all():
      raise ValueError(
        "timestamps must be contiguous uniform increasing; no gap compression"
      )
    for name in ["past_covariates", "target_mask", "past_mask"]:
      arr = self.arrays.get(name)
      if arr is not None and (arr.ndim != 2 or arr.shape[0] != len(target)):
        raise ValueError(f"{name} must align with target rows")
    cov = self.arrays.get("past_covariates")
    if cov is not None and (cov.dtype.kind != "f" or cov.shape[1] < 1):
      raise ValueError("past_covariates must be floating")
    for mask_name, values in [("target_mask", target), ("past_mask", cov)]:
      mask = self.arrays.get(mask_name)
      if mask is not None and (
        values is None or mask.shape != values.shape or mask.dtype != np.bool_
      ):
        raise ValueError(f"{mask_name} must be bool with matching values shape")

  def __len__(self):
    return len(self.arrays["target"]) - self.context - self.horizon + 1

  def __getitem__(self, i):
    if not 0 <= i < len(self):
      raise IndexError(i)
    c, h = i + self.context, i + self.context + self.horizon

    def tensor(name, start, stop):
      return torch.from_numpy(np.array(self.arrays[name][start:stop].T, copy=True))

    result = {
      "context": tensor("target", i, c).float(),
      "future_values": tensor("target", c, h).float(),
    }
    if "past_covariates" in self.arrays:
      result["past_only_covariates"] = tensor("past_covariates", i, c).float()
    if "target_mask" in self.arrays:
      result["context_mask"] = tensor("target_mask", i, c)
      result["label_mask"] = tensor("target_mask", c, h)
    if "past_mask" in self.arrays:
      result["past_only_mask"] = tensor("past_mask", i, c)
    return result


def _sha(path):
  h = hashlib.sha256()
  with Path(path).open("rb") as f:
    for chunk in iter(lambda: f.read(1024**2), b""):
      h.update(chunk)
  return h.hexdigest()


def run(args):
  train = WindowDataset(args.train_data, args.context, args.horizon)
  valid = WindowDataset(args.valid_data, args.context, args.horizon)
  tt, vt = train.arrays["timestamps"], valid.arrays["timestamps"]
  if tt.dtype != vt.dtype or not tt[-1] < vt[0]:
    raise ValueError(
      "train/validation timestamps overlap or use different clock dtypes"
    )
  if tt[1] - tt[0] != vt[1] - vt[0]:
    raise ValueError("train/validation sampling frequencies differ")
  for name in ["target", "past_covariates"]:
    a, b = train.arrays.get(name), valid.arrays.get(name)
    if (a is None) != (b is None) or a is not None and a.shape[1] != b.shape[1]:
      raise ValueError("train/validation feature schemas differ")
  if (
    args.epochs < 1 or args.batch_size < 1 or not math.isfinite(args.lr) or args.lr <= 0
  ):
    raise ValueError("positive epochs, batch_size and learning rate required")
  checkpoint = Path(args.checkpoint).resolve()
  if (
    not (checkpoint / "model.safetensors").is_file()
    or not (checkpoint / "config.json").is_file()
  ):
    raise ValueError("explicit local base checkpoint directory required; no download")
  # One input identity record also prevents resuming an optimizer against a
  # different base/data/settings. Not a registry or publishing service.
  identity = {
    "base": {n: _sha(checkpoint / n) for n in ["model.safetensors", "config.json"]},
    "data": {
      split: {p.name: _sha(p) for p in sorted(ds.directory.glob("*.npy"))}
      for split, ds in [("train", train), ("valid", valid)]
    },
    "settings": {
      k: getattr(args, k)
      for k in [
        "context",
        "horizon",
        "batch_size",
        "lr",
        "rank",
        "alpha",
        "dropout",
        "seed",
        "device",
      ]
    },
  }
  resumed = None
  if args.resume:
    resumed = Path(args.resume)
    if json.loads((resumed / "identity.json").read_text()) != identity:
      raise ValueError("resume base, data or settings mismatch")
  device = torch.device(args.device)
  if device.type == "cuda" and not torch.cuda.is_available():
    raise RuntimeError("requested CUDA unavailable; no CPU fallback")
  output = Path(args.output)
  output.mkdir(parents=True, exist_ok=False)
  receipt = {
    "status": "started",
    "mode": "supervised_lora",
    "device": str(device),
    "train_windows": len(train),
    "validation_windows": len(valid),
    "optimizer_steps": 0,
    "epochs": [],
  }
  try:
    torch.manual_seed(args.seed)
    base = TimesFM3Torch.from_pretrained(str(checkpoint), local_files_only=True).to(
      device
    )
    if resumed:
      from peft import PeftModel

      adapted = PeftModel.from_pretrained(base, resumed / "adapter", is_trainable=True)
    else:
      adapted = add_lora(base, rank=args.rank, alpha=args.alpha, dropout=args.dropout)
    model = TimesFM3ForTraining(adapted).to(device)
    parameters = [p for p in model.parameters() if p.requires_grad]
    receipt["trainable_parameters"] = sum(p.numel() for p in parameters)
    optimizer = torch.optim.AdamW(parameters, lr=args.lr)
    generator = torch.Generator().manual_seed(args.seed)
    first_epoch = 0
    if resumed:
      state = torch.load(resumed / "trainer.pt", weights_only=True, map_location="cpu")
      optimizer.load_state_dict(state["optimizer"])
      generator.set_state(state["sampler_rng"])
      torch.set_rng_state(state["torch_rng"])
      if device.type == "cuda":
        torch.cuda.set_rng_state_all(state["cuda_rng"])
      first_epoch = state["epoch"]
      receipt["optimizer_steps"] = state["optimizer_steps"]
    if args.epochs <= first_epoch:
      raise ValueError("epochs must exceed the completed resume epoch")
    loaders = [
      DataLoader(train, batch_size=args.batch_size, shuffle=True, generator=generator),
      DataLoader(valid, batch_size=args.batch_size, shuffle=False),
    ]
    best_record = state.get("best") if resumed else None
    best = best_record["validation_loss"] if best_record else float("inf")
    if best_record:
      (output / "best.json").write_text(json.dumps(best_record, indent=2))
    for epoch in range(first_epoch, args.epochs):
      metrics = {}
      for phase, loader in zip(["train", "validation"], loaders):
        model.train(phase == "train")
        total, count = 0.0, 0
        with torch.set_grad_enabled(phase == "train"):
          for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            if phase == "train":
              optimizer.zero_grad(set_to_none=True)
            result = model(**batch)
            loss = result["loss"]
            if not torch.isfinite(loss):
              raise FloatingPointError("nonfinite training/validation loss")
            if phase == "train":
              loss.backward()
              if any(
                p.grad is None or not torch.isfinite(p.grad).all() for p in parameters
              ):
                raise FloatingPointError("missing/nonfinite LoRA gradient")
              torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
              optimizer.step()
              receipt["optimizer_steps"] += 1
            total += loss.item() * result["valid_labels"]
            count += result["valid_labels"]
        metrics[phase + "_loss"] = total / count
      saved = output / f"epoch-{epoch + 1:04d}"
      saved.mkdir()
      adapted.save_pretrained(saved / "adapter", safe_serialization=True)
      if metrics["validation_loss"] < best:
        best = metrics["validation_loss"]
        best_record = {
          "epoch": epoch + 1,
          "adapter": str((saved / "adapter").resolve()),
          "validation_loss": best,
        }
        (output / "best.json").write_text(json.dumps(best_record, indent=2))
      (saved / "identity.json").write_text(json.dumps(identity, indent=2))
      torch.save(
        {
          "epoch": epoch + 1,
          "optimizer": optimizer.state_dict(),
          "optimizer_steps": receipt["optimizer_steps"],
          "best": best_record,
          "sampler_rng": generator.get_state(),
          "torch_rng": torch.get_rng_state(),
          "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
        },
        saved / "trainer.pt",
      )
      receipt["epochs"].append({"epoch": epoch + 1, **metrics})
      print(json.dumps(receipt["epochs"][-1]), flush=True)
    receipt["status"] = "completed"
  except BaseException as exc:
    receipt.update(status="failed", error_type=type(exc).__name__, error=str(exc))
    raise
  finally:
    (output / "result.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description=__doc__)
  for name in ["checkpoint", "train-data", "valid-data", "output"]:
    parser.add_argument("--" + name, required=True)
  parser.add_argument("--context", type=int, default=512)
  parser.add_argument("--horizon", type=int, default=64)
  parser.add_argument("--epochs", type=int, default=1)
  parser.add_argument("--batch-size", type=int, default=4)
  parser.add_argument("--lr", type=float, default=1e-4)
  parser.add_argument("--rank", type=int, default=8)
  parser.add_argument("--alpha", type=float, default=16)
  parser.add_argument("--dropout", type=float, default=0.0)
  parser.add_argument("--seed", type=int, default=42)
  parser.add_argument("--device", default="cuda:0")
  parser.add_argument("--resume", help="trusted local epoch checkpoint directory")
  run(parser.parse_args())
