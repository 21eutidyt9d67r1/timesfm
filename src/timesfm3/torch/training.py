"""Supervised horizon loss and PEFT LoRA for the native TimesFM-3 model.

No checkpoint download or PEFT import occurs when this module is imported.
This is fine-tuning support, not Google's original pretraining recipe.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from .model import TimesFM3Torch


def add_lora(
  model: TimesFM3Torch,
  rank: int = 8,
  alpha: float = 16,
  dropout: float = 0.0,
  target_modules=None,
):
  """Freeze the base and inject PEFT adapters into Q/V attention projections.

  Includes temporal and variate attention. PEFT owns serialization and reload;
  reload with PeftModel.from_pretrained(fresh_base, path, is_trainable=True).
  Materialize/load the base before adapter injection and optimizer creation.
  """
  if type(rank) is not int or rank < 1 or not math.isfinite(alpha) or alpha <= 0:
    raise ValueError("positive LoRA rank and alpha required")
  if not 0 <= dropout < 1:
    raise ValueError("LoRA dropout must be in [0, 1)")
  if not model.pre_transformer_resblock._input_dim_set:
    raise ValueError("initialize the base model before adding LoRA")
  from peft import LoraConfig, get_peft_model

  return get_peft_model(
    model,
    LoraConfig(
      r=rank,
      lora_alpha=alpha,
      lora_dropout=dropout,
      bias="none",
      target_modules=["query_proj", "value_proj"]
      if target_modules is None
      else target_modules,
    ),
  )


def _checked_tensor(values, mask, name):
  if values.ndim != 3 or not values.is_floating_point() or 0 in values.shape:
    raise ValueError(
      f"{name} must be a nonempty floating (batch, variates, time) tensor"
    )
  if mask is None:
    mask = torch.zeros_like(values, dtype=torch.bool)
  if (
    mask.shape != values.shape
    or mask.dtype != torch.bool
    or mask.device != values.device
  ):
    raise ValueError(f"{name} mask must be bool with the same shape and device")
  if not torch.isfinite(values[~mask]).all():
    raise ValueError(f"nonfinite unmasked {name}; use explicit missing masks")
  if not (~mask).any(dim=-1).all():
    raise ValueError(f"each {name} series needs at least one valid observation")
  return torch.where(mask, 0.0, values), mask


class TimesFM3ForTraining(nn.Module):
  """A horizon-aligned pinball loss over original-unit target quantiles.

  Future values are validated separately and used ONLY in the loss, never in
  model inputs or context normalization statistics.
  Optional masks use True=unavailable. Known-future covariates deliberately
  aren't supported here; the first training entry is historical-covariate-only.
  """

  def __init__(self, backbone, normalize_loss: bool = True):
    super().__init__()
    self.backbone = backbone
    self.normalize_loss = normalize_loss

  def forward(
    self,
    context,
    future_values=None,
    *,
    horizon=None,
    past_only_covariates=None,
    context_mask=None,
    past_only_mask=None,
    label_mask=None,
    return_aux_outputs=False,
  ):
    context, context_mask = _checked_tensor(context, context_mask, "context")
    if not 1 <= context.shape[-1] <= 15360:
      raise ValueError("context length must be 1..15360; no silent truncation")
    labels = None
    if future_values is not None:
      labels, label_mask = _checked_tensor(future_values, label_mask, "label")
      if labels.shape[:2] != context.shape[:2] or labels.device != context.device:
        raise ValueError("label batch, target variates and device must match context")
      if horizon is None:
        horizon = labels.shape[-1]
      elif horizon != labels.shape[-1]:
        raise ValueError("horizon must match label length")
    elif label_mask is not None:
      raise ValueError("label_mask requires labels")
    if type(horizon) is not int or horizon < 1:
      raise ValueError("positive integer horizon required")
    if past_only_covariates is not None:
      past_only_covariates, past_only_mask = _checked_tensor(
        past_only_covariates, past_only_mask, "past covariates"
      )
      if (
        past_only_covariates.shape[0] != context.shape[0]
        or past_only_covariates.shape[-1] != context.shape[-1]
        or past_only_covariates.device != context.device
      ):
        raise ValueError("past covariates must match context batch, time and device")
    elif past_only_mask is not None:
      raise ValueError("past_only_mask requires covariates")
    base = (
      self.backbone.get_base_model()
      if hasattr(self.backbone, "get_base_model")
      else self.backbone
    )
    variates = context.shape[1] + (
      0 if past_only_covariates is None else past_only_covariates.shape[1]
    )
    if variates > base.transformer_config.transformer.max_variates:
      raise ValueError("too many variates; no implicit feature subsampling")
    # Match the public forecaster's output-patch rounding before decode, then
    # trim only the returned forecast. Particularly important across patches.
    rounded_horizon = (
      (horizon + base.output_patch_len - 1) // base.output_patch_len
    ) * base.output_patch_len
    raw = base.forecast(
      context,
      rounded_horizon,
      past_only_covariates=past_only_covariates,
      target_mask=context_mask,
      past_only_mask=past_only_mask,
      return_aux_outputs=return_aux_outputs,
    )
    quantiles = (raw[0] if return_aux_outputs else raw)[:, : context.shape[1], :horizon]
    if not torch.isfinite(quantiles).all():
      raise ValueError("nonfinite model forecast")
    result = {"forecast": quantiles}
    if return_aux_outputs:
      result["aux"] = raw[1]
    if labels is not None:
      # Labels masked out are replaced BEFORE arithmetic; NaN * 0 is not zero.
      error = labels.unsqueeze(-1) - quantiles
      if self.normalize_loss:
        observed = (~context_mask).to(context.dtype)
        n = observed.sum(-1, keepdim=True)
        mean = context.sum(-1, keepdim=True) / n
        variance = (((context - mean) ** 2) * observed).sum(-1, keepdim=True) / n
        scale = variance.detach().sqrt()
        scale = torch.where(scale < 1e-6, 1.0, scale)
        error = error / scale.unsqueeze(-1)
      q = torch.as_tensor(base.quantiles, device=error.device, dtype=error.dtype)
      losses = torch.maximum(q * error, (q - 1) * error)
      result["loss"] = losses[~label_mask].mean()
      result["valid_labels"] = int((~label_mask).sum().item())
    return result
