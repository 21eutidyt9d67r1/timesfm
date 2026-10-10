"""Training/LoRA tests use tiny random networks; no pretrained downloads."""

import copy

import pytest
import torch

pytest.importorskip("peft", reason="install the train extra for LoRA tests")

from . import configs, util
from .model import TimesFM3Torch
from .training import TimesFM3ForTraining, add_lora


def tiny_model(**overrides):
  torch.manual_seed(7)
  kwargs = dict(
    input_patch_len=8,
    output_patch_len=16,
    quantiles=[0.1, 0.5, 0.9],
    residual_block_config=configs.ResidualBlockConfig(
      hidden_dims=16, output_dims=16, use_bias=False, activation="relu"
    ),
    transformer_config=configs.StackedTransformersConfig(
      num_layers=2,
      use_remat=False,
      transformer=configs.TransformerConfig(
        model_dims=16,
        hidden_dims=24,
        num_heads=2,
        use_sdpa=True,
        use_rope_seq=True,
        use_rope_var=True,
        attention_norm="rms",
        feedforward_norm="rms",
        qk_norm="rms",
        use_bias=False,
        ff_activation="relu",
        deterministic=True,
      ),
    ),
  )
  kwargs.update(overrides)
  return TimesFM3Torch(**kwargs)


def sample():
  torch.manual_seed(11)
  return torch.randn(2, 1, 32) + 10, torch.randn(2, 1, 32)


@pytest.mark.parametrize("stitching", [False, True])
@pytest.mark.parametrize("detrending", [False, True])
@pytest.mark.parametrize("frozen", [False, True])
def test_train_forecast_matches_inference_and_backward(stitching, detrending, frozen):
  m = tiny_model(
    use_stitching=stitching,
    use_linear_detrending=detrending,
    use_frozen_running_stats=frozen,
  )
  context, cov = sample()
  reference = m.decode(context, 19, past_only_covariates=cov)
  result = m.forecast(context, 19, past_only_covariates=cov)
  torch.testing.assert_close(result, reference, rtol=0, atol=0)
  assert result.requires_grad and not reference.requires_grad
  result.square().mean().backward()
  assert all(
    p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters()
  )


def test_labels_change_loss_not_forecast_or_context():
  model = TimesFM3ForTraining(tiny_model())
  context, cov = sample()
  labels = torch.ones(2, 1, 16) * 11
  before = context.clone()
  a = model(context, labels, past_only_covariates=cov)
  b = model(context, labels * 100, past_only_covariates=cov)
  torch.testing.assert_close(a["forecast"], b["forecast"], rtol=0, atol=0)
  assert a["loss"] != b["loss"]
  torch.testing.assert_close(context, before, rtol=0, atol=0)
  assert a["forecast"].shape == (2, 1, 16, 3)


def test_masked_labels_do_not_affect_loss_or_gradient():
  m = TimesFM3ForTraining(tiny_model())
  context, _ = sample()
  labels = torch.ones(2, 1, 16) * 11
  mask = torch.zeros_like(labels, dtype=torch.bool)
  mask[..., -4:] = True
  labels[mask] = float("nan")
  out = m(context, labels, label_mask=mask)
  assert torch.isfinite(out["loss"])
  labels[mask] = 1e8
  torch.testing.assert_close(out["loss"], m(context, labels, label_mask=mask)["loss"])
  out["loss"].backward()
  assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
  with pytest.raises(ValueError, match="label"):
    m(context, labels, label_mask=torch.ones_like(mask))


@pytest.mark.parametrize(
  "case", ["nan_context", "inf_label", "all_masked", "shape", "horizon", "mask_type"]
)
def test_reject_invalid_training_inputs(case):
  m = TimesFM3ForTraining(tiny_model())
  context, _ = sample()
  labels = torch.ones(2, 1, 16)
  kwargs = {}
  if case == "nan_context":
    context[..., 3] = float("nan")
  if case == "inf_label":
    labels[..., 3] = float("inf")
  if case == "all_masked":
    kwargs["context_mask"] = torch.ones_like(context, dtype=torch.bool)
  if case == "shape":
    labels = labels[:1]
  if case == "horizon":
    kwargs["horizon"] = 17
  if case == "mask_type":
    kwargs["label_mask"] = torch.zeros_like(labels)
  with pytest.raises(ValueError):
    m(context, labels, **kwargs)


def test_zero_lora_parity_and_only_adapters_update():
  base = tiny_model()
  context, cov = sample()
  reference = base.decode(context, 16, past_only_covariates=cov)
  adapted = add_lora(base, rank=2, alpha=4)
  actual = adapted.get_base_model().decode(context, 16, past_only_covariates=cov)
  torch.testing.assert_close(reference, actual, rtol=0, atol=0)
  trainer = TimesFM3ForTraining(adapted)
  before = {n: p.detach().clone() for n, p in trainer.named_parameters()}
  trainable = {n for n, p in trainer.named_parameters() if p.requires_grad}
  assert trainable and all("lora_" in n for n in trainable)
  optimizer = torch.optim.AdamW(
    [p for p in trainer.parameters() if p.requires_grad], lr=1e-3
  )
  for _ in range(2):
    optimizer.zero_grad()
    out = trainer(context, torch.full((2, 1, 16), 11.0), past_only_covariates=cov)
    out["loss"].backward()
    assert all(
      p.grad is not None and torch.isfinite(p.grad).all()
      for p in trainer.parameters()
      if p.requires_grad
    )
    optimizer.step()
  assert any(
    not torch.equal(before[n], p)
    for n, p in trainer.named_parameters()
    if n in trainable
  )
  assert all(
    torch.equal(before[n], p)
    for n, p in trainer.named_parameters()
    if n not in trainable
  )


def test_lora_save_reload_and_gradients(tmp_path):
  from peft import PeftModel

  base = tiny_model()
  fresh = copy.deepcopy(base)
  lora = add_lora(base, rank=2, alpha=4)
  trainer = TimesFM3ForTraining(lora)
  context, _ = sample()
  loss = trainer(context, torch.ones(2, 1, 16))["loss"]
  loss.backward()
  with torch.no_grad():
    for p in trainer.parameters():
      if p.requires_grad:
        p.add_(p.grad, alpha=-1e-3)
  lora.save_pretrained(tmp_path, safe_serialization=True)
  loaded = PeftModel.from_pretrained(fresh, tmp_path, is_trainable=True)
  expected = trainer(context, horizon=16)["forecast"]
  actual = TimesFM3ForTraining(loaded)(context, horizon=16)["forecast"]
  torch.testing.assert_close(expected, actual, rtol=0, atol=0)
  assert list(tmp_path.glob("*.safetensors"))


def test_running_stats_zero_variance_has_finite_backward():
  x = torch.zeros(1, 1, 8, requires_grad=True)
  z = torch.zeros(1, 1)
  for mask in (
    torch.ones_like(x, dtype=torch.bool),
    torch.zeros_like(x, dtype=torch.bool),
  ):
    _, mu, sigma = util.update_running_stats(z, z, z, x, mask)
    (grad,) = torch.autograd.grad((mu + sigma).sum(), x, retain_graph=True)
    assert torch.isfinite(grad).all()


def test_safe_sqrt_does_not_hide_invalid_values():
  values = torch.tensor([0.0, 1.0, -1.0, float("nan"), float("inf")])
  torch.testing.assert_close(
    util._sqrt_with_zero_grad(values), values.sqrt(), equal_nan=True
  )


def test_tiny_training_loss_can_decrease():
  trainer = TimesFM3ForTraining(add_lora(tiny_model(), rank=4, alpha=8))
  context, cov = sample()
  labels = torch.full((2, 1, 16), 10.5)
  opt = torch.optim.AdamW([p for p in trainer.parameters() if p.requires_grad], lr=0.01)
  first = trainer(context, labels, past_only_covariates=cov)["loss"].item()
  for _ in range(30):
    opt.zero_grad()
    loss = trainer(context, labels, past_only_covariates=cov)["loss"]
    loss.backward()
    torch.nn.utils.clip_grad_norm_(trainer.parameters(), 1.0)
    opt.step()
  assert trainer(context, labels, past_only_covariates=cov)["loss"].item() < first * 0.7


@pytest.mark.parametrize("constant", [False, True])
def test_masked_context_and_long_horizon_backward(constant):
  m = TimesFM3ForTraining(add_lora(tiny_model(), rank=2, alpha=4))
  context, cov = sample()
  if constant:
    context[:] = 10.0
    cov[:] = 0.0
  mask = torch.zeros_like(context, dtype=torch.bool)
  mask[..., :10] = True
  context[mask] = float("nan")
  loss = m(
    context, torch.ones(2, 1, 33) * 10.2, context_mask=mask, past_only_covariates=cov
  )["loss"]
  loss.backward()
  assert all(
    p.grad is not None and torch.isfinite(p.grad).all()
    for p in m.parameters()
    if p.requires_grad
  )


def test_full_parameter_training_changes_base():
  trainer = TimesFM3ForTraining(tiny_model())
  context, _ = sample()
  opt = torch.optim.SGD(trainer.parameters(), lr=1e-4)
  before = trainer.backbone.output_head.weight.detach().clone()
  trainer(context, torch.ones(2, 1, 16))["loss"].backward()
  opt.step()
  assert not torch.equal(before, trainer.backbone.output_head.weight)


def test_loss_respects_declared_quantiles():
  m = TimesFM3ForTraining(tiny_model(), normalize_loss=False)
  context, _ = sample()
  labels = torch.ones(2, 1, 16) * 11
  out = m(context, labels)
  error = labels.unsqueeze(-1) - out["forecast"]
  q = torch.tensor([0.1, 0.5, 0.9])
  torch.testing.assert_close(
    out["loss"], torch.maximum(q * error, (q - 1) * error).mean()
  )


@pytest.mark.parametrize("horizon", [1, 8, 16, 17, 33])
def test_training_forecast_matches_native_public_horizon_rounding(horizon):
  base = tiny_model()
  context, cov = sample()
  expected = base.decode(
    context, ((horizon + 15) // 16) * 16, past_only_covariates=cov
  )[:, :1, :horizon]
  actual = TimesFM3ForTraining(base)(
    context, horizon=horizon, past_only_covariates=cov
  )["forecast"]
  torch.testing.assert_close(expected, actual, rtol=0, atol=0)


def test_lora_gradient_matches_numerical_difference():
  trainer = TimesFM3ForTraining(add_lora(tiny_model(), rank=2, alpha=4))
  context, cov = sample()
  labels = torch.full((2, 1, 16), 11.0)
  trainer(context, labels, past_only_covariates=cov)["loss"].backward()
  parameter = next(p for n, p in trainer.named_parameters() if "lora_B" in n)
  index = torch.unravel_index(parameter.grad.abs().argmax(), parameter.shape)
  analytic = parameter.grad[index].item()
  epsilon = 1e-3
  assert abs(analytic) > 1e-5
  with torch.no_grad():
    original = parameter[index].item()
    parameter[index] = original + epsilon
    plus = trainer(context, labels, past_only_covariates=cov)["loss"].item()
    parameter[index] = original - epsilon
    minus = trainer(context, labels, past_only_covariates=cov)["loss"].item()
    parameter[index] = original
  assert (plus - minus) / (2 * epsilon) == pytest.approx(analytic, rel=0.03, abs=1e-4)
