# TimesFM-3 native PyTorch training and LoRA (experimental fork)

This fork adds supervised training to the native TimesFM-3 model. It is **not**
the TimesFM-2.5 Transformers recipe or a reproduction of Google's pretraining.
The original architecture, tensor names and inference API are preserved.

- `model.decode(...)` remains inference-only.
- `model.forecast(...)` runs the same decode mathematics with autograd enabled.
- `TimesFM3ForTraining` masks future targets out of the model input and computes
  pinball loss over the target quantiles. Optional loss scaling uses only the
  historical context; returned forecasts remain in original units.
- `add_lora` uses PEFT, freezing the base and adapting `query_proj` and
  `value_proj` in both temporal and variate attention. Targets/rank are configurable.
  LoRA dropout defaults to zero; the checkpoint's model architecture is unchanged.
- PEFT owns adapter serialization/reloading. No custom LoRA layer is introduced.

The shared statistics utility now defines finite gradients at zero variance and
avoids division by zero for fully masked patches. Its forward values remain
unchanged on valid inputs. This is necessary for backpropagation through CPM
RevIN refinement; merely removing `no_grad` is insufficient.

## Install

In a dedicated environment with the correct PyTorch build for your device:

```sh
pip install -e '.[train]'
```

Verified development dependencies: Python 3.11, PyTorch 2.8.0, PEFT 0.17.1,
Transformers 4.55.4. The training extra pins the tested PEFT/Transformers minor
versions; it does not replace your CUDA build. Other versions require validation.

## Python API

```python
import torch
from timesfm3.torch.model import TimesFM3Torch
from timesfm3.torch.training import TimesFM3ForTraining, add_lora

# Use an already prepared local checkpoint; load it before injecting LoRA.
base = TimesFM3Torch.from_pretrained('/path/to/checkpoint', local_files_only=True)
adapted = add_lora(base, rank=8, alpha=16)
model = TimesFM3ForTraining(adapted).to('cuda:0').train()
optimizer = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad], lr=1e-4)

# context: [batch, target_variates, context_length]
# future_values: [batch, target_variates, forecast_horizon], used ONLY by loss
# past_only_covariates: [batch, covariates, context_length], optional
optimizer.zero_grad(set_to_none=True)
result = model(context, future_values, past_only_covariates=past_covariates)
result['loss'].backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
optimizer.step()
adapted.save_pretrained('/path/to/adapter', safe_serialization=True)
```

Pass `context_mask`, `past_only_mask` or `label_mask` explicitly when needed:
`True` means unavailable. Unmasked nonfinite values, empty series, all-masked
series and shape mismatches fail instead of filling or silently dropping rows.
Missing labels do not contribute to loss. Each sample/target must have at least
one valid label. There is no known-future-covariate shortcut in the training
wrapper, so realized future covariates cannot accidentally leak into it.

Quantiles are unsorted during training, preserving their individually supervised
heads. The public forecaster's patch-rounded decode horizon is preserved and
only the returned output is trimmed. Full base-model training is also possible
by passing the unwrapped `base` to `TimesFM3ForTraining`; the CLI is LoRA-only.

Reload the **same base checkpoint** using PEFT:

```python
from peft import PeftModel

fresh_base = TimesFM3Torch.from_pretrained('/path/to/checkpoint', local_files_only=True)
adapted = PeftModel.from_pretrained(fresh_base, '/path/to/adapter', is_trainable=True)
model = TimesFM3ForTraining(adapted).to('cuda:0')
```

`return_aux_outputs=True` exposes autograd-connected native hidden tensors for
future decision heads. No decision policy, PPO loop or account-value reward is
implemented in this change.

## Memory-mapped chronological data and training command

Prepare **separate** training and validation directories. Each has:

| File | Shape / meaning |
|---|---|
| `target.npy` | floating `[time, targets]` |
| `timestamps.npy` | increasing uniform int64 or datetime64 `[time]` |
| `past_covariates.npy` | optional floating `[time, covariates]` |
| `target_mask.npy` | optional bool, same shape as target |
| `past_mask.npy` | optional bool, same shape as covariates |

The series must already be availability-aligned and use identical column order
and semantics across splits. Windows never cross split boundaries. Validation
also uses its own context warmup, so its first `context` rows are not labels.
No interpolating, resampling, random train/test split or fitted preprocessing is
performed. Data stay memory-mapped; overlapping windows are materialized only
for the current batch. Do not modify the inputs during a run.

```sh
python -m timesfm3.torch.finetune \
  --checkpoint /path/to/local/checkpoint \
  --train-data /path/to/train --valid-data /path/to/valid \
  --output /path/to/new-run --context 512 --horizon 64 \
  --epochs 1 --batch-size 4 --rank 8 --alpha 16 --lr 1e-4 --device cuda:0
```

Defaults are starting points, not validated financial hyperparameters. The CLI
has no CPU fallback and never downloads a checkpoint. `--device cpu` is useful
for tiny random-model unit tests, not an implied recommendation to train the
official checkpoint on CPU. Full-model activations still consume memory under
LoRA; a GPU memory/gradient smoke test is required before scaling a real job.

Each completed epoch saves an adapter and trusted-local trainer state (optimizer
and RNGs). `best.json` selects by validation loss, not test results. To resume at
an epoch boundary, use a **new output directory**, the same base/data/settings,
and `--resume /path/to/run/epoch-0001 --epochs 2`. Interrupted partial epochs are
not marked complete; failure information is saved. Do not delete a prior run if
the resumed run's `best.json` still references its better adapter.

## Acceptance and limits

Tests instantiate tiny random models, with no downloaded weights, network or
GPU dependency:

```sh
OMP_NUM_THREADS=1 HF_HUB_OFFLINE=1 PYTHONPATH=src \
  pytest src/timesfm3/torch/ -q
```

They cover native decode parity, multiple horizon/normalization modes, finite
gradients, numerical LoRA gradient checks, label isolation, zero-adapter parity,
frozen-base updates, loss decrease, safe adapter reload, failure evidence and
deterministic epoch-boundary resume. These gates do **not** establish actual
checkpoint training quality, CUDA correctness/throughput, financial usefulness
or profitability. No raw market dataset or model weights are included here.

Development verification (2026-10-10): 88 CPU tests passed using tiny random
networks (16 hidden dimensions, two layers for the new training tests). An
independent comparison with unmodified upstream `e51928e` covered 192 combinations
of context/horizon, covariates, stitching, detrending, frozen stats and CPM:
all inference values were exactly equal. PEFT save/reload and epoch-boundary
resume matched continuous training exactly on the CPU test fixture. Real
pretrained-checkpoint and GPU validation have not been performed.

The repository source license remains Apache-2.0. Official TimesFM-3 weights and
their derivatives retain Google's separate non-commercial/non-production terms:
<https://huggingface.co/google/timesfm-3.0-pytorch/blob/main/LICENSE>.
Adding training code does not change those restrictions.
