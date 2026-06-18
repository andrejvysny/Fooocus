# Backend & platform strategy

This document records the engineering decisions behind the migration of Fooocus to a
**uv-managed, macOS + Linux** project, and the research into accelerating image generation on
Apple Silicon (MLX vs PyTorch-MPS vs Core ML). It explains the *why* so future changes stay aligned.

## Targets (priority order)

1. **Linux + NVIDIA (CUDA)** — the production target. Fully supported and best-optimized.
2. **macOS Apple Silicon (MPS)** — development / experiments. Correct and usable, but slower.
3. Windows — **removed** (DirectML, embedded-Python launcher, `.bat` files all deleted).

CUDA is roughly **2–5× faster** than Apple Silicon for SDXL regardless of backend, so heavy
generation belongs on Linux/NVIDIA; the Mac path is tuned for correctness and fast dev iteration.

## Why the core stays on PyTorch (MLX is *not* integrated)

Deep web research (June 2026, adversarially verified) concluded that porting Fooocus's pipeline
to Apple's MLX is **not worthwhile**:

- **No MLX stack supports Fooocus's feature set.** Fooocus needs SDXL UNet + custom karras/dpm
  samplers + LoRA + ControlNet + inpainting + IP-Adapter + GPT2 prompt-expansion. The MLX
  ecosystem does not cover this:
  - `mlx-examples/stable_diffusion` — only SDXL-Turbo + SD 2.1, no LoRA/ControlNet/inpainting.
  - `mflux` — FLUX-only; never implemented SDXL.
  - `DiffusionKit` (argmax) — SD3/FLUX-only, and **archived/read-only** in 2026.
  - `ComfyUI-MLX` — listed SDXL ControlNet/LoRA as roadmap-only; repo now **404**.
- **A port ≈ rewriting the whole backend.** `ldm_patched/` is ~35k LOC across 111 files (104 import
  torch). Reimplementing UNet, samplers, ControlNet, LoRA, inpainting and IP-Adapter in MLX is a
  multi-month effort with numerical-parity risk.
- **MLX is 2–5× slower on Conv2D** — the operation that dominates SDXL's convolutional UNet, VAE
  decoder, and ControlNet. MLX wins on linear/softmax/attention ops, but loses on exactly the op
  SDXL leans on most, so even a successful port might not be faster.
- **MLX↔PyTorch interop is experimental and copy-based** (memoryview/NumPy round-trips, no
  documented zero-copy GPU handoff), so a hybrid that swaps tensors every denoise step is fragile
  and likely copy-bound.

**Decision:** keep the proven PyTorch backend (CUDA on Linux, MPS on macOS). MLX is provided as an
isolated **sandbox** for experimentation — see [`../experiments/`](../experiments/) — not as a
Fooocus backend. (If you just want the fastest turnkey Mac generator, the native app *Draw Things*
is the practical leader, but it is a separate Swift/Metal product, not a reusable Python library.)

## Apple Silicon (MPS) tuning — conservative on purpose

Research confirms Fooocus's existing MPS behavior is already the correct, safe configuration, so
**no behavioral MPS changes were made**:

- `fp32` is the safe default on MPS. The classic fp16 "black image" problem originates in the VAE;
  Fooocus already keeps fp16 off for MPS (`should_use_fp16` returns `False` for MPS devices).
- The prompt-expansion GPT2 model runs on CPU for MPS (a deliberate, working workaround).
- `launch.py` sets the recommended environment variables: `PYTORCH_ENABLE_MPS_FALLBACK=1` (routes
  unsupported ops to CPU) and `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0` (avoids the OOM watermark).
- xformers is CUDA-only and correctly disabled on MPS; attention falls back to PyTorch SDPA.

Enabling fp16/bf16 on MPS or moving expansion to MPS could speed things up but risks NaNs/black
images and must be validated on real Apple hardware — left as a future, opt-in experiment.

## Attention / xformers

xformers is **dropped from the default dependencies**. Modern PyTorch enables SDPA flash-attention
automatically on NVIDIA (`ENABLE_PYTORCH_ATTENTION`), and `model_management` already falls back to
SDPA when xformers is absent. To use xformers on Linux/CUDA anyway: `uv add xformers`.

## Dependency version policy ("modernize broadly", with documented holds)

`pyproject.toml` uses `>=` floors so `uv lock` modernizes the stack (e.g. torch `2.1 → 2.11/2.12`,
torchvision `0.16 → 0.26`). A few packages are **deliberately held** because bumping them breaks the
app — each is annotated in `pyproject.toml`:

| Held package | Why |
|---|---|
| `gradio==3.41.2` | the entire `webui.py` is built on the Gradio 3 API; 4/5 is a separate migration |
| `transformers==4.42.4` (+ `tokenizers<0.20`) | Fooocus pins these for SDXL CLIP + GPT2 loading parity |
| `numpy<2` | required by Gradio 3.x (`numpy~=1.0`) and by older deps (groundingdino-py, rembg) |
| `pillow<11` | capped by Gradio 3.41.2 |

`groundingdino-py==0.4.0` is sdist-only and imports torch at build time, so
`[tool.uv.extra-build-dependencies]` injects torch into its build environment for `uv lock`/`uv sync`.

## Indicative performance (from research — directional, not lab-grade)

| Backend / tool | Model | Hardware | ~Time |
|---|---|---|---|
| PyTorch-MPS (ComfyUI-class) | SDXL 1024² | M4 Max | ~85 s |
| Apple Core ML (GPU) | SDXL 1024², 20 steps | M2 Ultra | ~20 s (1.1 it/s) |
| mflux (MLX) | FLUX schnell 1024², 2 steps, 4-bit | M4 Max | ~10–19 s |
| PyTorch-MPS | SDXL + 5 LoRA, batch 8 | M4 Max | ~42 s/batch |
| PyTorch-CUDA | SDXL + 5 LoRA, batch 8 | RTX 4090 | ~11 s/batch |

Takeaway: use Linux/NVIDIA for real throughput; use the Mac for development and the MLX sandbox for
exploring FLUX-class models.

## Key sources

- uv PyTorch integration guide — https://docs.astral.sh/uv/guides/integration/pytorch/
- uv Docker guide — https://docs.astral.sh/uv/guides/integration/docker/
- PyTorch on Mac (MPS) — https://developer.apple.com/metal/pytorch/ , https://huggingface.co/docs/diffusers/en/optimization/mps
- mflux — https://github.com/filipstrand/mflux
- DiffusionKit (archived) — https://github.com/argmaxinc/DiffusionKit
- Apple ml-stable-diffusion — https://github.com/apple/ml-stable-diffusion
- MLX vs MPS op benchmark — https://towardsdatascience.com/how-fast-is-mlx-a-comprehensive-benchmark-on-8-apple-silicon-chips-and-4-cuda-gpus-378a0ae356a0/
