# Experiments (Apple Silicon / MLX)

Standalone, **isolated** experiments for the macOS development box. Nothing here is imported
by Fooocus or part of its dependency lock — each script declares its own dependencies via
[PEP 723](https://peps.python.org/pep-0723/) inline metadata and runs in an ephemeral `uv`
environment, so it cannot conflict with Fooocus's pinned (Gradio-3-era) dependencies.

See [`../docs/backend-strategy.md`](../docs/backend-strategy.md) for the research that explains
why MLX is offered only as a sandbox and is **not** integrated into Fooocus's SDXL pipeline.

## `mlx_flux_generate.py` — FLUX on Apple Silicon via MLX (mflux)

```sh
# macOS, Apple Silicon only:
uv run experiments/mlx_flux_generate.py --prompt "a red panda barista, studio light"
uv run experiments/mlx_flux_generate.py -m dev --steps 20 -q 8 -p "a watercolor fox"
```

Or skip the wrapper entirely and use mflux's CLI in an isolated tool environment:

```sh
uvx --from mflux mflux-generate --model schnell --steps 2 --quantize 4 \
    --prompt "a red panda barista" --output mlx_out.png
```

mflux covers FLUX.1 schnell/dev (and newer models), 3/4/6/8-bit quantization, LoRA and
ControlNet — but it is a **different engine** from Fooocus (no SDXL, no Fooocus pipeline/styles).
Use it to explore raw MLX generation speed on your Mac, not as a Fooocus backend.
