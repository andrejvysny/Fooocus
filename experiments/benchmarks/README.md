# Headless generation & benchmark harnesses

Standalone scripts that drive Fooocus's real pipeline **without the Gradio UI** — used to
verify the uv/CUDA migration and to benchmark models. They build an `AsyncTask`, set its fields
by name, submit it to the worker, and save results to `/tmp`. Run from anywhere (each script
`chdir`s to the repo root) on a CUDA box:

```sh
uv run python -u experiments/benchmarks/smoke_generate.py     # one image, end-to-end smoke test
uv run python -u experiments/benchmarks/matrix.py             # txt2img types + realistic/anime models + upscale
uv run python -u experiments/benchmarks/portrait_bench.py     # portraits across models, time + VRAM table
uv run python -u experiments/benchmarks/perf_bench.py         # Speed/Quality/Extreme/Lightning/Hyper-SD + batch + resolution
uv run python -u experiments/benchmarks/quality_bench.py      # quality-first photoreal persons + 2x upscale
uv run python -u experiments/benchmarks/fullbody_bench.py     # full-length / full-pose persons
```

Notes:
- First run downloads the relevant SDXL checkpoints / acceleration LoRAs / upscaler into `models/`.
- Output images and per-case time + peak VRAM are written/printed; images go to `/tmp/`.
- These are development/CI harnesses (hardcoded prompts + `/tmp` paths), not part of the app.

Measured on a single RTX 4090 (torch cu128, SDPA flash-attention, bf16 VAE): Speed ≈ 6 s/image,
Lightning ≈ 1.8 s/image; peak VRAM ≈ 6 GB (≈8 GB at 2× upscale). See `../../docs/backend-strategy.md`.
