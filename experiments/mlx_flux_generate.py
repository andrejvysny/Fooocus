#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "mflux ; sys_platform == 'darwin' and platform_machine == 'arm64'",
# ]
# ///
"""
Apple Silicon MLX image-generation experiment (FLUX via mflux) — Mac development only.

This is an ISOLATED sandbox, deliberately NOT part of Fooocus's dependency graph
(see ../docs/backend-strategy.md for why MLX is not integrated into the SDXL pipeline).
It runs in its own ephemeral uv environment thanks to the inline PEP 723 metadata above,
so it cannot conflict with Fooocus's pinned (Gradio-3-era) dependencies.

Usage (macOS, Apple Silicon):
    uv run experiments/mlx_flux_generate.py --prompt "a red panda barista, studio light"

Equivalent one-liner without this wrapper (also isolated):
    uvx --from mflux mflux-generate --model schnell --steps 2 --quantize 4 \
        --prompt "a red panda barista" --output mlx_out.png

mflux supports FLUX.1 schnell/dev (and newer models), 3/4/6/8-bit quantization, LoRA and
ControlNet — but only on Apple Silicon. It is a separate engine from Fooocus's SDXL backend.
"""
import argparse
import platform
import shutil
import subprocess
import sys


def main() -> int:
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        print("This experiment only runs on macOS Apple Silicon (arm64), where MLX/mflux work.")
        print(f"Detected: {platform.system()} / {platform.machine()}.")
        print("On the production target (Linux + NVIDIA) use Fooocus itself: ./run.sh")
        return 1

    parser = argparse.ArgumentParser(
        description="Generate an image with FLUX on Apple Silicon via mflux (MLX).")
    parser.add_argument("--prompt", "-p", required=True, help="Text prompt.")
    parser.add_argument("--model", "-m", default="schnell",
                        help="schnell (fast, few-step) or dev (slower, higher quality). Default: schnell.")
    parser.add_argument("--steps", type=int, default=2,
                        help="Inference steps (schnell: 2-4; dev: 20-25). Default: 2.")
    parser.add_argument("--quantize", "-q", type=int, default=4, choices=[3, 4, 6, 8],
                        help="Weight quantization bits (lower = less memory). Default: 4.")
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output", "-o", default="mlx_out.png", help="Output PNG path.")
    args = parser.parse_args()

    # Delegate to mflux's own CLI (more stable across mflux versions than its Python API).
    cli = shutil.which("mflux-generate")
    if cli is None:
        print("mflux-generate not found. Run this script with uv so mflux is provisioned:")
        print("    uv run experiments/mlx_flux_generate.py --prompt \"...\"")
        return 1

    cmd = [
        cli,
        "--model", args.model,
        "--prompt", args.prompt,
        "--steps", str(args.steps),
        "--quantize", str(args.quantize),
        "--height", str(args.height),
        "--width", str(args.width),
        "--output", args.output,
    ]
    if args.seed is not None:
        cmd += ["--seed", str(args.seed)]

    print("Running:", " ".join(cmd))
    return subprocess.call(cmd)


if __name__ == "__main__":
    sys.exit(main())
