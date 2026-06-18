"""Headless smoke test: generate one SDXL image through Fooocus's real pipeline (CUDA).

Run with:  uv run python _smoke_generate.py
Temporary file — delete after use.
"""
import os
import sys
import ssl
import time

# --- mirror launch.py preamble (must run before importing args/config) ---
sys.argv = ["_smoke_generate.py"]
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")
ssl._create_default_https_context = ssl._create_unverified_context

root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(root)
os.chdir(root)

PROMPT = ("a red vintage bicycle leaning against a blue wooden door, "
          "a pot of yellow flowers beside it, bright sunny day, photorealistic")
SEED = 1234
OUT = "/tmp/fooocus_smoke.png"

import torch
print(f"[env] torch {torch.__version__} | cuda_available={torch.cuda.is_available()} | "
      f"device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}")

import args_manager  # noqa: F401  (parses argv -> defaults)
from modules import config
from modules.model_loader import load_file_from_url

# --- download the default models (idempotent), mirroring launch.py.download_models ---
print("[models] ensuring default models are present ...")
for fn, url in [
    ("xlvaeapp.pth", "https://huggingface.co/lllyasviel/misc/resolve/main/xlvaeapp.pth"),
    ("vaeapp_sd15.pth", "https://huggingface.co/lllyasviel/misc/resolve/main/vaeapp_sd15.pt"),
    ("xl-to-v1_interposer-v4.0.safetensors",
     "https://huggingface.co/mashb1t/misc/resolve/main/xl-to-v1_interposer-v4.0.safetensors"),
]:
    load_file_from_url(url=url, model_dir=config.path_vae_approx, file_name=fn)
load_file_from_url(
    url="https://huggingface.co/lllyasviel/misc/resolve/main/fooocus_expansion.bin",
    model_dir=config.path_fooocus_expansion, file_name="pytorch_model.bin")
for file_name, url in config.checkpoint_downloads.items():
    model_dir = os.path.dirname(
        __import__("modules.util", fromlist=["get_file_from_folder_list"])
        .get_file_from_folder_list(file_name, config.paths_checkpoints))
    load_file_from_url(url=url, model_dir=model_dir, file_name=file_name)
config.update_files()

# --- import the worker (starts its background thread + patch_all) ---
import modules.async_worker as worker
from modules.flags import Performance, MetadataScheme, disabled, ip_list, enhancement_uov_before
from modules.util import get_enabled_loras

import ldm_patched.modules.model_management as mm
print(f"[device] Fooocus torch device = {mm.get_torch_device()}")

# --- build an AsyncTask with empty args, then set every field by name ---
t = worker.AsyncTask(args=[])
t.args = [None]  # non-empty sentinel

t.generate_image_grid = False
t.prompt = PROMPT
t.negative_prompt = ""
t.style_selections = list(config.default_styles)
t.performance_selection = Performance(config.default_performance)
t.steps = t.performance_selection.steps()
t.original_steps = t.steps
t.aspect_ratios_selection = config.default_aspect_ratio
t.image_number = 1
t.output_format = "png"
t.seed = SEED
t.read_wildcards_in_order = False
t.sharpness = config.default_sample_sharpness
t.cfg_scale = config.default_cfg_scale
t.base_model_name = config.default_base_model_name
t.refiner_model_name = config.default_refiner_model_name
t.refiner_switch = config.default_refiner_switch
t.loras = get_enabled_loras(list(config.default_loras))
t.input_image_checkbox = False
t.current_tab = "uov"
t.uov_method = disabled
t.uov_input_image = None
t.outpaint_selections = []
t.inpaint_input_image = None
t.inpaint_additional_prompt = ""
t.inpaint_mask_image_upload = None
t.disable_preview = True
t.disable_intermediate_results = True
t.disable_seed_increment = True
t.black_out_nsfw = False
t.adm_scaler_positive = 1.5
t.adm_scaler_negative = 0.8
t.adm_scaler_end = 0.3
t.adaptive_cfg = config.default_cfg_tsnr
t.clip_skip = config.default_clip_skip
t.sampler_name = config.default_sampler
t.scheduler_name = config.default_scheduler
t.vae_name = config.default_vae
t.overwrite_step = -1
t.overwrite_switch = -1
t.overwrite_width = -1
t.overwrite_height = -1
t.overwrite_vary_strength = -1
t.overwrite_upscale_strength = -1
t.mixing_image_prompt_and_vary_upscale = False
t.mixing_image_prompt_and_inpaint = False
t.debugging_cn_preprocessor = False
t.skipping_cn_preprocessor = False
t.canny_low_threshold = 64
t.canny_high_threshold = 128
t.refiner_swap_method = "joint"
t.controlnet_softness = 0.25
t.freeu_enabled = False
t.freeu_b1, t.freeu_b2, t.freeu_s1, t.freeu_s2 = 1.01, 1.02, 0.99, 0.95
t.debugging_inpaint_preprocessor = False
t.inpaint_disable_initial_latent = False
t.inpaint_engine = config.default_inpaint_engine_version
t.inpaint_strength = 1.0
t.inpaint_respective_field = 0.618
t.inpaint_advanced_masking_checkbox = False
t.invert_mask_checkbox = False
t.inpaint_erode_or_dilate = 0
t.save_final_enhanced_image_only = False
t.save_metadata_to_images = False
t.metadata_scheme = MetadataScheme.FOOOCUS
t.cn_tasks = {x: [] for x in ip_list}
t.debugging_dino = False
t.dino_erode_or_dilate = 0
t.debugging_enhance_masks_checkbox = False
t.enhance_input_image = None
t.enhance_checkbox = False
t.enhance_uov_method = disabled
t.enhance_uov_processing_order = enhancement_uov_before
t.enhance_uov_prompt_type = "Original Prompts"
t.enhance_ctrls = []
t.should_enhance = False
t.images_to_enhance_count = 0
t.enhance_stats = {}
t.performance_loras = []

print(f"[task] perf={t.performance_selection.value} steps={t.steps} ratio={t.aspect_ratios_selection} "
      f"sampler={t.sampler_name} scheduler={t.scheduler_name} model={t.base_model_name}")
print(f"[task] PROMPT: {PROMPT}")

# --- submit + wait ---
worker.async_tasks.append(t)
start = time.perf_counter()
finished = False
deadline = start + 900
while not finished and time.perf_counter() < deadline:
    time.sleep(0.2)
    while t.yields:
        flag, product = t.yields.pop(0)
        if flag == "preview":
            pct, title, _ = product
            print(f"  ... {pct}% {title}", flush=True)
        elif flag == "finish":
            finished = True

elapsed = time.perf_counter() - start
if not finished:
    print(f"[FAIL] timed out after {elapsed:.0f}s")
    sys.exit(2)

print(f"[done] generation finished in {elapsed:.1f}s; {len(t.results)} image(s)")

# --- save the first result for inspection ---
from PIL import Image
import numpy as np
img = t.results[0]
if isinstance(img, str):
    Image.open(img).save(OUT)
    print(f"[saved] copied {img} -> {OUT}")
else:
    Image.fromarray(np.asarray(img).astype("uint8")).save(OUT)
    print(f"[saved] {OUT}  shape={np.asarray(img).shape}")

# report the PNG Fooocus wrote under outputs/
import glob
pngs = sorted(glob.glob("outputs/**/*.png", recursive=True), key=os.path.getmtime)
if pngs:
    print(f"[outputs] newest: {pngs[-1]}")
print("[OK] smoke test complete")
