"""Comprehensive headless verification matrix for Fooocus (CUDA).

Covers: txt2img (Speed), Quality mode, different aspect ratios, upscale/img2img,
and two additional base models (realistic + anime) to verify model switching.

Run:  uv run python _smoke_matrix.py
Temporary file — delete after use.
"""
import os
import sys
import ssl
import time
import glob

sys.argv = ["_smoke_matrix.py"]
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")
ssl._create_default_https_context = ssl._create_unverified_context
root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(root)
os.chdir(root)

import torch
import numpy as np
from PIL import Image

import args_manager  # noqa: F401
from modules import config
from modules.model_loader import load_file_from_url
from modules.util import get_file_from_folder_list, get_enabled_loras

import modules.async_worker as worker
from modules.flags import Performance, MetadataScheme, disabled, ip_list, enhancement_uov_before, upscale_2
import ldm_patched.modules.model_management as mm

print(f"[env] torch {torch.__version__} cuda={torch.cuda.is_available()} dev={mm.get_torch_device()}")
print(f"[attn] xformers_enabled={mm.xformers_enabled()} pytorch_attention={mm.pytorch_attention_enabled()} "
      f"flash={mm.pytorch_attention_flash_attention()} is_nvidia={mm.is_nvidia()} VAE_DTYPE={mm.VAE_DTYPE}")


def ensure_checkpoint(file_name, url):
    target = get_file_from_folder_list(file_name, config.paths_checkpoints)
    if not os.path.isfile(target):
        load_file_from_url(url=url, model_dir=os.path.dirname(target), file_name=file_name)
    config.update_files()


def make_task(prompt, base_model, aspect, performance, styles, *,
              uov_method=disabled, uov_input_image=None, image_number=1, seed=1234):
    t = worker.AsyncTask(args=[])
    t.args = [None]
    t.generate_image_grid = False
    t.prompt = prompt
    t.negative_prompt = ""
    t.style_selections = list(styles)
    t.performance_selection = Performance(performance)
    t.steps = t.performance_selection.steps()
    t.original_steps = t.steps
    t.aspect_ratios_selection = config.add_ratio(aspect)
    t.image_number = image_number
    t.output_format = "png"
    t.seed = seed
    t.read_wildcards_in_order = False
    t.sharpness = config.default_sample_sharpness
    t.cfg_scale = config.default_cfg_scale
    t.base_model_name = base_model
    t.refiner_model_name = config.default_refiner_model_name
    t.refiner_switch = config.default_refiner_switch
    t.loras = get_enabled_loras(list(config.default_loras))
    t.input_image_checkbox = uov_input_image is not None
    t.current_tab = "uov"
    t.uov_method = uov_method
    t.uov_input_image = uov_input_image
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
    return t


def run_case(name, task):
    print(f"\n===== CASE {name} | model={task.base_model_name} perf={task.performance_selection.value} "
          f"steps={task.steps} ratio={task.aspect_ratios_selection.split(' ')[0]} =====", flush=True)
    print(f"  prompt: {task.prompt}")
    worker.async_tasks.append(task)
    start = time.perf_counter()
    finished = False
    deadline = start + 900
    while not finished and time.perf_counter() < deadline:
        time.sleep(0.2)
        while task.yields:
            flag, _ = task.yields.pop(0)
            if flag == "finish":
                finished = True
    elapsed = time.perf_counter() - start
    if not finished or not task.results:
        print(f"  [FAIL] {name}: finished={finished} results={len(task.results)} ({elapsed:.0f}s)")
        return name, elapsed, False, None
    img = task.results[0]
    arr = np.asarray(Image.open(img)) if isinstance(img, str) else np.asarray(img).astype("uint8")
    out = f"/tmp/fooocus_{name}.png"
    Image.fromarray(arr).save(out)
    print(f"  [OK] {name}: {elapsed:.1f}s -> {out}  shape={arr.shape}")
    return name, elapsed, True, arr


URL_REAL = "https://huggingface.co/lllyasviel/fav_models/resolve/main/fav/realisticStockPhoto_v20.safetensors"
URL_ANIME = "https://huggingface.co/mashb1t/fav_models/resolve/main/fav/animaPencilXL_v500.safetensors"
JUG = "juggernautXL_v8Rundiffusion.safetensors"
DEF_STYLES = ['Fooocus V2', 'Fooocus Enhance', 'Fooocus Sharp']

summary = []

# 1) photoreal landscape (Speed)
r1 = run_case("01_landscape", make_task(
    "a serene mountain lake at sunrise, pine forest, misty reflection, photorealistic, ultra detailed",
    JUG, "1152*896", "Speed", DEF_STYLES))
summary.append(r1)

# 2) portrait, Quality mode (more steps)
summary.append(run_case("02_portrait_quality", make_task(
    "studio portrait of an elderly fisherman, weathered face, soft window light, 85mm, sharp focus",
    JUG, "896*1152", "Quality", DEF_STYLES)))

# 3) sci-fi, wide aspect
summary.append(run_case("03_scifi", make_task(
    "a futuristic city skyline at night, neon lights, flying cars, cyberpunk, cinematic, highly detailed",
    JUG, "1216*832", "Speed", DEF_STYLES)))

# 4) upscale 2x of case 1 (img2img path)
if r1[3] is not None:
    summary.append(run_case("04_upscale2x", make_task(
        "a serene mountain lake at sunrise, pine forest, misty reflection, photorealistic, ultra detailed",
        JUG, "1152*896", "Speed", DEF_STYLES,
        uov_method=upscale_2, uov_input_image=r1[3])))
else:
    print("[skip] 04_upscale2x: no input from case 1")

# 5) realistic base model
ensure_checkpoint("realisticStockPhoto_v20.safetensors", URL_REAL)
summary.append(run_case("05_realistic_model", make_task(
    "candid photograph of a woman laughing in a cozy cafe, natural window light, 50mm, shallow depth of field",
    "realisticStockPhoto_v20.safetensors", "896*1152", "Speed",
    ['Fooocus V2', 'Fooocus Photograph', 'Fooocus Negative'])))

# 6) anime base model
ensure_checkpoint("animaPencilXL_v500.safetensors", URL_ANIME)
summary.append(run_case("06_anime_model", make_task(
    "anime illustration of a girl with red hair standing in a sunflower field under a blue sky, detailed, masterpiece",
    "animaPencilXL_v500.safetensors", "896*1152", "Speed",
    ['Fooocus V2', 'Fooocus Semi Realistic', 'Fooocus Masterpiece'])))

print("\n================ SUMMARY ================")
ok = 0
for name, elapsed, success, _ in summary:
    print(f"  {'PASS' if success else 'FAIL'}  {name:22s} {elapsed:6.1f}s")
    ok += int(success)
print(f"  {ok}/{len(summary)} cases passed")
print("[OUTPUTS]", *[f"/tmp/fooocus_{s[0]}.png" for s in summary if s[2]])
