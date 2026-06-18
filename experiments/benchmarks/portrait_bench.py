"""Portrait/person generation benchmark across multiple base models on CUDA.

Generates portrait photos with identical settings across models, tracking wall time
(cold incl. model load vs warm) and peak GPU/CPU memory, then prints a comparison table.

Run:  uv run python _portrait_bench.py
Temporary file — delete after use.
"""
import os
import sys
import ssl
import time
import psutil

sys.argv = ["_portrait_bench.py"]
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
from modules.flags import Performance, MetadataScheme, disabled, ip_list, enhancement_uov_before
import ldm_patched.modules.model_management as mm

DEV = mm.get_torch_device()
proc = psutil.Process()
print(f"[env] torch {torch.__version__} dev={DEV} gpu={torch.cuda.get_device_name(0)}")

MODELS = [
    ("Juggernaut XL v8 (general)", "juggernautXL_v8Rundiffusion.safetensors",
     "https://huggingface.co/lllyasviel/fav_models/resolve/main/fav/juggernautXL_v8Rundiffusion.safetensors"),
    ("RealisticStockPhoto v2 (realistic)", "realisticStockPhoto_v20.safetensors",
     "https://huggingface.co/lllyasviel/fav_models/resolve/main/fav/realisticStockPhoto_v20.safetensors"),
    ("AnimaPencil XL v5 (anime)", "animaPencilXL_v500.safetensors",
     "https://huggingface.co/mashb1t/fav_models/resolve/main/fav/animaPencilXL_v500.safetensors"),
]
# Identical prompts + settings across all models for a fair comparison.
PROMPTS = [
    ("woman", "professional studio headshot portrait of a young woman, soft beauty lighting, "
              "sharp focus, 85mm, detailed skin, looking at camera"),
    ("man", "portrait photo of an elderly man with a short white beard, dramatic rim lighting, "
            "weathered skin, shallow depth of field, photorealistic"),
]
STYLES = ['Fooocus V2', 'Fooocus Enhance', 'Fooocus Sharp']
ASPECT = "896*1152"
PERF = "Speed"
SEED = 7777


def ensure_checkpoint(file_name, url):
    target = get_file_from_folder_list(file_name, config.paths_checkpoints)
    if not os.path.isfile(target):
        print(f"[dl] {file_name} ...")
        load_file_from_url(url=url, model_dir=os.path.dirname(target), file_name=file_name)
    config.update_files()


def make_task(prompt, base_model, seed):
    t = worker.AsyncTask(args=[])
    t.args = [None]
    t.generate_image_grid = False
    t.prompt = prompt
    t.negative_prompt = ""
    t.style_selections = list(STYLES)
    t.performance_selection = Performance(PERF)
    t.steps = t.performance_selection.steps()
    t.original_steps = t.steps
    t.aspect_ratios_selection = config.add_ratio(ASPECT)
    t.image_number = 1
    t.output_format = "png"
    t.seed = seed
    t.read_wildcards_in_order = False
    t.sharpness = config.default_sample_sharpness
    t.cfg_scale = config.default_cfg_scale
    t.base_model_name = base_model
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
    return t


def run(task):
    torch.cuda.reset_peak_memory_stats(DEV)
    worker.async_tasks.append(task)
    start = time.perf_counter()
    finished = False
    while not finished and time.perf_counter() - start < 900:
        time.sleep(0.15)
        while task.yields:
            flag, _ = task.yields.pop(0)
            finished = finished or flag == "finish"
    elapsed = time.perf_counter() - start
    peak_alloc = torch.cuda.max_memory_allocated(DEV) / 1e9
    peak_resv = torch.cuda.max_memory_reserved(DEV) / 1e9
    rss = proc.memory_info().rss / 1e9
    ok = finished and bool(task.results)
    return elapsed, peak_alloc, peak_resv, rss, (task.results[0] if ok else None)


rows = []
for label, fname, url in MODELS:
    ensure_checkpoint(fname, url)
    per_model = []
    for i, (tag, prompt) in enumerate(PROMPTS):
        t = make_task(prompt, fname, SEED + i)
        elapsed, pa, pr, rss, img = run(t)
        phase = "cold(+load)" if i == 0 else "warm"
        print(f"[{label}] {tag} {phase}: {elapsed:.1f}s  peakVRAM(alloc/resv)={pa:.2f}/{pr:.2f}GB  RSS={rss:.1f}GB")
        if img is not None:
            out = f"/tmp/portrait_{fname.split('.')[0]}_{tag}.png"
            arr = np.asarray(Image.open(img)) if isinstance(img, str) else np.asarray(img).astype("uint8")
            Image.fromarray(arr).save(out)
        per_model.append((tag, elapsed, pa, pr, rss, img is not None))
    rows.append((label, fname, per_model))

# ---- comparison table ----
print("\n\n================ PORTRAIT MODEL COMPARISON ================")
print(f"Settings: performance={PERF} (30 steps), resolution={ASPECT}, styles={STYLES}, seed={SEED}+i, device={DEV}")
hdr = f"| {'Model':34s} | {'cold s (incl load)':>18s} | {'warm s':>7s} | {'~load s':>7s} | {'peak VRAM resv':>14s} | {'RSS':>6s} | ok |"
print("\n" + hdr)
print("|" + "-" * (len(hdr) - 2) + "|")
for label, fname, pm in rows:
    cold = pm[0][1]
    warm = pm[1][1] if len(pm) > 1 else float('nan')
    load = cold - warm
    peak = max(x[3] for x in pm)
    rss = max(x[4] for x in pm)
    allok = all(x[5] for x in pm)
    print(f"| {label:34s} | {cold:18.1f} | {warm:7.1f} | {load:7.1f} | {peak:11.2f} GB | {rss:4.1f}G | {'Y' if allok else 'N'} |")

print("\n[IMAGES]")
for label, fname, pm in rows:
    for tag, *_ in pm:
        print(f"  /tmp/portrait_{fname.split('.')[0]}_{tag}.png   ({label} / {tag})")
print("\n[OK] portrait benchmark complete")
