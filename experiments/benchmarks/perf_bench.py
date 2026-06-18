"""Performance/throughput benchmark on CUDA (single base model = Juggernaut XL).

Part A: performance modes (Speed/Quality/Extreme Speed/Lightning/Hyper-SD) -> s/image, img/min, VRAM
Part B: batch throughput (image_number=1,2,4) at Speed
Part C: resolution scaling at Speed

Run:  uv run python _perf_bench.py
Temporary file — delete after use.
"""
import os, sys, ssl, time
sys.argv = ["_perf_bench.py"]
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")
ssl._create_default_https_context = ssl._create_unverified_context
root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')); sys.path.append(root); os.chdir(root)

import torch, numpy as np
from PIL import Image
import args_manager  # noqa
from modules import config
from modules.util import get_enabled_loras
import modules.async_worker as worker
from modules.flags import Performance, MetadataScheme, disabled, ip_list, enhancement_uov_before
import ldm_patched.modules.model_management as mm

DEV = mm.get_torch_device()
MODEL = "juggernautXL_v8Rundiffusion.safetensors"
PROMPT = "professional studio headshot portrait of a young woman, soft lighting, sharp focus, 85mm, detailed skin"
STYLES = ['Fooocus V2', 'Fooocus Enhance', 'Fooocus Sharp']
print(f"[env] torch {torch.__version__} dev={DEV} gpu={torch.cuda.get_device_name(0)}")


def prefetch(fn, name):
    for a in range(4):
        try:
            return fn()
        except Exception as e:
            print(f"  [retry {name} {a+1}/4] {e}"); time.sleep(3)
    print(f"  [warn] could not prefetch {name}")
    return None


print("[prefetch] acceleration LoRAs (lcm/lightning/hyper) ...")
prefetch(config.downloading_sdxl_lcm_lora, "lcm")
prefetch(config.downloading_sdxl_lightning_lora, "lightning")
prefetch(config.downloading_sdxl_hyper_sd_lora, "hyper")


def make_task(performance, aspect="1152*896", image_number=1, seed=999):
    t = worker.AsyncTask(args=[]); t.args = [None]
    t.generate_image_grid = False
    t.prompt = PROMPT; t.negative_prompt = ""
    t.style_selections = list(STYLES)
    t.performance_selection = Performance(performance)
    t.steps = t.performance_selection.steps(); t.original_steps = t.steps
    t.aspect_ratios_selection = config.add_ratio(aspect)
    t.image_number = image_number; t.output_format = "png"; t.seed = seed
    t.read_wildcards_in_order = False
    t.sharpness = config.default_sample_sharpness; t.cfg_scale = config.default_cfg_scale
    t.base_model_name = MODEL; t.refiner_model_name = config.default_refiner_model_name
    t.refiner_switch = config.default_refiner_switch
    t.loras = get_enabled_loras(list(config.default_loras))
    t.input_image_checkbox = False; t.current_tab = "uov"; t.uov_method = disabled; t.uov_input_image = None
    t.outpaint_selections = []; t.inpaint_input_image = None; t.inpaint_additional_prompt = ""
    t.inpaint_mask_image_upload = None
    t.disable_preview = True; t.disable_intermediate_results = True; t.disable_seed_increment = True
    t.black_out_nsfw = False
    t.adm_scaler_positive = 1.5; t.adm_scaler_negative = 0.8; t.adm_scaler_end = 0.3
    t.adaptive_cfg = config.default_cfg_tsnr; t.clip_skip = config.default_clip_skip
    t.sampler_name = config.default_sampler; t.scheduler_name = config.default_scheduler
    t.vae_name = config.default_vae
    t.overwrite_step = -1; t.overwrite_switch = -1; t.overwrite_width = -1; t.overwrite_height = -1
    t.overwrite_vary_strength = -1; t.overwrite_upscale_strength = -1
    t.mixing_image_prompt_and_vary_upscale = False; t.mixing_image_prompt_and_inpaint = False
    t.debugging_cn_preprocessor = False; t.skipping_cn_preprocessor = False
    t.canny_low_threshold = 64; t.canny_high_threshold = 128
    t.refiner_swap_method = "joint"; t.controlnet_softness = 0.25
    t.freeu_enabled = False; t.freeu_b1, t.freeu_b2, t.freeu_s1, t.freeu_s2 = 1.01, 1.02, 0.99, 0.95
    t.debugging_inpaint_preprocessor = False; t.inpaint_disable_initial_latent = False
    t.inpaint_engine = config.default_inpaint_engine_version; t.inpaint_strength = 1.0
    t.inpaint_respective_field = 0.618; t.inpaint_advanced_masking_checkbox = False
    t.invert_mask_checkbox = False; t.inpaint_erode_or_dilate = 0
    t.save_final_enhanced_image_only = False; t.save_metadata_to_images = False
    t.metadata_scheme = MetadataScheme.FOOOCUS
    t.cn_tasks = {x: [] for x in ip_list}
    t.debugging_dino = False; t.dino_erode_or_dilate = 0; t.debugging_enhance_masks_checkbox = False
    t.enhance_input_image = None; t.enhance_checkbox = False; t.enhance_uov_method = disabled
    t.enhance_uov_processing_order = enhancement_uov_before; t.enhance_uov_prompt_type = "Original Prompts"
    t.enhance_ctrls = []; t.should_enhance = False; t.images_to_enhance_count = 0; t.enhance_stats = {}
    t.performance_loras = []
    return t


def run(task):
    torch.cuda.reset_peak_memory_stats(DEV)
    worker.async_tasks.append(task)
    start = time.perf_counter(); finished = False
    while not finished and time.perf_counter() - start < 900:
        time.sleep(0.1)
        while task.yields:
            flag, _ = task.yields.pop(0)
            finished = finished or flag == "finish"
    elapsed = time.perf_counter() - start
    return elapsed, torch.cuda.max_memory_reserved(DEV) / 1e9, len(task.results), task.sampler_name


MODES = ["Speed", "Quality", "Extreme Speed", "Lightning", "Hyper-SD"]
print("\n########## PART A: performance modes (1 image, 1152x896) ##########")
partA = []
for m in MODES:
    run(make_task(m))                       # warm-up (load model + mode LoRA), discarded
    times, vram, sampler, steps = [], 0, "", 0
    for _ in range(2):                      # 2 timed warm runs
        t = make_task(m)
        el, vr, n, samp = run(t)
        times.append(el); vram = max(vram, vr); sampler = samp; steps = t.steps
    avg = sum(times) / len(times)
    partA.append((m, steps, sampler, avg, 60.0 / avg, vram))
    print(f"  {m:14s} steps={steps:2d} sampler={sampler:14s} {avg:5.2f}s/img  {60/avg:5.1f} img/min  VRAM={vram:.2f}GB")

print("\n########## PART B: batch throughput (Speed, 1152x896) ##########")
partB = []
for n in [1, 2, 4]:
    t = make_task("Speed", image_number=n)
    el, vr, cnt, _ = run(t)
    partB.append((n, el, el / max(cnt, 1), vr, cnt))
    print(f"  batch={n}: {el:5.2f}s total  {el/max(cnt,1):5.2f}s/img  VRAM={vr:.2f}GB  ({cnt} imgs)")

print("\n########## PART C: resolution scaling (Speed, 1 image) ##########")
partC = []
for ar in ["1024*1024", "1152*896", "1344*768"]:
    run(make_task("Speed", aspect=ar))      # warm-up for this res
    t = make_task("Speed", aspect=ar)
    el, vr, cnt, _ = run(t)
    partC.append((ar, el, vr, cnt))
    print(f"  {ar}: {el:5.2f}s  VRAM={vr:.2f}GB")

# ---- tables ----
print("\n\n================ A) PERFORMANCE MODES (Juggernaut XL, 1152x896) ================")
print(f"| {'Mode':14s} | {'Steps':5s} | {'Sampler':14s} | {'s/image':7s} | {'img/min':7s} | {'peak VRAM':9s} |")
print("|" + "-"*16 + "|" + "-"*7 + "|" + "-"*16 + "|" + "-"*9 + "|" + "-"*9 + "|" + "-"*11 + "|")
for m, steps, samp, avg, ipm, vram in partA:
    print(f"| {m:14s} | {steps:5d} | {samp:14s} | {avg:6.2f}s | {ipm:6.1f}  | {vram:6.2f} GB |")

print("\n================ B) BATCH THROUGHPUT (Speed) ================")
print(f"| {'Batch':5s} | {'total s':7s} | {'s/image':7s} | {'img/min':7s} | {'peak VRAM':9s} |")
print("|" + "-"*7 + "|" + "-"*9 + "|" + "-"*9 + "|" + "-"*9 + "|" + "-"*11 + "|")
for n, el, per, vr, cnt in partB:
    print(f"| {n:5d} | {el:6.2f}s | {per:6.2f}s | {60/per:6.1f}  | {vr:6.2f} GB |")

print("\n================ C) RESOLUTION SCALING (Speed) ================")
print(f"| {'Resolution':11s} | {'s/image':7s} | {'peak VRAM':9s} |")
print("|" + "-"*13 + "|" + "-"*9 + "|" + "-"*11 + "|")
for ar, el, vr, cnt in partC:
    print(f"| {ar:11s} | {el:6.2f}s | {vr:6.2f} GB |")
print("\n[OK] perf benchmark complete")
