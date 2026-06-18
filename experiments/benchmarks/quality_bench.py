"""Quality-first photorealistic PERSON benchmark on CUDA. Quality over speed.

Compares photoreal models + quality settings (Quality 60-step, 100-step, strong negative,
2x upscale) for high-quality person portraits. Tracks time/VRAM but prioritizes quality.

Run:  uv run python _quality_bench.py
Temporary file — delete after use.
"""
import os, sys, ssl, time
sys.argv = ["_quality_bench.py"]
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")
ssl._create_default_https_context = ssl._create_unverified_context
root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')); sys.path.append(root); os.chdir(root)

import torch, numpy as np
from PIL import Image
import args_manager  # noqa
from modules import config
from modules.model_loader import load_file_from_url
from modules.util import get_file_from_folder_list, get_enabled_loras
import modules.async_worker as worker
from modules.flags import Performance, MetadataScheme, disabled, ip_list, enhancement_uov_before, upscale_2
import ldm_patched.modules.model_management as mm

DEV = mm.get_torch_device()
print(f"[env] torch {torch.__version__} dev={DEV} gpu={torch.cuda.get_device_name(0)}")

REAL = "realisticStockPhoto_v20.safetensors"
JUG = "juggernautXL_v8Rundiffusion.safetensors"
STYLES = ['Fooocus V2', 'Fooocus Photograph', 'Fooocus Negative']
PROMPT_W = ("RAW photo, head-and-shoulders portrait of a 30 year old woman with light freckles, "
            "natural makeup, gentle smile, looking at camera, soft window light, shot on 85mm f/1.4, "
            "shallow depth of field, ultra detailed skin texture with pores, catchlights in the eyes, "
            "professional photography, 4k")
PROMPT_M = ("RAW photo, head-and-shoulders portrait of a 55 year old man with grey stubble and "
            "weathered skin, wearing a wool coat, overcast natural light, shot on 85mm f/1.8, "
            "shallow depth of field, ultra detailed skin texture, sharp eyes, professional photography, 4k")
NEG_STRONG = ("cartoon, 3d render, cgi, illustration, painting, anime, plastic skin, waxy skin, "
              "airbrushed, deformed, disfigured, extra fingers, mutated hands, blurry, low quality, "
              "jpeg artifacts, watermark, text, oversaturated")
SEED = 20260618


def prefetch(fn, name):
    for a in range(5):
        try:
            return fn()
        except Exception as e:
            print(f"  [retry {name} {a+1}/5] {e}"); time.sleep(4)
    print(f"  [warn] could not prefetch {name}")
    return None


print("[prefetch] upscaler model (retry-safe) ...")
prefetch(config.downloading_upscale_model, "upscaler")


def ensure(fname, url):
    target = get_file_from_folder_list(fname, config.paths_checkpoints)
    if not os.path.isfile(target):
        prefetch(lambda: load_file_from_url(url=url, model_dir=os.path.dirname(target), file_name=fname), fname)
    config.update_files()


def make_task(prompt, base_model, *, aspect="896*1152", performance="Quality", negative="",
              overwrite_step=-1, uov_method=disabled, uov_input_image=None, seed=SEED):
    t = worker.AsyncTask(args=[]); t.args = [None]
    t.generate_image_grid = False
    t.prompt = prompt; t.negative_prompt = negative
    t.style_selections = list(STYLES)
    t.performance_selection = Performance(performance)
    t.steps = t.performance_selection.steps(); t.original_steps = t.steps
    t.aspect_ratios_selection = config.add_ratio(aspect)
    t.image_number = 1; t.output_format = "png"; t.seed = seed
    t.read_wildcards_in_order = False
    t.sharpness = config.default_sample_sharpness; t.cfg_scale = config.default_cfg_scale
    t.base_model_name = base_model; t.refiner_model_name = config.default_refiner_model_name
    t.refiner_switch = config.default_refiner_switch
    t.loras = get_enabled_loras(list(config.default_loras))
    t.input_image_checkbox = uov_input_image is not None
    t.current_tab = "uov"; t.uov_method = uov_method; t.uov_input_image = uov_input_image
    t.outpaint_selections = []; t.inpaint_input_image = None; t.inpaint_additional_prompt = ""
    t.inpaint_mask_image_upload = None
    t.disable_preview = True; t.disable_intermediate_results = True; t.disable_seed_increment = True
    t.black_out_nsfw = False
    t.adm_scaler_positive = 1.5; t.adm_scaler_negative = 0.8; t.adm_scaler_end = 0.3
    t.adaptive_cfg = config.default_cfg_tsnr; t.clip_skip = config.default_clip_skip
    t.sampler_name = config.default_sampler; t.scheduler_name = config.default_scheduler
    t.vae_name = config.default_vae
    t.overwrite_step = overwrite_step; t.overwrite_switch = -1
    t.overwrite_width = -1; t.overwrite_height = -1
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


def run(task, name):
    torch.cuda.reset_peak_memory_stats(DEV)
    worker.async_tasks.append(task)
    start = time.perf_counter(); finished = False
    while not finished and time.perf_counter() - start < 1200:
        time.sleep(0.2)
        while task.yields:
            flag, _ = task.yields.pop(0)
            finished = finished or flag == "finish"
    el = time.perf_counter() - start
    vr = torch.cuda.max_memory_reserved(DEV) / 1e9
    if not finished or not task.results:
        print(f"  [FAIL] {name}: finished={finished} results={len(task.results)} ({el:.0f}s)")
        return None, el, vr
    img = task.results[0]
    arr = np.asarray(Image.open(img)) if isinstance(img, str) else np.asarray(img).astype("uint8")
    out = f"/tmp/quality_{name}.png"
    Image.fromarray(arr).save(out)
    print(f"  [OK] {name}: {el:.1f}s  VRAM={vr:.2f}GB  res={arr.shape[1]}x{arr.shape[0]} -> {out}")
    return arr, el, vr


ensure(JUG, "https://huggingface.co/lllyasviel/fav_models/resolve/main/fav/juggernautXL_v8Rundiffusion.safetensors")
ensure(REAL, "https://huggingface.co/lllyasviel/fav_models/resolve/main/fav/realisticStockPhoto_v20.safetensors")

results = {}
print("\n# 1 realistic, Quality 60 (woman)")
a1, t1, v1 = run(make_task(PROMPT_W, REAL), "1_realistic_q60_woman"); results["1 realistic Q60 (woman)"] = (60, "896x1152", t1, v1)
print("# 2 juggernaut, Quality 60 (woman) [model comparison]")
_, t2, v2 = run(make_task(PROMPT_W, JUG), "2_juggernaut_q60_woman"); results["2 juggernaut Q60 (woman)"] = (60, "896x1152", t2, v2)
print("# 3 realistic, Quality 60 + strong negative (woman, tall)")
_, t3, v3 = run(make_task(PROMPT_W, REAL, aspect="832*1216", negative=NEG_STRONG), "3_realistic_q60_strongneg_woman"); results["3 realistic Q60 +neg (woman)"] = (60, "832x1216", t3, v3)
print("# 4 realistic, 100 steps (woman) [max steps]")
_, t4, v4 = run(make_task(PROMPT_W, REAL, overwrite_step=100), "4_realistic_q100_woman"); results["4 realistic 100-step (woman)"] = (100, "896x1152", t4, v4)
print("# 5 realistic, upscale 2x of #1 (high-res finisher)")
if a1 is not None:
    _, t5, v5 = run(make_task(PROMPT_W, REAL, uov_method=upscale_2, uov_input_image=a1), "5_realistic_upscale2x_woman")
    results["5 realistic upscale2x (woman)"] = ("uov", "~1792x2304", t5, v5)
print("# 6 realistic, Quality 60 (man) [subject range]")
_, t6, v6 = run(make_task(PROMPT_M, REAL, aspect="832*1216"), "6_realistic_q60_man"); results["6 realistic Q60 (man)"] = (60, "832x1216", t6, v6)

print("\n================ QUALITY (PHOTOREAL PERSON) COMPARISON ================")
print(f"| {'Case':32s} | {'Steps':6s} | {'Resolution':11s} | {'Time':7s} | {'VRAM':8s} |")
print("|" + "-"*34 + "|" + "-"*8 + "|" + "-"*13 + "|" + "-"*9 + "|" + "-"*10 + "|")
for k, (steps, res, el, vr) in results.items():
    print(f"| {k:32s} | {str(steps):6s} | {res:11s} | {el:5.1f}s | {vr:5.2f}GB |")
print("\n[IMAGES]", *[f"/tmp/quality_{n}.png" for n in
      ["1_realistic_q60_woman","2_juggernaut_q60_woman","3_realistic_q60_strongneg_woman",
       "4_realistic_q100_woman","5_realistic_upscale2x_woman","6_realistic_q60_man"]])
print("[OK] quality benchmark complete")
