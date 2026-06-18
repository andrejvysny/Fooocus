"""Full-body / full-length PERSON generation benchmark on CUDA.

Full poses (head-to-toe), not just faces: explicit full-body prompts, tall aspect ratios,
and anti-crop negatives. Includes a 2x upscale to recover small-face detail at full body.

Run:  uv run python -u _fullbody_bench.py
Temporary file — delete after use.
"""
import os, sys, ssl, time
sys.argv = ["_fullbody_bench.py"]
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
from modules.flags import Performance, MetadataScheme, disabled, ip_list, enhancement_uov_before, upscale_2
import ldm_patched.modules.model_management as mm

DEV = mm.get_torch_device()
print(f"[env] torch {torch.__version__} dev={DEV} gpu={torch.cuda.get_device_name(0)}")

REAL = "realisticStockPhoto_v20.safetensors"
JUG = "juggernautXL_v8Rundiffusion.safetensors"
ANIME = "animaPencilXL_v500.safetensors"
PHOTO_STYLES = ['Fooocus V2', 'Fooocus Photograph', 'Fooocus Negative']
ANIME_STYLES = ['Fooocus V2', 'Fooocus Semi Realistic', 'Fooocus Masterpiece']
NEG_FULL = ("close-up, headshot, face portrait, cropped, out of frame, cut off, "
            "deformed hands, extra fingers, extra limbs, missing limbs, bad anatomy, "
            "mutated, disfigured, blurry, low quality, watermark, text")
SEED = 31415

W_FULL = ("full body photograph of a young woman in a flowing summer dress, standing in a sunlit park, "
          "full length shot from head to toe, entire body visible, natural light, 35mm, sharp focus, "
          "photorealistic, highly detailed")
M_FULL = ("full body photograph of a man in a tailored navy three-piece suit and leather shoes, standing "
          "on a city sidewalk, full length shot from head to toe, entire body visible, overcast daylight, "
          "35mm, photorealistic, highly detailed")
ATH = ("full body action photograph of a female athlete in running gear sprinting on an outdoor track, "
       "dynamic motion, full length, entire body visible, sports photography, sharp, photorealistic")
ANIME_FULL = ("full body anime illustration of a girl with long blue hair wearing a school uniform, "
              "standing pose, full length, entire body visible from head to shoes, detailed background, "
              "masterpiece, best quality")


def make_task(prompt, base_model, styles, *, aspect="832*1216", negative=NEG_FULL,
              uov_method=disabled, uov_input_image=None, seed=SEED):
    t = worker.AsyncTask(args=[]); t.args = [None]
    t.generate_image_grid = False
    t.prompt = prompt; t.negative_prompt = negative
    t.style_selections = list(styles)
    t.performance_selection = Performance("Quality")
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
        print(f"  [FAIL] {name}: finished={finished} results={len(task.results)} ({el:.0f}s)"); return None, el, vr
    img = task.results[0]
    arr = np.asarray(Image.open(img)) if isinstance(img, str) else np.asarray(img).astype("uint8")
    out = f"/tmp/fullbody_{name}.png"; Image.fromarray(arr).save(out)
    print(f"  [OK] {name}: {el:.1f}s VRAM={vr:.2f}GB res={arr.shape[1]}x{arr.shape[0]} -> {out}")
    return arr, el, vr


results = {}
print("# 1 realistic full-body woman (832x1216)")
a1, t1, v1 = run(make_task(W_FULL, REAL, PHOTO_STYLES), "1_realistic_woman"); results["1 realistic woman (832x1216)"] = (t1, v1)
print("# 2 realistic full-body man (832x1216)")
_, t2, v2 = run(make_task(M_FULL, REAL, PHOTO_STYLES), "2_realistic_man"); results["2 realistic man (832x1216)"] = (t2, v2)
print("# 3 juggernaut full-body woman (832x1216) [model compare]")
_, t3, v3 = run(make_task(W_FULL, JUG, PHOTO_STYLES), "3_juggernaut_woman"); results["3 juggernaut woman (832x1216)"] = (t3, v3)
print("# 4 juggernaut full-body athlete, dynamic pose (1216x832 wide)")
_, t4, v4 = run(make_task(ATH, JUG, PHOTO_STYLES, aspect="1216*832"), "4_juggernaut_athlete"); results["4 juggernaut athlete (1216x832)"] = (t4, v4)
print("# 5 realistic full-body woman -> 2x upscale (recover face detail)")
if a1 is not None:
    _, t5, v5 = run(make_task(W_FULL, REAL, PHOTO_STYLES, uov_method=upscale_2, uov_input_image=a1), "5_realistic_woman_upscale2x")
    results["5 realistic woman upscale2x"] = (t5, v5)
print("# 6 anime full-body girl (832x1216) [range]")
_, t6, v6 = run(make_task(ANIME_FULL, ANIME, ANIME_STYLES), "6_anime_girl"); results["6 anime girl (832x1216)"] = (t6, v6)

print("\n================ FULL-BODY PERSON COMPARISON (Quality/60) ================")
print(f"| {'Case':34s} | {'Time':7s} | {'VRAM':8s} |")
print("|" + "-"*36 + "|" + "-"*9 + "|" + "-"*10 + "|")
for k, (el, vr) in results.items():
    print(f"| {k:34s} | {el:5.1f}s | {vr:5.2f}GB |")
print("\n[IMAGES]", *[f"/tmp/fullbody_{n}.png" for n in
      ["1_realistic_woman","2_realistic_man","3_juggernaut_woman","4_juggernaut_athlete",
       "5_realistic_woman_upscale2x","6_anime_girl"]])
print("[OK] full-body benchmark complete")
