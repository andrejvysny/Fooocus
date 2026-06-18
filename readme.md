<div align=center>
<img src="https://github.com/lllyasviel/Fooocus/assets/19834515/483fb86d-c9a2-4c20-997c-46dafc124f25">
</div>

# Fooocus

[>>> Click Here to Install Fooocus <<<](#install)

Fooocus is an image generating software (based on [Gradio](https://www.gradio.app/) <a href='https://github.com/gradio-app/gradio'><img src='https://img.shields.io/github/stars/gradio-app/gradio'></a>).

Fooocus presents a rethinking of image generator designs. The software is offline, open source, and free, while at the same time, similar to many online image generators like Midjourney, the manual tweaking is not needed, and users only need to focus on the prompts and images. Fooocus has also simplified the installation: between pressing "download" and generating the first image, the number of needed mouse clicks is strictly limited to less than 3. Minimal GPU memory requirement is 4GB (Nvidia).

**Recently many fake websites exist on Google when you search “fooocus”. Do not trust those – here is the only official source of Fooocus.**

# Project Status: Limited Long-Term Support (LTS) with Bug Fixes Only

The Fooocus project, built entirely on the **Stable Diffusion XL** architecture, is now in a state of limited long-term support (LTS) with bug fixes only. As the existing functionalities are considered as nearly free of programmartic issues (Thanks to [mashb1t](https://github.com/mashb1t)'s huge efforts), future updates will focus exclusively on addressing any bugs that may arise. 

**There are no current plans to migrate to or incorporate newer model architectures.** However, this may change during time with the development of open-source community. For example, if the community converge to one single dominant method for image generation (which may really happen in half or one years given the current status), Fooocus may also migrate to that exact method.

For those interested in utilizing newer models such as **Flux**, we recommend exploring alternative platforms such as [WebUI Forge](https://github.com/lllyasviel/stable-diffusion-webui-forge) (also from us), [ComfyUI/SwarmUI](https://github.com/comfyanonymous/ComfyUI). Additionally, several [excellent forks of Fooocus](https://github.com/lllyasviel/Fooocus?tab=readme-ov-file#forks) are available for experimentation.

Again, recently many fake websites exist on Google when you search “fooocus”. Do **NOT** get Fooocus from those websites – this page is the only official source of Fooocus. We never have any website like such as “fooocus.com”, “fooocus.net”, “fooocus.co”, “fooocus.ai”, “fooocus.org”, “fooocus.pro”, “fooocus.one”. Those websites are ALL FAKE. **They have ABSOLUTLY no relationship to us. Fooocus is a 100% non-commercial offline open-source software.**

# Features

Below is a quick list using Midjourney's examples:

| Midjourney | Fooocus |
| - | - |
| High-quality text-to-image without needing much prompt engineering or parameter tuning. <br> (Unknown method) | High-quality text-to-image without needing much prompt engineering or parameter tuning. <br> (Fooocus has an offline GPT-2 based prompt processing engine and lots of sampling improvements so that results are always beautiful, no matter if your prompt is as short as “house in garden” or as long as 1000 words) |
| V1 V2 V3 V4 | Input Image -> Upscale or Variation -> Vary (Subtle) / Vary (Strong)|
| U1 U2 U3 U4 | Input Image -> Upscale or Variation -> Upscale (1.5x) / Upscale (2x) |
| Inpaint / Up / Down / Left / Right (Pan) | Input Image -> Inpaint or Outpaint -> Inpaint / Up / Down / Left / Right <br> (Fooocus uses its own inpaint algorithm and inpaint models so that results are more satisfying than all other software that uses standard SDXL inpaint method/model) |
| Image Prompt | Input Image -> Image Prompt <br> (Fooocus uses its own image prompt algorithm so that result quality and prompt understanding are more satisfying than all other software that uses standard SDXL methods like standard IP-Adapters or Revisions) |
| --style | Advanced -> Style |
| --stylize | Advanced -> Advanced -> Guidance |
| --niji | [Multiple presets: default, anime, and realistic (use `--preset`).](https://github.com/lllyasviel/Fooocus/discussions/679) <br> Fooocus support SDXL models on Civitai <br> (You can google search “Civitai” if you do not know about it) |
| --quality | Advanced -> Quality |
| --repeat | Advanced -> Image Number |
| Multi Prompts (::) | Just use multiple lines of prompts |
| Prompt Weights | You can use " I am (happy:1.5)". <br> Fooocus uses A1111's reweighting algorithm so that results are better than ComfyUI if users directly copy prompts from Civitai. (Because if prompts are written in ComfyUI's reweighting, users are less likely to copy prompt texts as they prefer dragging files) <br> To use embedding, you can use "(embedding:file_name:1.1)" |
| --no | Advanced -> Negative Prompt |
| --ar | Advanced -> Aspect Ratios |
| InsightFace | Input Image -> Image Prompt -> Advanced -> FaceSwap |
| Describe | Input Image -> Describe |

Below is a quick list using LeonardoAI's examples:

| LeonardoAI | Fooocus |
| - | - |
| Prompt Magic | Advanced -> Style -> Fooocus V2 |
| Advanced Sampler Parameters (like Contrast/Sharpness/etc) | Advanced -> Advanced -> Sampling Sharpness / etc |
| User-friendly ControlNets | Input Image -> Image Prompt -> Advanced |

Also, [click here to browse the advanced features.](https://github.com/lllyasviel/Fooocus/discussions/117)

# Install

Fooocus is managed with the [uv](https://docs.astral.sh/uv/) package manager and targets
**Linux with an NVIDIA GPU (CUDA) — the primary/production target** and
**macOS on Apple Silicon (MPS) — for development and experiments**. Windows is not supported.

### Quick start (Linux & macOS)

Install uv once:

    curl -LsSf https://astral.sh/uv/install.sh | sh

Then clone and launch:

    git clone https://github.com/lllyasviel/Fooocus.git
    cd Fooocus
    ./run.sh

`./run.sh` runs `uv run`, which automatically creates the virtual environment from
`pyproject.toml` + `uv.lock` (installing the correct PyTorch build for your platform) and starts
Fooocus. The first launch also downloads the [default models](#models).

Useful equivalents:

    uv sync                                  # just create/update the environment from the lockfile
    uv run python entry_with_update.py       # launch with git auto-update (what run.sh does)
    uv run python launch.py                  # launch without auto-update
    uv run python launch.py --listen         # expose on your local network
    uv run python launch.py --preset anime   # Anime edition (or: --preset realistic)

Regenerate the lockfile after changing dependencies with `uv lock`; verify it in CI with `uv sync --locked`.

### Linux — NVIDIA / CUDA (primary target)

No extra steps: on Linux, `pyproject.toml` installs CUDA (cu128) PyTorch wheels via the
`pytorch-cu128` index. If your NVIDIA driver requires a different CUDA toolkit, edit that index URL
in `pyproject.toml` (e.g. `cu126` or `cu129`) and run `uv lock` before `uv sync`.

### Linux — AMD / ROCm

Replace the PyTorch index in `pyproject.toml` with a ROCm index, then re-lock:

    [[tool.uv.index]]
    name = "pytorch-rocm"
    url = "https://download.pytorch.org/whl/rocm6.2"
    explicit = true

    [tool.uv.sources]
    torch = [{ index = "pytorch-rocm", marker = "sys_platform == 'linux'" }]
    torchvision = [{ index = "pytorch-rocm", marker = "sys_platform == 'linux'" }]

Then `uv lock && ./run.sh`. ROCm is roughly 1.5x slower than an equivalent NVIDIA RTX 3XXX and is less tested.

### macOS — Apple Silicon (development)

    git clone https://github.com/lllyasviel/Fooocus.git
    cd Fooocus
    ./run.sh

On macOS, uv installs the standard PyPI PyTorch arm64 wheels, which already include the **MPS**
(Metal) backend — there is no conda step and no PyTorch-nightly step. Some Macs load models faster
with `uv run python launch.py --disable-offload-from-vram`.

Apple Silicon is considerably slower than a discrete NVIDIA GPU (≈9x slower than an RTX 3XXX), so
use the Mac for development and a Linux/NVIDIA machine for heavy generation. To experiment with
MLX/FLUX-class models on Mac, see [`experiments/`](experiments/) and
[`docs/backend-strategy.md`](docs/backend-strategy.md).

### Colab

| Colab | Info
| --- | --- |
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lllyasviel/Fooocus/blob/main/fooocus_colab.ipynb) | Fooocus Official

The notebook's launch cell uses `!uv run python entry_with_update.py --share --always-high-vram`
(optionally add `--preset anime` or `--preset realistic`). `--always-high-vram` gives the best
balance on the default T4 instance.

### Docker

See [docker.md](docker.md)

### Download Previous Version

See the guidelines [here](https://github.com/lllyasviel/Fooocus/discussions/1405).

## Minimal Requirement

Below is the minimal requirement for running Fooocus locally. If your device capability is lower than this spec, you may not be able to use Fooocus locally. (Please let us know, in any case, if your device capability is lower but Fooocus still works.)

| Operating System  | GPU                          | Minimal GPU Memory           | Minimal System Memory     | [System Swap](troubleshoot.md) | Note                                                                       |
|-------------------|------------------------------|------------------------------|---------------------------|--------------------------------|----------------------------------------------------------------------------|
| Linux             | Nvidia RTX 4XXX              | 4GB                          | 8GB                       | Required                       | fastest                                                                    |
| Linux             | Nvidia RTX 3XXX              | 4GB                          | 8GB                       | Required                       | usually faster than RTX 2XXX                                               |
| Linux             | Nvidia RTX 2XXX              | 4GB                          | 8GB                       | Required                       | usually faster than GTX 1XXX                                               |
| Linux             | Nvidia GTX 1XXX              | 8GB (&ast; 6GB uncertain)    | 8GB                       | Required                       | only marginally faster than CPU                                            |
| Linux             | Nvidia GTX 9XX               | 8GB                          | 8GB                       | Required                       | faster or slower than CPU                                                  |
| Linux             | Nvidia GTX < 9XX             | Not supported                | /                         | /                              | /                                                                          |
| Linux             | AMD GPU (ROCm)               | 8GB                          | 8GB                       | Required                       | via ROCm, about 1.5x slower than Nvidia RTX 3XXX                           |
| Mac               | Apple Silicon (M-series) MPS | Shared                       | Shared                    | Shared                         | about 9x slower than Nvidia RTX 3XXX                                       |
| Linux/Mac         | only use CPU                 | 0GB                          | 32GB                      | Required                       | about 17x slower than Nvidia RTX 3XXX                                      |

&ast; Nvidia GTX 1XXX 6GB uncertain: Some people report 6GB success on GTX 10XX, but some other people report failure cases.

*Note that Fooocus is only for extremely high quality image generating. We will not support smaller models to reduce the requirement and sacrifice result quality.*

## Troubleshoot

See the common problems [here](troubleshoot.md).

## Default Models
<a name="models"></a>

Given different goals, the default models and configs of Fooocus are different:

| Task      | Launch args | Main Model                  | Refiner | Config                                                                         |
|-----------| --- |-----------------------------| --- |--------------------------------------------------------------------------------|
| General   | (default) | juggernautXL_v8Rundiffusion | not used | [here](https://github.com/lllyasviel/Fooocus/blob/main/presets/default.json)   |
| Realistic | --preset realistic | realisticStockPhoto_v20     | not used | [here](https://github.com/lllyasviel/Fooocus/blob/main/presets/realistic.json) |
| Anime     | --preset anime | animaPencilXL_v500          | not used | [here](https://github.com/lllyasviel/Fooocus/blob/main/presets/anime.json)     |

Note that the download is **automatic** - you do not need to do anything if the internet connection is okay. However, you can download them manually if you (or move them from somewhere else) have your own preparation.

## UI Access and Authentication
In addition to running on localhost, Fooocus can also expose its UI in two ways: 
* Local UI listener: use `--listen` (specify port e.g. with `--port 8888`). 
* API access: use `--share` (registers an endpoint at `.gradio.live`).

In both ways the access is unauthenticated by default. You can add basic authentication by creating a file called `auth.json` in the main directory, which contains a list of JSON objects with the keys `user` and `pass` (see example in [auth-example.json](./auth-example.json)).

## List of "Hidden" Tricks
<a name="tech_list"></a>

<details>
<summary>Click to see a list of tricks. Those are based on SDXL and are not very up-to-date with latest models.</summary>

1. GPT2-based [prompt expansion as a dynamic style "Fooocus V2".](https://github.com/lllyasviel/Fooocus/discussions/117#raw) (similar to Midjourney's hidden pre-processing and "raw" mode, or the LeonardoAI's Prompt Magic).
2. Native refiner swap inside one single k-sampler. The advantage is that the refiner model can now reuse the base model's momentum (or ODE's history parameters) collected from k-sampling to achieve more coherent sampling. In Automatic1111's high-res fix and ComfyUI's node system, the base model and refiner use two independent k-samplers, which means the momentum is largely wasted, and the sampling continuity is broken. Fooocus uses its own advanced k-diffusion sampling that ensures seamless, native, and continuous swap in a refiner setup. (Update Aug 13: Actually, I discussed this with Automatic1111 several days ago, and it seems that the “native refiner swap inside one single k-sampler” is [merged]( https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/12371) into the dev branch of webui. Great!)
3. Negative ADM guidance. Because the highest resolution level of XL Base does not have cross attentions, the positive and negative signals for XL's highest resolution level cannot receive enough contrasts during the CFG sampling, causing the results to look a bit plastic or overly smooth in certain cases. Fortunately, since the XL's highest resolution level is still conditioned on image aspect ratios (ADM), we can modify the adm on the positive/negative side to compensate for the lack of CFG contrast in the highest resolution level. (Update Aug 16, the IOS App [Draw Things](https://apps.apple.com/us/app/draw-things-ai-generation/id6444050820) will support Negative ADM Guidance. Great!)
4. We implemented a carefully tuned variation of Section 5.1 of ["Improving Sample Quality of Diffusion Models Using Self-Attention Guidance"](https://arxiv.org/pdf/2210.00939.pdf). The weight is set to very low, but this is Fooocus's final guarantee to make sure that the XL will never yield an overly smooth or plastic appearance (examples [here](https://github.com/lllyasviel/Fooocus/discussions/117#sharpness)). This can almost eliminate all cases for which XL still occasionally produces overly smooth results, even with negative ADM guidance. (Update 2023 Aug 18, the Gaussian kernel of SAG is changed to an anisotropic kernel for better structure preservation and fewer artifacts.)
5. We modified the style templates a bit and added the "cinematic-default".
6. We tested the "sd_xl_offset_example-lora_1.0.safetensors" and it seems that when the lora weight is below 0.5, the results are always better than XL without lora.
7. The parameters of samplers are carefully tuned.
8. Because XL uses positional encoding for generation resolution, images generated by several fixed resolutions look a bit better than those from arbitrary resolutions (because the positional encoding is not very good at handling int numbers that are unseen during training). This suggests that the resolutions in UI may be hard coded for best results.
9. Separated prompts for two different text encoders seem unnecessary. Separated prompts for the base model and refiner may work, but the effects are random, and we refrain from implementing this.
10. The DPM family seems well-suited for XL since XL sometimes generates overly smooth texture, but the DPM family sometimes generates overly dense detail in texture. Their joint effect looks neutral and appealing to human perception.
11. A carefully designed system for balancing multiple styles as well as prompt expansion.
12. Using automatic1111's method to normalize prompt emphasizing. This significantly improves results when users directly copy prompts from civitai.
13. The joint swap system of the refiner now also supports img2img and upscale in a seamless way.
14. CFG Scale and TSNR correction (tuned for SDXL) when CFG is bigger than 10.
</details>

## Customization

After the first time you run Fooocus, a config file will be generated at `Fooocus\config.txt`. This file can be edited to change the model path or default parameters.

For example, an edited `Fooocus\config.txt` (this file will be generated after the first launch) may look like this:

```json
{
    "path_checkpoints": "D:\\Fooocus\\models\\checkpoints",
    "path_loras": "D:\\Fooocus\\models\\loras",
    "path_embeddings": "D:\\Fooocus\\models\\embeddings",
    "path_vae_approx": "D:\\Fooocus\\models\\vae_approx",
    "path_upscale_models": "D:\\Fooocus\\models\\upscale_models",
    "path_inpaint": "D:\\Fooocus\\models\\inpaint",
    "path_controlnet": "D:\\Fooocus\\models\\controlnet",
    "path_clip_vision": "D:\\Fooocus\\models\\clip_vision",
    "path_fooocus_expansion": "D:\\Fooocus\\models\\prompt_expansion\\fooocus_expansion",
    "path_outputs": "D:\\Fooocus\\outputs",
    "default_model": "realisticStockPhoto_v10.safetensors",
    "default_refiner": "",
    "default_loras": [["lora_filename_1.safetensors", 0.5], ["lora_filename_2.safetensors", 0.5]],
    "default_cfg_scale": 3.0,
    "default_sampler": "dpmpp_2m",
    "default_scheduler": "karras",
    "default_negative_prompt": "low quality",
    "default_positive_prompt": "",
    "default_styles": [
        "Fooocus V2",
        "Fooocus Photograph",
        "Fooocus Negative"
    ]
}
```

Many other keys, formats, and examples are in `Fooocus\config_modification_tutorial.txt` (this file will be generated after the first launch).

Consider twice before you really change the config. If you find yourself breaking things, just delete `Fooocus\config.txt`. Fooocus will go back to default.

A safer way is just to try `--preset anime` or `--preset realistic` - they should already be good enough for different tasks.

~Note that `user_path_config.txt` is deprecated and will be removed soon.~ (Edit: it is already removed.)

### All CMD Flags

```
entry_with_update.py  [-h] [--listen [IP]] [--port PORT]
                      [--disable-header-check [ORIGIN]]
                      [--web-upload-size WEB_UPLOAD_SIZE]
                      [--hf-mirror HF_MIRROR]
                      [--external-working-path PATH [PATH ...]]
                      [--output-path OUTPUT_PATH]
                      [--temp-path TEMP_PATH] [--cache-path CACHE_PATH]
                      [--in-browser] [--disable-in-browser]
                      [--gpu-device-id DEVICE_ID]
                      [--async-cuda-allocation | --disable-async-cuda-allocation]
                      [--disable-attention-upcast]
                      [--all-in-fp32 | --all-in-fp16]
                      [--unet-in-bf16 | --unet-in-fp16 | --unet-in-fp8-e4m3fn | --unet-in-fp8-e5m2]
                      [--vae-in-fp16 | --vae-in-fp32 | --vae-in-bf16]
                      [--vae-in-cpu]
                      [--clip-in-fp8-e4m3fn | --clip-in-fp8-e5m2 | --clip-in-fp16 | --clip-in-fp32]
                      [--disable-ipex-hijack]
                      [--preview-option [none,auto,fast,taesd]]
                      [--attention-split | --attention-quad | --attention-pytorch]
                      [--disable-xformers]
                      [--always-gpu | --always-high-vram | --always-normal-vram | --always-low-vram | --always-no-vram | --always-cpu [CPU_NUM_THREADS]]
                      [--always-offload-from-vram]
                      [--pytorch-deterministic] [--disable-server-log]
                      [--debug-mode]
                      [--disable-server-info] [--multi-user] [--share]
                      [--preset PRESET] [--disable-preset-selection]
                      [--language LANGUAGE]
                      [--disable-offload-from-vram] [--theme THEME]
                      [--disable-image-log] [--disable-analytics]
                      [--disable-metadata] [--disable-preset-download]
                      [--disable-enhance-output-sorting]
                      [--enable-auto-describe-image]
                      [--always-download-new-model]
                      [--rebuild-hash-cache [CPU_NUM_THREADS]]
```

## Inline Prompt Features

### Wildcards

Example prompt: `__color__ flower`

Processed for positive and negative prompt.

Selects a random wildcard from a predefined list of options, in this case the `wildcards/color.txt` file. 
The wildcard will be replaced with a random color (randomness based on seed). 
You can also disable randomness and process a wildcard file from top to bottom by enabling the checkbox `Read wildcards in order` in Developer Debug Mode.

Wildcards can be nested and combined, and multiple wildcards can be used in the same prompt (example see `wildcards/color_flower.txt`).

### Array Processing

Example prompt: `[[red, green, blue]] flower`

Processed only for positive prompt.

Processes the array from left to right, generating a separate image for each element in the array. In this case 3 images would be generated, one for each color.
Increase the image number to 3 to generate all 3 variants.

Arrays can not be nested, but multiple arrays can be used in the same prompt.
Does support inline LoRAs as array elements!

### Inline LoRAs

Example prompt: `flower <lora:sunflowers:1.2>`

Processed only for positive prompt.

Applies a LoRA to the prompt. The LoRA file must be located in the `models/loras` directory.

## Advanced Features

[Click here to browse the advanced features.](https://github.com/lllyasviel/Fooocus/discussions/117)

## Forks

Below are some Forks to Fooocus:

| Fooocus' forks |
| - |
| [fenneishi/Fooocus-Control](https://github.com/fenneishi/Fooocus-Control) </br>[runew0lf/RuinedFooocus](https://github.com/runew0lf/RuinedFooocus) </br> [MoonRide303/Fooocus-MRE](https://github.com/MoonRide303/Fooocus-MRE) </br> [mashb1t/Fooocus](https://github.com/mashb1t/Fooocus) </br> and so on ... |

## Thanks

Many thanks to [twri](https://github.com/twri) and [3Diva](https://github.com/3Diva) and [Marc K3nt3L](https://github.com/K3nt3L) for creating additional SDXL styles available in Fooocus. 

The project starts from a mixture of [Stable Diffusion WebUI](https://github.com/AUTOMATIC1111/stable-diffusion-webui) and [ComfyUI](https://github.com/comfyanonymous/ComfyUI) codebases.

Also, thanks [daswer123](https://github.com/daswer123) for contributing the Canvas Zoom!

## Update Log

The log is [here](update_log.md).

## Localization/Translation/I18N

You can put json files in the `language` folder to translate the user interface.

For example, below is the content of `Fooocus/language/example.json`:

```json
{
  "Generate": "生成",
  "Input Image": "入力画像",
  "Advanced": "고급",
  "SAI 3D Model": "SAI 3D Modèle"
}
```

If you add `--language example` arg, Fooocus will read `Fooocus/language/example.json` to translate the UI.

For example, launch Fooocus with:

    uv run python entry_with_update.py --language example

Or together with the Anime / Realistic presets:

    uv run python entry_with_update.py --language example --preset anime
    uv run python entry_with_update.py --language example --preset realistic

For practical translation, you may create your own file like `Fooocus/language/jp.json` or `Fooocus/language/cn.json` and then use flag `--language jp` or `--language cn`. Apparently, these files do not exist now. **We need your help to create these files!**

Note that if no `--language` is given and at the same time `Fooocus/language/default.json` exists, Fooocus will always load `Fooocus/language/default.json` for translation. By default, the file `Fooocus/language/default.json` does not exist.
