Below are many common problems that people encountered:

### RuntimeError: CPUAllocator

See also the section: **System Swap**

### Model loaded, then paused, then nothing happens

See also the section: **System Swap**

### Segmentation Fault

See also the section: **System Swap**

### Aborted

See also the section: **System Swap**

### core dumped

See also the section: **System Swap**

### Killed

See also the section: **System Swap**

### ^C, then quit

See also the section: **System Swap**

### adm 2816, then stuck

See also the section: **System Swap**

### Connection errored out

See also the section: **System Swap**

### Error 1006

See also the section: **System Swap**

### WinError 10060

See also the section: **System Swap**

### Read timed out

See also the section: **System Swap**

### No error, but the console close in a flash. Cannot find any error.

See also the section: **System Swap**

### Model loading is extremely slow (more than 1 minute)

See also the section: **System Swap**

### System Swap

All above problems are caused by the fact that you do not have enough System Swap.

Please make sure that you have at least 40GB System Swap. In fact, it does not need so much Swap, but 40Gb should be safe for you to run Fooocus in 100% success.

(If you have more than 64GB RAM, then *perhaps* you do not need any System Swap, but we are not exactly sure about this.)

Also, if your system swap is on HDD, the speed of model loading will be very slow. Please try best to put system swap on SSD.

If you are using Linux/Mac, please follow your provider's instructions to set Swap Space. Herein, the "provider" refers to Ubuntu official, CentOS official, Mac official, etc.

On Linux, you can add a swapfile (example: 40GB):

    sudo fallocate -l 40G /swapfile
    sudo chmod 600 /swapfile
    sudo mkswap /swapfile
    sudo swapon /swapfile
    # make it permanent by appending to /etc/fstab:  /swapfile none swap sw 0 0

Prefer placing the swapfile on an SSD/NVMe drive so model loading stays fast.

On macOS, the system manages swap automatically; just keep plenty of free disk space.

### MetadataIncompleteBuffer

See also the section: **Model corrupted**

### PytorchStreamReader failed

See also the section: **Model corrupted**

### Model corrupted

If you see Model Corrupted, then your model is corrupted. Fooocus will re-download corrupted models for you if your internet connection is good. Otherwise, you may also manually download models. You can find model url and their local location in the console each time a model download is requested.

### UserWarning: The operator 'aten::std_mean.correction' is not currently supported on the DML

This is a warning that you can ignore.

### Torch not compiled with CUDA enabled

You are not following the official installation guide. 

Please do not trust those wrong tutorials on the internet, and please only trust the official installation guide. 

### subprocess-exited-with-error

Please use python 3.10

Also, you are not following the official installation guide. 

Please do not trust those wrong tutorials on the internet, and please only trust the official installation guide. 

### SSL: CERTIFICATE_VERIFY_FAILED

Are you living in China? If yes, please consider turn off VPN, and/or try to download models manually.

If you get this error elsewhere in the world, then you may need to look at [this search](https://www.google.com/search?q=SSL+Certificate+Error). We cannot give very specific guide to fix this since the cause can vary a lot.

### CUDA kernel errors might be asynchronously reported at some other API call

A very small amount of devices does have this problem. The cause can be complicated but usually can be resolved after following these steps:

1. Make sure that you are using official version and latest version installed from [here](https://github.com/lllyasviel/Fooocus#install). (Some forks and other versions are more likely to cause this problem.)
2. Upgrade your Nvidia driver to the latest version. (Usually the version of your Nvidia driver should be 53X, not 3XX or 4XX.)
3. If things still do not work, it may be a mismatch between the bundled CUDA wheels and your driver. Edit the `pytorch-cu128` index URL in `pyproject.toml` to a CUDA version matching your driver (e.g. `cu126`), then run `uv lock` followed by `uv sync`. As a last resort you can also add xformers with `uv add xformers`.
4. If it still does not work, please open an issue for us to take a look.

### Found no NVIDIA driver on your system

Please upgrade your Nvidia Driver. 

If you are using AMD, please follow official installation guide.

### NVIDIA driver too old

Please upgrade your Nvidia Driver.

### I am using Mac, the speed is very slow.

Some MAC users may need `--disable-offload-from-vram` to speed up model loading.

Besides, the current support for MAC is very experimental, and we encourage users to also try Diffusionbee or Drawingthings: they are developed only for MAC.

### I am using Nvidia with 8GB VRAM, I get CUDA Out Of Memory

It is a BUG. Please let us know as soon as possible. Please make an issue. See also [minimal requirements](https://github.com/lllyasviel/Fooocus/tree/main?tab=readme-ov-file#minimal-requirement).

### I am using Nvidia with 6GB VRAM, I get CUDA Out Of Memory

It is very likely a BUG. Please let us know as soon as possible. Please make an issue. See also [minimal requirements](https://github.com/lllyasviel/Fooocus/tree/main?tab=readme-ov-file#minimal-requirement).

### I am using Nvidia with 4GB VRAM with Float16 support, like RTX 3050, I get CUDA Out Of Memory

It is a BUG. Please let us know as soon as possible. Please make an issue. See also [minimal requirements](https://github.com/lllyasviel/Fooocus/tree/main?tab=readme-ov-file#minimal-requirement).

### I am using Nvidia with 4GB VRAM without Float16 support, like GTX 960, I get CUDA Out Of Memory

Supporting GPU with 4GB VRAM without fp16 is extremely difficult, and you may not be able to use SDXL. However, you may still make an issue and let us know. You may try SD1.5 in Automatic1111 or other software for your device. See also [minimal requirements](https://github.com/lllyasviel/Fooocus/tree/main?tab=readme-ov-file#minimal-requirement).

### I am using AMD GPU on Linux, I get CUDA Out Of Memory

AMD support on Linux uses ROCm and is still experimental. Make sure you installed the ROCm PyTorch wheels (see the "Linux — AMD / ROCm" section in the readme). If you are able to run SDXL on this same device with other software, please let us know and we will try to support it. See also [minimal requirements](https://github.com/lllyasviel/Fooocus/tree/main?tab=readme-ov-file#minimal-requirement).

### I tried flags like --lowvram or --gpu-only or --bf16 or so on, and things are not getting any better?

Please remove these flags if you are mislead by some wrong tutorials. In most cases these flags are making things worse and introducing more problems.

### Fooocus suddenly becomes very slow and I have not changed anything

Are you accidentally running two Fooocus at the same time?
