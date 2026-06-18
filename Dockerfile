# Fooocus container image — production target: Linux + NVIDIA (CUDA).
# Dependencies are managed by the uv package manager (pyproject.toml + uv.lock).
# PyTorch is installed from the cu128 wheels (see [tool.uv.index] in pyproject.toml).
# If your NVIDIA driver needs a different CUDA version, change BOTH the base image tag
# below and the pytorch-cu128 index URL in pyproject.toml (e.g. cu126 / cu129), then re-lock.
FROM nvidia/cuda:12.8.1-base-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    CMDARGS=--listen \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_PYTHON_INSTALL_DIR=/opt/uv-python \
    PATH="/opt/venv/bin:$PATH"

# System packages:
#   build-essential -> compiles groundingdino-py's C++ extension during `uv sync`
#   libgl1 / libglib2.0-0 -> required by opencv at runtime
RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
        ca-certificates curl git build-essential libgl1 libglib2.0-0 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Pinned uv binary for reproducible builds.
COPY --from=ghcr.io/astral-sh/uv:0.11.3 /uv /uvx /bin/

RUN adduser --disabled-password --gecos '' user && \
    mkdir -p /content/app /content/data /opt/venv /opt/uv-python && \
    chown -R user:user /content /opt/venv /opt/uv-python

WORKDIR /content/app
USER user

# 1) Install dependencies first so this layer is cached independently of the source.
#    uv provisions the Python interpreter from .python-version automatically.
COPY --chown=user:user pyproject.toml uv.lock .python-version ./
RUN uv sync --locked --no-dev

# 2) Copy the application source.
COPY --chown=user:user . /content/app

# Move bundled model placeholders aside so the entrypoint can symlink a persistent volume.
RUN mv /content/app/models /content/app/models.org

COPY --chown=user:user entrypoint.sh /content/

# entrypoint.sh launches `python launch.py`, which resolves to /opt/venv/bin/python via PATH.
CMD [ "sh", "-c", "/content/entrypoint.sh ${CMDARGS}" ]
