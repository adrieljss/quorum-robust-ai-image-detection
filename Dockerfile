# Hugging Face Space, SDK "docker". Serves the Flask demo on 7860.
#
# The whole trained model is 272 KB of .npz plus a 233 KB face detector.
# Everything heavy here is CLIP, downloaded ONCE at build time (see the bake
# step) so a cold container starts from local disk instead of pulling 1.71 GB.
#
# NOTE what is deliberately NOT in this image: data/cache, data/.cache and
# data/raw. Those hold SID_Set, So-Fake-OOD and the CC BY-NC face corpus, whose
# licences do not permit redistribution, and a Space is PUBLIC. .dockerignore
# enforces it with an ALLOWLIST; do not turn that back into a blocklist.
FROM python:3.11-slim

# libglib2.0 is for opencv, libgomp is torch's OpenMP runtime.
#
# The X11 libs are NOT gratuitous: rapidocr_onnxruntime depends on the full
# opencv-python (not headless), which links libGL and libxcb even when nothing
# is ever drawn. Without them the OCR bake dies on "ImportError: libxcb.so.1"
# and signals.text is null in production while the build still passes.
# Measured -- it is what this file did.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 libgomp1 libgl1 libxcb1 libsm6 libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Spaces runs containers as uid 1000. Everything the app writes -- the model
# cache above all -- has to be owned by that user or the first request fails on
# a read-only HOME.
RUN useradd -m -u 1000 user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    HF_HOME=/home/user/.cache/huggingface \
    PYTHONUNBUFFERED=1
USER user
WORKDIR /home/user/app

# CPU-only torch, explicitly: the default index gives a CUDA build, ~2.5 GB of
# wheels that cannot run on a free Space.
#
# torchvision MUST come from the SAME index in the SAME command. open_clip_torch
# depends on it, so installing torch alone lets pip satisfy that dependency with
# a PyPI torchvision built against a different torch. It then imports far enough
# to look fine and dies with "operator torchvision::nms does not exist", so
# open_clip never loads and the demo is dead. Measured, not hypothetical -- it is
# what this file did on its first build.
COPY --chown=user requirements-space.txt .
RUN pip install --no-cache-dir --user \
        torch torchvision --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir --user -r requirements-space.txt \
 && python -c "import torch, torchvision, open_clip; print('torch', torch.__version__, 'torchvision', torchvision.__version__, 'open_clip ok')"

COPY --chown=user . .

# Bake the 1.71 GB of CLIP weights into an image LAYER.
#
# NO `|| true` here, deliberately. This step is the difference between a ~10s
# cold start and a 1.71 GB download on the first request, so a failure must fail
# the BUILD. It previously shared an `||` with the optional OCR step below,
# which meant a CLIP failure also exited 0 -- and shipped an image with no
# weights in it, while the build reported success.
RUN python -c "import sys; sys.path.insert(0, '.'); from quorum.embed import Embedder; Embedder(device='cpu'); print('CLIP weights baked')"

# This one IS optional: signals.text is display-only and analyzer.py reports
# null when the OCR engine is missing. On its own line so it cannot mask the
# step above.
RUN python -c "from rapidocr_onnxruntime import RapidOCR; RapidOCR(); print('OCR weights baked')" \
 || echo "!!! OCR bake FAILED -- signals.text will be null in production !!!"

# Prove the weights landed in the LAYER rather than a discarded build cache, and
# that no dataset came along. A bake that silently no-ops, and a .dockerignore
# that silently misses, are the two failures this file has actually had.
RUN test -s "$(find $HF_HOME -name 'open_clip_model.safetensors' | head -1)" \
 && echo "cache: $(du -sh $HF_HOME | cut -f1)" \
 && test "$(du -sb data | cut -f1)" -lt 2000000 \
 && echo "data: $(du -sh data | cut -f1) -- no dataset in the image"

EXPOSE 7860

# ONE worker, on purpose. Each loads its own copy of CLIP (~2.4 GB resident), so
# a second doubles memory for no throughput on a 2-vCPU box. Threads handle
# concurrent uploads; torch releases the GIL.
#
# --timeout 300 because a CPU request is ~10s and can be more on a cold cache;
# gunicorn's 30s default would kill exactly the requests we care about.
# --preload loads the app and its models BEFORE forking, so container start pays
# the ~9.5s model load rather than the first visitor.
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "--workers", "1", "--threads", "4", \
     "--timeout", "300", "--preload", "--chdir", "app", "app:app"]
