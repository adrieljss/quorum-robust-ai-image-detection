# Hugging Face Space, SDK "docker". Serves the Flask demo on 7860.
#
# The whole trained model is 305 KB of .npz. Everything heavy here is CLIP,
# which is downloaded ONCE at build time (see the bake step below) so a cold
# container starts from local disk instead of pulling 1.71 GB.
#
# NOTE what is deliberately NOT in this image: data/cache and data/raw. Those
# hold SID_Set, So-Fake-OOD and the CC BY-NC face corpus, whose licences do not
# permit redistribution, and a Space is PUBLIC. .dockerignore enforces it; do
# not "fix" a missing-file error by loosening that file.
FROM python:3.11-slim

# opencv-python-headless still needs libglib2.0; the rest of the GUI stack it
# does not. libgomp is torch's OpenMP runtime.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 libgomp1 \
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

# CPU-only torch, explicitly. The default index gives a CUDA build: ~2.5 GB of
# wheels that cannot be used on a free Space and push the image past its limit.
COPY --chown=user requirements-space.txt .
RUN pip install --no-cache-dir --user \
        torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir --user -r requirements-space.txt

COPY --chown=user . .

# Bake the 1.71 GB of CLIP weights into an image LAYER. Without this the first
# request on every cold container downloads them, which is the difference
# between a judge waiting ~10s and waiting for a 1.71 GB transfer.
# RapidOCR's ~10M ONNX models come down the same way, for signals.text.
RUN python -c "\
import sys; sys.path.insert(0, '.');\
from quorum.embed import Embedder; Embedder(device='cpu');\
print('CLIP weights baked')" \
 && python -c "\
from rapidocr_onnxruntime import RapidOCR; RapidOCR();\
print('OCR weights baked')" || echo "OCR bake skipped; signals.text will be null"

EXPOSE 7860

# ONE worker, on purpose. Each worker loads its own copy of CLIP (~2.4 GB
# resident), so a second one doubles memory for no throughput on a 2-vCPU box.
# Threads handle concurrent uploads; the GIL is released inside torch anyway.
#
# --timeout 300 because a CPU request is ~10s and can be more on a cold cache;
# gunicorn's 30s default would kill exactly the requests we care about.
# --preload loads the app (and its models) BEFORE forking, so container start
# pays the 9.5s model load once rather than on the first request.
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "--workers", "1", "--threads", "4", \
     "--timeout", "300", "--preload", "--chdir", "app", "app:app"]
