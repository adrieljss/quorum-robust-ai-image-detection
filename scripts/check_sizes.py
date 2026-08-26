"""Real on-disk sizes before anyone starts a download. >5GB -> stream."""
from huggingface_hub import HfApi

DATASETS = ["saberzl/SID_Set", "saberzl/So-Fake-OOD",
            "pujanpaudel/deepfake_face_classification"]

api = HfApi()
for name in DATASETS:
    try:
        info = api.dataset_info(name, files_metadata=True)
        gb = sum(f.size for f in info.siblings if f.size) / 1e9
        print(f"{name:50s} {gb:8.1f} GB  ({len(info.siblings)} files)"
              f"  -> {'STREAM' if gb > 5 else 'download'}")
    except Exception as e:
        print(f"{name:50s} ERROR: {e}")
