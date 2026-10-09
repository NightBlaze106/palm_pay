from huggingface_hub import hf_hub_download
import os

os.makedirs("models", exist_ok=True)
print("Downloading CCNet fp32 (~257 MB)...")
p = hf_hub_download(
    repo_id="kyereboatengcaleb/palm-ccnet-onnx",
    filename="palm_ccnet.onnx",
    local_dir="models"
)
print(f"Saved: {p}")
