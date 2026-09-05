#!/usr/bin/env python3
"""
PalmPay Phase-2 DeepPalm Model Training Script.
Trains MobileNetV3-Small on consent-collected pilot palm ROIs using ArcFace / Triplet metric loss.
Enforces subject-disjoint splits, early stopping, and CPU-friendly execution.
"""

import argparse
import os
import random
import sys
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from training.dataset import PalmDataset


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def train_deep_model(
    data_dir: str,
    output_onnx: str = "models/palm_embedder.onnx",
    epochs: int = 25,
    batch_size: int = 16,
    lr: float = 1e-3,
    seed: int = 42,
):
    set_seed(seed)

    try:
        import torch
        import torch.optim as optim
        from torch.utils.data import DataLoader, Dataset
        from training.model import MobileNetV3PalmEmbedder, ArcFaceLoss
    except ImportError:
        print("[Train Error] PyTorch is required for model training.")
        print("To install training dependencies: pip install -r requirements-train.txt")
        return 1

    ds_mgr = PalmDataset(data_dir=data_dir)
    subjects = ds_mgr.discover_subjects()
    if not subjects:
        print(f"[Train] No enrolled subjects found in {data_dir}.")
        print("[Train] Run the consent-based pilot enrollment at /enroll first to capture data.")
        return 1

    print(f"[Train] Found {len(subjects)} subjects across {sum(len(v) for v in subjects.values())} ROI crops.")
    train_samples, val_samples = ds_mgr.split_subject_disjoint(subjects, val_ratio=0.2, seed=seed)
    print(f"[Train] Subject-disjoint split: {len(train_samples)} train samples, {len(val_samples)} validation samples.")

    # Build train-only subject class index (ArcFace trains exclusively on training identities)
    train_subjects = sorted(list({s["subject_id"] for s in train_samples}))
    subject_to_idx = {k: idx for idx, k in enumerate(train_subjects)}

    in_channels = 3  # Standard MobileNetV3 ImageNet backbone input channels

    class PyTorchPalmDataset(Dataset):
        def __init__(self, sample_list):
            self.samples = sample_list

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            item = self.samples[idx]
            roi = ds_mgr.load_and_preprocess(item["image_path"])
            roi = ds_mgr.augment_roi(roi)

            # Normalize to [-1.0, 1.0] float tensor (128x128)
            norm_roi = (roi.astype(np.float32) / 127.5) - 1.0
            if in_channels == 3:
                tensor = np.repeat(np.expand_dims(norm_roi, 0), 3, axis=0)
            else:
                tensor = np.expand_dims(norm_roi, 0)
            label = subject_to_idx[item["subject_id"]]
            return torch.tensor(tensor, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

    train_loader = DataLoader(PyTorchPalmDataset(train_samples), batch_size=batch_size, shuffle=True)

    # Model and metric loss
    device = torch.device("cpu")
    model = MobileNetV3PalmEmbedder(feature_dim=config.DEEP_FEATURE_DIM, in_channels=in_channels, pretrained=True).to(device)
    criterion = ArcFaceLoss(num_classes=len(train_subjects), embedding_dim=config.DEEP_FEATURE_DIM).to(device)
    optimizer = optim.AdamW(list(model.parameters()) + list(criterion.parameters()), lr=lr, weight_decay=1e-4)

    best_val_score = -float("inf")
    os.makedirs("models/checkpoints", exist_ok=True)
    best_ckpt_path = "models/checkpoints/deep_palm_best.pt"

    def evaluate_val_biometrics():
        """
        Computes genuine and impostor pair similarities on the subject-disjoint validation set,
        evaluating biometric verification metrics: EER, Zero-FAR threshold, and separation.
        """
        if not val_samples:
            return None

        model.eval()
        subject_embs = {}
        with torch.no_grad():
            for it in val_samples:
                s_id = it["subject_id"]
                roi = ds_mgr.load_and_preprocess(it["image_path"])
                norm_roi = (roi.astype(np.float32) / 127.5) - 1.0
                if in_channels == 3:
                    tensor = np.repeat(np.expand_dims(norm_roi, 0), 3, axis=0)
                else:
                    tensor = np.expand_dims(norm_roi, 0)
                t_in = torch.tensor(tensor, dtype=torch.float32).unsqueeze(0).to(device)
                emb = model(t_in).cpu().numpy().flatten()
                subject_embs.setdefault(s_id, []).append(emb)

        genuine_scores = []
        impostor_scores = []
        s_keys = sorted(list(subject_embs.keys()))

        for s_id, embs in subject_embs.items():
            for i in range(len(embs)):
                for j in range(i + 1, len(embs)):
                    genuine_scores.append(float(np.dot(embs[i], embs[j])))

        for i in range(len(s_keys)):
            for j in range(i + 1, len(s_keys)):
                for ea in subject_embs[s_keys[i]]:
                    for eb in subject_embs[s_keys[j]]:
                        impostor_scores.append(float(np.dot(ea, eb)))

        if not genuine_scores or not impostor_scores:
            return None

        from scripts.eval_deep_model import compute_fine_metrics
        fine = compute_fine_metrics(genuine_scores, impostor_scores)
        gen_m = float(np.mean(genuine_scores))
        imp_m = float(np.mean(impostor_scores))
        eer_val = fine["eer"]["eer_value"] if fine.get("eer") else 0.0

        return {
            "eer": eer_val,
            "zero_far_gar": fine.get("zero_far_gar", 0.0),
            "zero_far_threshold": fine.get("zero_far_threshold", 0.0),
            "separation": gen_m - imp_m,
            "gen_mean": gen_m,
            "imp_mean": imp_m,
        }

    print("[Train] Beginning training on CPU...")
    for epoch in range(1, epochs + 1):
        model.train()
        criterion.train()
        total_loss = 0.0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            embeddings = model(imgs)
            loss = criterion(embeddings, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(labels)

        train_loss = total_loss / len(train_samples)

        # Biometric verification evaluation on subject-disjoint validation set
        val_metrics = evaluate_val_biometrics()
        if val_metrics is not None:
            # Score combining low EER and high margin of separation between genuine and impostor distributions
            val_score = val_metrics["separation"] - (2.0 * val_metrics["eer"])
            print(
                f"  Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} | "
                f"Val EER: {val_metrics['eer']*100:.2f}% | "
                f"Val Sep: {val_metrics['separation']:+.4f} | "
                f"Zero-FAR GAR: {val_metrics['zero_far_gar']*100:.1f}%"
            )
            if val_score > best_val_score:
                best_val_score = val_score
                torch.save(model.state_dict(), best_ckpt_path)
        else:
            print(f"  Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.4f}")
            # Fallback if validation pairs cannot be formed
            val_score = -train_loss
            if val_score > best_val_score:
                best_val_score = val_score
                torch.save(model.state_dict(), best_ckpt_path)

    # Export best model to ONNX
    print(f"[Train] Exporting best model to ONNX: {output_onnx}")
    os.makedirs(os.path.dirname(output_onnx), exist_ok=True)
    if os.path.exists(best_ckpt_path):
        model.load_state_dict(torch.load(best_ckpt_path, map_location=device))
    model.eval()

    dummy_input = torch.randn(1, in_channels, config.DEEP_INPUT_SIZE, config.DEEP_INPUT_SIZE, device=device)
    torch.onnx.export(
        model,
        dummy_input,
        output_onnx,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        opset_version=13,
    )
    print(f"[Train] Model exported successfully to {output_onnx} (1x{in_channels}x{config.DEEP_INPUT_SIZE}x{config.DEEP_INPUT_SIZE} -> {config.DEEP_FEATURE_DIM}-D).")
    return 0


def main():
    parser = argparse.ArgumentParser(description="PalmPay Deep Model Training")
    parser.add_argument("--data-dir", type=str, default=config.DATASET_DIR, help="Path to pilot ROI directory")
    parser.add_argument("--output", type=str, default=config.DEEP_MODEL_PATH, help="Path for exported ONNX model")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic seed")
    args = parser.parse_args()

    sys.exit(train_deep_model(
        data_dir=args.data_dir,
        output_onnx=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
    ))


if __name__ == "__main__":
    main()
