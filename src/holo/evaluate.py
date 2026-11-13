from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from .config import CFG, DATA_DIR, MODEL_PATH, CLASSES
from .features import extract_features
from .triangulation import extract_fourier_triangulation_vector

def evaluate_saved_model(model_file: Path, target_dir: Path = DATA_DIR, holdout_only: bool = False):
    if not model_file.exists():
        raise FileNotFoundError(f"Model file not found at {model_file}. Run train.py first.")

    saved = joblib.load(model_file)
    model = saved["model"]
    triangulator = saved.get("triangulator", None)
    w = float(saved.get("fourier_weight", 0.50))
    feature_set = saved.get("feature_set", "enhanced")
    classes = saved.get("classes", CLASSES)
    geofence = saved.get("geofence", None)

    # 1. Load In-Bounds Samples
    X_ml, X_fourier, y, y_labels = [], [], [], []
    for idx, label in enumerate(classes):
        folder = target_dir / label
        if not folder.exists():
            continue
        for file in sorted(folder.glob("*.npy")):
            waveform = np.load(file)
            X_ml.append(extract_features(waveform, feature_set=feature_set))
            X_fourier.append(extract_fourier_triangulation_vector(waveform))
            y.append(idx)
            y_labels.append(label)

    # 2. Load Out-of-Bounds Samples (prefixed with oob_)
    X_ml_oob, X_fourier_oob, y_oob_labels = [], [], []
    if target_dir.exists():
        for folder in sorted(target_dir.iterdir()):
            if folder.is_dir() and folder.name.startswith("oob_"):
                for file in sorted(folder.glob("*.npy")):
                    waveform = np.load(file)
                    X_ml_oob.append(extract_features(waveform, feature_set=feature_set))
                    X_fourier_oob.append(extract_fourier_triangulation_vector(waveform))
                    y_oob_labels.append(folder.name)

    X_ml = np.array(X_ml, dtype=np.float32)
    X_fourier = np.array(X_fourier, dtype=np.float64)
    y = np.array(y, dtype=np.int64)
    y_labels = np.array(y_labels)

    if len(y) == 0:
        raise ValueError(f"No in-bounds samples found in {target_dir}")

    eval_label = "FULL DATASET"
    if holdout_only and len(y) > 5:
        tr_cfg = CFG.get("training", {})
        seed = int(tr_cfg.get("random_seed", 42))
        test_size = float(tr_cfg.get("test_size", 0.20))
        indices = np.arange(len(y))
        _, test_idx = train_test_split(indices, test_size=test_size, stratify=y, random_state=seed)
        X_ml, X_fourier, y, y_labels = X_ml[test_idx], X_fourier[test_idx], y[test_idx], y_labels[test_idx]
        eval_label = f"UNSEEN HOLDOUT SET ({len(test_idx)} samples)"

    # Macro Model Probability Prediction
    p_ml = model.predict_proba(X_ml)
    if triangulator is not None:
        p_fourier = triangulator.predict_proba(X_fourier)
        probs = (1.0 - w) * p_ml + w * p_fourier
    else:
        probs = p_ml

    preds = np.argmax(probs, axis=1)
    acc = accuracy_score(y, preds)
    f1 = f1_score(y, preds, average="macro")

    print("\n" + "=" * 70)
    print(f"[EVALUATE - {eval_label}] Model: {model_file.name} | In-Bounds Samples: {len(y)}")
    print(f"Accuracy: {acc * 100:.2f}% | Macro F1: {f1:.4f}\n")
    print(classification_report(y, preds, target_names=classes))
    print("Confusion Matrix:")
    print(confusion_matrix(y, preds))

    # Geofence Evaluation
    print("\n" + "-" * 70)
    if geofence is None:
        print("[Geofence] Legacy model loaded without geofence metadata; skipping geofence evaluation.")
    else:
        print("[Geofence Evaluation & Boundary Verification]")
        valid_tap_indices = [i for i in range(len(X_ml)) if y_labels[i] != "noise"]
        in_rejected = 0
        in_stage_counts: Dict[str, int] = {}
        for i in valid_tap_indices:
            pred_class = classes[preds[i]]
            dec = geofence.verify(X_ml[i], X_fourier[i], predicted_class=pred_class)
            if not dec.passed:
                in_rejected += 1
                stage = dec.stage_failed or "unknown"
                in_stage_counts[stage] = in_stage_counts.get(stage, 0) + 1

        frr = (in_rejected / max(1, len(valid_tap_indices))) * 100.0
        print(f"  In-Bounds Tap False Rejection Rate (FRR): {frr:.2f}% ({in_rejected}/{len(valid_tap_indices)} valid taps falsely blocked)")
        if in_stage_counts:
            print(f"    In-Bounds Rejection Breakdown: {in_stage_counts}")

        noise_indices = [i for i in range(len(X_ml)) if y_labels[i] == "noise"]
        if noise_indices:
            noise_blocked = sum(1 for i in noise_indices if not geofence.verify(X_ml[i], X_fourier[i]).passed)
            print(f"  Ambient Noise Rejection Rate: {(noise_blocked / len(noise_indices))*100.0:.2f}% ({noise_blocked}/{len(noise_indices)} blocked by geofence)")

        if len(X_ml_oob) > 0:
            oob_blocked = 0
            oob_stage_counts: Dict[str, int] = {}
            for i in range(len(X_ml_oob)):
                dec = geofence.verify(X_ml_oob[i], X_fourier_oob[i])
                if not dec.passed:
                    oob_blocked += 1
                    stage = dec.stage_failed or "unknown"
                    oob_stage_counts[stage] = oob_stage_counts.get(stage, 0) + 1
            oob_rr = (oob_blocked / len(X_ml_oob)) * 100.0
            print(f"  Out-of-Bounds Rejection Rate (OOB-RR): {oob_rr:.2f}% ({oob_blocked}/{len(X_ml_oob)} correctly blocked)")
            print(f"    OOB Rejection Breakdown by Stage: {oob_stage_counts}")
        else:
            print("  Out-of-Bounds Rejection Rate (OOB-RR): N/A (no oob_* samples found, using analytical Ledoit-Wolf threshold)")
    print("=" * 70 + "\n")

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate any saved .joblib model on a dataset")
    parser.add_argument("--model", type=Path, default=MODEL_PATH, help="Path to .joblib model file")
    parser.add_argument("--data-dir", "--dataset", dest="data_dir", type=Path, default=DATA_DIR, help="Path to evaluation dataset directory")
    parser.add_argument("--holdout-only", action="store_true", help="Evaluate strictly on the isolated test partition")
    return parser

if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    evaluate_saved_model(args.model, args.data_dir, args.holdout_only)
