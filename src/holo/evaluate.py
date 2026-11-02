from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any
import joblib
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

try:
    from .config import CFG, CLASSES, DATA_DIR, MODEL_PATH, RANDOM_SEED, SAMPLE_RATE, TEST_SIZE
    from .features import extract_features
    from .train import load_dataset, train_and_benchmark
    from .triangulation import FourierTriangulator, extract_fourier_triangulation_vector
    from .ultrasonic import UltrasonicSonarGate
except (ImportError, ValueError):
    from config import CFG, CLASSES, DATA_DIR, MODEL_PATH, RANDOM_SEED, SAMPLE_RATE, TEST_SIZE  # type: ignore
    from features import extract_features  # type: ignore
    from train import load_dataset, train_and_benchmark  # type: ignore
    from triangulation import FourierTriangulator, extract_fourier_triangulation_vector  # type: ignore
    from ultrasonic import UltrasonicSonarGate  # type: ignore


def evaluate_saved_model(
    model_file: Path = MODEL_PATH,
    data_dir: Path = DATA_DIR,
    holdout_only: bool = False,
) -> dict[str, float]:
    """
    Evaluates the saved production model artifact on the target dataset
    using the Hybrid ML + Fourier Triangulation + Ultrasonic Gate pipeline.
    """
    if not model_file.is_file():
        print(f"[Evaluate] Model not found at {model_file}. Training a new model first...")
        train_and_benchmark()

    bundle: dict[str, Any] = joblib.load(model_file)
    model = bundle["model"]
    classes = list(bundle.get("classes", CLASSES))
    w_four = float(bundle.get("fourier_weight", 0.50))
    feature_set = bundle.get("feature_set", "enhanced")
    expected_dim = bundle.get("feature_dim", getattr(model, "n_features_in_", None))

    tri_data = bundle.get("triangulator")
    if isinstance(tri_data, dict):
        triangulator = FourierTriangulator.from_dict(tri_data)
    else:
        triangulator = tri_data

    waveforms, str_labels = load_dataset(data_dir=data_dir, sr=SAMPLE_RATE, classes=classes)
    if len(waveforms) == 0:
        raise ValueError(f"No valid evaluation audio samples found in {data_dir}")

    label_to_idx = {name: idx for idx, name in enumerate(classes)}
    valid_pairs = [(w, label_to_idx[lbl]) for w, lbl in zip(waveforms, str_labels) if lbl in label_to_idx]
    if len(valid_pairs) == 0:
        raise ValueError("No audio clips matched known classes")

    eval_waves = [p[0] for p in valid_pairs]
    y_all = np.array([p[1] for p in valid_pairs], dtype=np.int64)

    eval_label = "FULL DATASET"
    if holdout_only:
        indices = np.arange(len(y_all))
        min_class = int(np.min(np.unique(y_all, return_counts=True)[1]))
        if min_class >= 2:
            _, te_idx = train_test_split(
                indices, test_size=TEST_SIZE, stratify=y_all, random_state=RANDOM_SEED
            )
            eval_waves = [eval_waves[i] for i in te_idx]
            y_all = y_all[te_idx]
            eval_label = f"UNSEEN HOLDOUT SET ({len(te_idx)} clips)"

    # Extract deterministic features
    X_ml_list = []
    X_four_list = []
    sonar = UltrasonicSonarGate(sr=SAMPLE_RATE)
    sonar_results = []

    for w in eval_waves:
        feats = extract_features(w, sr=SAMPLE_RATE, feature_set=feature_set)
        fourier_vec = extract_fourier_triangulation_vector(w, sr=SAMPLE_RATE)
        X_ml_list.append(feats)
        X_four_list.append(fourier_vec)
        sonar_results.append(sonar.evaluate_window(w))

    X_ml = np.array(X_ml_list, dtype=np.float32)
    X_four = np.array(X_four_list, dtype=np.float32)

    # Verify feature dimension against trained model
    if expected_dim is not None and X_ml.shape[1] != expected_dim:
        raise ValueError(
            f"Feature dimension mismatch: extracted {X_ml.shape[1]} features, but model expects {expected_dim}"
        )

    # Compute Hybrid Probabilities
    p_ml = model.predict_proba(X_ml)
    if triangulator is not None:
        p_four = triangulator.predict_proba(X_four)
        p_hyb = (1.0 - w_four) * p_ml + w_four * p_four
    else:
        p_hyb = p_ml

    # Apply Ultrasonic Sonar Gate modulation when carrier active
    final_preds = []
    for i in range(len(p_hyb)):
        probs = {cls_name: float(p_hyb[i, j]) for j, cls_name in enumerate(classes)}
        us = sonar_results[i]
        if us["active"]:
            if "left_tap" in probs:
                probs["left_tap"] *= float(0.5 + 0.5 * us["gate_left"])
            if "right_tap" in probs:
                probs["right_tap"] *= float(0.5 + 0.5 * us["gate_right"])

        best_cls = max(probs, key=lambda k: probs.get(k, 0.0))
        final_preds.append(label_to_idx[best_cls])

    y_pred = np.array(final_preds, dtype=np.int64)
    acc = float(accuracy_score(y_all, y_pred))
    macro_f1 = float(f1_score(y_all, y_pred, average="macro", zero_division=0))

    # Safe classification report extraction avoiding KeyErrors
    report_dict = classification_report(
        y_all, y_pred, target_names=classes, output_dict=True, zero_division=0
    )
    report_str = classification_report(
        y_all, y_pred, target_names=classes, zero_division=0
    )
    cm = confusion_matrix(y_all, y_pred, labels=range(len(classes)))

    print("\n" + "=" * 70)
    print(f"[EVALUATION - {eval_label}] Model: {model_file.name}")
    print(f"Architecture: {bundle.get('model_name', 'HybridModel')} ({round((1-w_four)*100)}% ML + {round(w_four*100)}% Fourier)")
    print(f"Accuracy: {acc * 100:.2f}% | Macro F1: {macro_f1:.4f} | Total Samples: {len(y_all)}")
    print("=" * 70)
    print("\nClassification Report:\n" + report_str)
    print("Confusion Matrix:\n" + str(cm))

    # Safe dictionary inspection
    macro_avg = report_dict.get("macro avg", {})
    macro_f1_safe = float(macro_avg.get("f1-score", macro_f1))

    return {"accuracy": acc, "macro_f1": macro_f1_safe}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate AcousticTap V3 Hybrid model")
    parser.add_argument("--model", type=Path, default=MODEL_PATH, help="Path to .joblib model")
    parser.add_argument("--data-dir", "--dataset", dest="data_dir", type=Path, default=DATA_DIR)
    parser.add_argument("--holdout-only", action="store_true", help="Evaluate strictly on unseen holdout set")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    evaluate_saved_model(
        model_file=args.model,
        data_dir=args.data_dir,
        holdout_only=args.holdout_only,
    )


if __name__ == "__main__":
    main()
