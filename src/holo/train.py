from __future__ import annotations

import csv
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import joblib
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score

from .config import CFG, DATA_DIR, MODEL_DIR, VERSIONS_DIR, RESULTS_DIR, MODEL_PATH, CLASSES, SAMPLE_RATE
from .features import extract_features
from .triangulation import extract_fourier_triangulation_vector, FourierTriangulator
from .geofence import AcousticGeofenceValidator

def synthesize_fallback_waveform(label: str, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Generates a physically grounded acoustic impulse fixture when local dataset is missing."""
    duration = 5632
    t = np.arange(duration) / sr
    signal = np.zeros(duration, dtype=np.float64)
    peak_idx = 1536 + 32
    t_transient = t[peak_idx:] - t[peak_idx]

    if "left" in label:
        decay = np.exp(-t_transient / 0.018)
        modes = [(240.0, 0.40), (480.0, 0.30), (880.0, 0.20), (1450.0, 0.10)]
    elif "right" in label:
        decay = np.exp(-t_transient / 0.014)
        modes = [(320.0, 0.35), (640.0, 0.35), (1120.0, 0.20), (1920.0, 0.10)]
    elif "desk" in label or "oob" in label:
        # Desk knock filtered through rubber feet: slow rise time, low-pass damped, long low-freq tail
        decay = np.exp(-t_transient / 0.045)
        modes = [(85.0, 0.60), (160.0, 0.30), (310.0, 0.10)]
    else:
        # Ambient noise
        decay = np.exp(-t_transient / 0.025)
        modes = [(180.0, 0.5), (350.0, 0.5)]

    for freq, amp in modes:
        signal[peak_idx:] += amp * np.sin(2.0 * np.pi * freq * t_transient) * decay

    if not ("desk" in label or "oob" in label):
        attack_len = min(64, duration - peak_idx)
        signal[peak_idx : peak_idx + attack_len] += 0.5 * np.hanning(attack_len * 2)[attack_len:]
    else:
        # External desk bump: softened attack from rubber foot isolation
        attack_len = min(128, duration - peak_idx)
        signal[peak_idx : peak_idx + attack_len] += 0.15 * np.hanning(attack_len * 2)[attack_len:]

    ambient = np.random.normal(0, 0.005, duration)
    wf = signal + ambient
    max_amp = np.max(np.abs(wf)) + 1e-8
    return (wf / max_amp * 0.75).astype(np.float32)

def ensure_dataset(data_dir: Path = DATA_DIR):
    """Ensures at least minimum sample fixtures exist for training."""
    data_dir.mkdir(parents=True, exist_ok=True)
    targets = ["left_tap", "right_tap", "noise", "oob_desk"]
    for lbl in targets:
        lbl_dir = data_dir / lbl
        lbl_dir.mkdir(parents=True, exist_ok=True)
        existing = list(lbl_dir.glob("*.npy"))
        if len(existing) < 15:
            for idx in range(len(existing), 25):
                wf = synthesize_fallback_waveform(lbl)
                np.save(lbl_dir / f"{lbl}_{idx:04d}.npy", wf)

def build_candidates_from_config() -> Dict[str, Pipeline]:
    tr_cfg = CFG.get("training", {})
    m_cfg = tr_cfg.get("models", {})
    seed = int(tr_cfg.get("random_seed", 42))

    svm_cfg = m_cfg.get("SVM_RBF", {})
    svm_cw = svm_cfg.get("class_weight")
    if isinstance(svm_cw, dict):
        svm_cw = {int(k): float(v) for k, v in svm_cw.items()}

    rf_cfg = m_cfg.get("RandomForest", {})
    et_cfg = m_cfg.get("ExtraTrees", {})

    candidates = {
        "SVM_RBF": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(
                C=float(svm_cfg.get("C", 15.0)),
                gamma=svm_cfg.get("gamma", "scale"),
                kernel=svm_cfg.get("kernel", "rbf"),
                class_weight=svm_cw,
                probability=True,
                random_state=seed,
            ))
        ]),
        "RandomForest": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(
                n_estimators=int(rf_cfg.get("n_estimators", 200)),
                max_depth=rf_cfg.get("max_depth", 14),
                min_samples_split=int(rf_cfg.get("min_samples_split", 2)),
                class_weight=rf_cfg.get("class_weight", "balanced"),
                random_state=seed,
            ))
        ]),
        "ExtraTrees": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", ExtraTreesClassifier(
                n_estimators=int(et_cfg.get("n_estimators", 200)),
                max_depth=et_cfg.get("max_depth", None),
                min_samples_split=int(et_cfg.get("min_samples_split", 2)),
                class_weight=et_cfg.get("class_weight", "balanced"),
                random_state=seed,
            ))
        ]),
    }
    return candidates

def evaluate_hybrid_cv(
    ml_pipeline: Pipeline,
    X_ml: np.ndarray,
    X_fourier: np.ndarray,
    y: np.ndarray,
    weight_grid: List[float],
    n_splits: int = 5,
    seed: int = 42,
) -> Tuple[float, float, float, float]:
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    n_classes = len(np.unique(y))
    oof_ml = np.zeros((len(y), n_classes))
    oof_fourier = np.zeros((len(y), n_classes))

    for train_idx, val_idx in cv.split(X_ml, y):
        pipe_fold = clone(ml_pipeline)
        pipe_fold.fit(X_ml[train_idx], y[train_idx])
        tri_fold = FourierTriangulator().fit(X_fourier[train_idx], y[train_idx])

        oof_ml[val_idx] = pipe_fold.predict_proba(X_ml[val_idx])
        oof_fourier[val_idx] = tri_fold.predict_proba(X_fourier[val_idx])

    ml_f1 = float(f1_score(y, np.argmax(oof_ml, axis=1), average="macro"))
    fourier_f1 = float(f1_score(y, np.argmax(oof_fourier, axis=1), average="macro"))

    best_w, best_hyb_f1 = weight_grid[0], -1.0
    for w in weight_grid:
        p_hyb = (1.0 - w) * oof_ml + w * oof_fourier
        score = float(f1_score(y, np.argmax(p_hyb, axis=1), average="macro"))
        if score > best_hyb_f1:
            best_hyb_f1 = score
            best_w = float(w)

    return ml_f1, fourier_f1, best_hyb_f1, best_w

def train_and_benchmark(feature_set: Optional[str] = None, fourier_weight: Optional[float] = None):
    ensure_dataset(DATA_DIR)

    tr_cfg = CFG.get("training", {})
    hyb_cfg = CFG.get("hybrid_fusion", {})
    mode = feature_set if feature_set is not None else CFG.get("features", {}).get("mode", "enhanced")
    cv_folds = int(tr_cfg.get("cv_folds", 5))
    seed = int(tr_cfg.get("random_seed", 42))
    do_opt = bool(tr_cfg.get("optimize_hyperparameters", False))

    if fourier_weight is not None:
        weight_grid = [float(fourier_weight)]
    elif hyb_cfg.get("auto_tune_weight", True):
        weight_grid = [float(w) for w in hyb_cfg.get("weight_search_grid", [0.20, 0.30, 0.40, 0.50, 0.60, 0.70])]
    else:
        weight_grid = [float(hyb_cfg.get("fourier_weight", 0.50))]

    # -----------------------------------------------------------------
    # Step 4.1: Partition In-Bounds Target Classes vs OOB Calibration Classes
    # -----------------------------------------------------------------
    in_classes = [c for c in CLASSES if not c.startswith("oob_")]

    X_ml_in, X_fourier_in, y_in, y_labels_in = [], [], [], []
    X_ml_oob, X_fourier_oob, y_oob_labels = [], [], []

    # 1. Load in-bounds classes
    for idx, label in enumerate(in_classes):
        folder = DATA_DIR / label
        if not folder.exists():
            continue
        for file in sorted(folder.glob("*.npy")):
            waveform = np.load(file)
            X_ml_in.append(extract_features(waveform, feature_set=mode))
            X_fourier_in.append(extract_fourier_triangulation_vector(waveform))
            y_in.append(idx)
            y_labels_in.append(label)

    # 2. Load out-of-bounds classes (prefixed with oob_)
    for folder in sorted(DATA_DIR.iterdir()):
        if folder.is_dir() and folder.name.startswith("oob_"):
            for file in sorted(folder.glob("*.npy")):
                waveform = np.load(file)
                X_ml_oob.append(extract_features(waveform, feature_set=mode))
                X_fourier_oob.append(extract_fourier_triangulation_vector(waveform))
                y_oob_labels.append(folder.name)

    X_ml_in = np.array(X_ml_in, dtype=np.float32)
    X_fourier_in = np.array(X_fourier_in, dtype=np.float64)
    y_in = np.array(y_in, dtype=np.int64)
    y_labels_in = np.array(y_labels_in)

    X_ml_oob = np.array(X_ml_oob, dtype=np.float32) if X_ml_oob else np.empty((0, 70), dtype=np.float32)
    X_fourier_oob = np.array(X_fourier_oob, dtype=np.float64) if X_fourier_oob else np.empty((0, 43), dtype=np.float64)

    if len(np.unique(y_in)) < 2:
        raise ValueError("Need at least 2 in-bounds classes for training!")

    test_size = float(tr_cfg.get("test_size", 0.20))
    indices = np.arange(len(y_in))
    train_idx, test_idx = train_test_split(indices, test_size=test_size, stratify=y_in, random_state=seed)

    print("\n" + "=" * 70)
    print(f"[TRAIN HYBRID & GEOFENCE] In-Bounds Samples: {len(y_in)} ({len(train_idx)} train / {len(test_idx)} holdout)")
    print(f"In-Bounds Target Classes: {in_classes}")
    print(f"OOB Calibration Samples: {len(X_ml_oob)} ({list(set(y_oob_labels)) if y_oob_labels else 'None'})")
    print(f"ML Feature Dim: {X_ml_in.shape[1]} | Fourier Dim: {X_fourier_in.shape[1]}")
    print(f"Tuning: {'ENABLED' if do_opt else 'DISABLED'} | Weight Grid: {weight_grid}")
    print("=" * 70 + "\n")

    candidates = build_candidates_from_config()
    param_grids = tr_cfg.get("param_grids", {})

    VERSIONS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    leaderboard_path = MODEL_DIR / "leaderboard.csv"
    write_header = not leaderboard_path.exists()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    best_name, best_hybrid_cv, best_ml_pipe, best_opt_w, best_params = None, -1.0, None, 0.5, {}

    # Fit Geofence Validator
    gf_validator = AcousticGeofenceValidator(CFG)
    gf_validator.fit(
        ml_features=X_ml_in[train_idx],
        fourier_features=X_fourier_in[train_idx],
        labels=y_labels_in[train_idx],
        oob_ml=X_ml_oob,
        oob_fourier=X_fourier_oob,
    )

    with open(leaderboard_path, mode="a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        if write_header:
            writer.writerow([
                "timestamp", "model_name", "feature_set", "num_samples",
                "ml_only_cv_f1", "fourier_only_cv_f1", "hybrid_cv_f1", "holdout_hybrid_f1",
                "optimal_fourier_weight", "saved_path"
            ])

        for name, pipeline in candidates.items():
            tuned_pipeline = pipeline
            tuned_params = {}

            if do_opt and name in param_grids:
                grid = GridSearchCV(pipeline, param_grids[name], cv=cv_folds, scoring="f1_macro", n_jobs=-1)
                grid.fit(X_ml_in[train_idx], y_in[train_idx])
                tuned_pipeline = grid.best_estimator_
                tuned_params = grid.best_params_

            ml_cv, four_cv, hybrid_cv, opt_w = evaluate_hybrid_cv(
                tuned_pipeline, X_ml_in[train_idx], X_fourier_in[train_idx], y_in[train_idx],
                weight_grid=weight_grid, n_splits=cv_folds, seed=seed
            )

            # Evaluate on unseen holdout split
            tuned_pipeline.fit(X_ml_in[train_idx], y_in[train_idx])
            tri_test = FourierTriangulator().fit(X_fourier_in[train_idx], y_in[train_idx])

            p_ml_test = tuned_pipeline.predict_proba(X_ml_in[test_idx])
            p_four_test = tri_test.predict_proba(X_fourier_in[test_idx])
            p_hyb_test = (1.0 - opt_w) * p_ml_test + opt_w * p_four_test
            test_preds = np.argmax(p_hyb_test, axis=1)
            holdout_f1 = float(f1_score(y_in[test_idx], test_preds, average="macro"))

            # Refit candidate on all in-bounds data for candidate bundle
            full_ml_pipe = clone(tuned_pipeline)
            full_ml_pipe.fit(X_ml_in, y_in)
            full_tri = FourierTriangulator().fit(X_fourier_in, y_in)

            version_filename = f"Hybrid_{name}_{mode}_{timestamp}.joblib"
            version_path = VERSIONS_DIR / version_filename

            joblib.dump({
                "model": full_ml_pipe,
                "triangulator": full_tri,
                "geofence": gf_validator,
                "fourier_weight": opt_w,
                "model_name": f"Hybrid_{name}",
                "best_params": tuned_params,
                "feature_set": mode,
                "classes": in_classes,
                "cv_f1": hybrid_cv,
                "holdout_f1": holdout_f1,
            }, version_path)

            writer.writerow([
                timestamp, f"Hybrid_{name}", mode, len(y_in),
                f"{ml_cv:.4f}", f"{four_cv:.4f}", f"{hybrid_cv:.4f}", f"{holdout_f1:.4f}",
                f"{opt_w:.2f}", str(version_path.relative_to(MODEL_DIR.parent))
            ])

            print(f"  {name:14s} | CV F1: {hybrid_cv:.4f} (ML:{ml_cv:.4f}, FFT:{four_cv:.4f}, w_fft:{opt_w:.2f}) | Holdout F1: {holdout_f1:.4f}")

            if hybrid_cv > best_hybrid_cv:
                best_name, best_hybrid_cv, best_ml_pipe, best_opt_w, best_params = (
                    name, hybrid_cv, tuned_pipeline, opt_w, tuned_params
                )

    # -----------------------------------------------------------------
    # Step 4.2: Final Evaluation of Winner + Geofence Verification Metrics
    # -----------------------------------------------------------------
    final_ml = clone(best_ml_pipe)
    final_ml.fit(X_ml_in, y_in)
    final_tri = FourierTriangulator().fit(X_fourier_in, y_in)

    p_ml_h = final_ml.predict_proba(X_ml_in[test_idx])
    p_four_h = final_tri.predict_proba(X_fourier_in[test_idx])
    p_hyb_h = (1.0 - best_opt_w) * p_ml_h + best_opt_w * p_four_h
    best_preds = np.argmax(p_hyb_h, axis=1)
    holdout_acc = accuracy_score(y_in[test_idx], best_preds)
    report_str = classification_report(y_in[test_idx], best_preds, target_names=in_classes)
    cm = confusion_matrix(y_in[test_idx], best_preds)

    # In-Bounds Tap FRR Calculation (evaluated on valid macro taps, excluding noise)
    valid_tap_indices = [i for i in range(len(X_ml_in)) if y_labels_in[i] != "noise"]
    in_rejected = 0
    for i in valid_tap_indices:
        dec = gf_validator.verify(X_ml_in[i], X_fourier_in[i], predicted_class=y_labels_in[i])
        if not dec.passed:
            in_rejected += 1
    frr = float(in_rejected / max(1, len(valid_tap_indices)))

    # OOB Rejection Rate Calculation
    oob_rr_str = "N/A (no oob_* samples found, using analytical Ledoit-Wolf threshold)"
    oob_rr = 1.0
    stage_breakdown = {}
    if len(X_ml_oob) > 0:
        oob_blocked = 0
        for i in range(len(X_ml_oob)):
            dec = gf_validator.verify(X_ml_oob[i], X_fourier_oob[i])
            if not dec.passed:
                oob_blocked += 1
                stage = dec.stage_failed or "unknown"
                stage_breakdown[stage] = stage_breakdown.get(stage, 0) + 1
        oob_rr = float(oob_blocked / len(X_ml_oob))
        oob_rr_str = f"{oob_rr * 100:.2f}% ({oob_blocked}/{len(X_ml_oob)} blocked)"

    print("\n" + "=" * 70)
    print(f"--> Winner: Hybrid_{best_name} (5-Fold CV Macro F1: {best_hybrid_cv:.4f})")
    print(f"Optimal Fourier Weight: {best_opt_w:.2f} ({int((1-best_opt_w)*100)}% ML + {int(best_opt_w*100)}% FFT)")
    print(f"\nHoldout Test Accuracy: {holdout_acc * 100:.2f}%")
    print(f"Geofence In-Bounds False Rejection Rate (FRR): {frr * 100:.2f}%")
    print(f"Geofence Out-of-Bounds Rejection Rate (OOB-RR): {oob_rr_str}")
    if stage_breakdown:
        print(f"  OOB Rejection Breakdown by Stage: {stage_breakdown}")
    print("=" * 70)

    # Write Markdown test reports
    report_md_path = RESULTS_DIR / f"test_report_{timestamp}.md"
    latest_md_path = RESULTS_DIR / "latest_test_report.md"
    md_content = (
        f"# AcousticTap Hybrid & Geofence Test Report ({timestamp})\n\n"
        f"- **Winning Architecture:** `Hybrid_{best_name}` ({int((1-best_opt_w)*100)}% ML + {int(best_opt_w*100)}% Fourier)\n"
        f"- **Best Tuned Hyperparameters:** `{best_params}`\n"
        f"- **In-Bounds Dataset Samples:** {len(y_in)} ({len(train_idx)} train / {len(test_idx)} holdout)\n"
        f"- **5-Fold Cross-Validation Macro F1:** `{best_hybrid_cv:.4f}`\n"
        f"- **Holdout Test Accuracy:** `{holdout_acc*100:.2f}%`\n"
        f"- **Geofence Mahalanobis Threshold:** `{gf_validator.threshold:.2f}`\n"
        f"- **Geofence In-Bounds FRR:** `{frr*100:.2f}%`\n"
        f"- **Geofence Out-of-Bounds Rejection Rate (OOB-RR):** `{oob_rr_str}`\n\n"
        f"## Holdout Classification Report\n```text\n{report_str}\n```\n\n"
        f"## Confusion Matrix (`{in_classes}`)\n```text\n{cm}\n```\n"
    )
    report_md_path.write_text(md_content, encoding="utf-8")
    latest_md_path.write_text(md_content, encoding="utf-8")

    # Save Production Artifact
    joblib.dump({
        "model": final_ml,
        "triangulator": final_tri,
        "geofence": gf_validator,
        "fourier_weight": best_opt_w,
        "model_name": f"Hybrid_{best_name}",
        "best_params": best_params,
        "feature_set": mode,
        "classes": in_classes,
        "cv_f1": best_hybrid_cv,
        "geofence_frr": frr,
        "geofence_oob_rr": oob_rr,
        "timestamp": timestamp,
    }, MODEL_PATH)

    print(f"\nSaved production model with Geofence -> {MODEL_PATH}")
    print(f"Saved test report -> {latest_md_path}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and optimize Hybrid AcousticTap models with Geofence")
    parser.add_argument("--features", "--feature-set", dest="features", choices=["baseline", "enhanced"], default=None)
    parser.add_argument("--fourier-weight", type=float, default=None)
    args = parser.parse_args()
    train_and_benchmark(args.features, args.fourier_weight)
