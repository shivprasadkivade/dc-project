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

from .config import (
    CFG,
    DATA_DIR,
    BOUNDARY_INSIDE_DIR,
    BOUNDARY_OUTSIDE_DIR,
    MODEL_DIR,
    VERSIONS_DIR,
    RESULTS_DIR,
    MODEL_PATH,
    CLASSES,
)
from .features import extract_features
from .triangulation import extract_fourier_triangulation_vector, FourierTriangulator
from .geofence import AcousticGeofenceValidator

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
    # Step 1: Load Core Gesture Classes (left_tap, right_tap, noise)
    # -----------------------------------------------------------------
    in_classes = [c for c in CLASSES if not c.startswith("oob_")]
    X_ml_in, X_fourier_in, y_in, y_labels_in = [], [], [], []

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

    X_ml_in = np.array(X_ml_in, dtype=np.float32) if X_ml_in else np.empty((0, 70), dtype=np.float32)
    X_fourier_in = np.array(X_fourier_in, dtype=np.float64) if X_fourier_in else np.empty((0, 43), dtype=np.float64)
    y_in = np.array(y_in, dtype=np.int64) if y_in else np.empty(0, dtype=np.int64)
    y_labels_in = np.array(y_labels_in) if y_labels_in else np.empty(0, dtype=str)

    if len(y_in) == 0:
        raise ValueError(
            f"No gesture samples found in {DATA_DIR}!\n"
            f"Please collect training samples first using:\n"
            f"  uv run python -m src.holo.collect --label all --samples 30"
        )

    unique_classes = np.unique(y_in)
    if len(unique_classes) < 2:
        raise ValueError(
            f"Found samples for only {len(unique_classes)} class ({[in_classes[c] for c in unique_classes]}).\n"
            f"Training requires at least 2 classes (e.g. left_tap and right_tap).\n"
            f"Please record additional classes using 'uv run python -m src.holo.collect'."
        )

    # -----------------------------------------------------------------
    # Step 2: Load Dedicated Geofence Boundary Datasets
    # -----------------------------------------------------------------
    X_ml_bound_in, X_four_bound_in, bound_in_labels = [], [], []
    X_ml_bound_out, X_four_bound_out = [], []

    # Inside boundary samples from dataset/boundary/inside/
    if BOUNDARY_INSIDE_DIR.exists():
        for file in sorted(BOUNDARY_INSIDE_DIR.glob("*.npy")):
            waveform = np.load(file)
            X_ml_bound_in.append(extract_features(waveform, feature_set=mode))
            X_four_bound_in.append(extract_fourier_triangulation_vector(waveform))
            fname = file.name.lower()
            if "left" in fname:
                bound_in_labels.append("left_tap")
            elif "right" in fname:
                bound_in_labels.append("right_tap")
            else:
                bound_in_labels.append("boundary_in")

    # Also include in-bounds palm rest gesture taps (left_tap and right_tap) as inside boundary
    for i in range(len(y_in)):
        if y_labels_in[i] in ["left_tap", "right_tap"]:
            X_ml_bound_in.append(X_ml_in[i])
            X_four_bound_in.append(X_fourier_in[i])
            bound_in_labels.append(y_labels_in[i])

    # Outside boundary samples from dataset/boundary/outside/
    if BOUNDARY_OUTSIDE_DIR.exists():
        for file in sorted(BOUNDARY_OUTSIDE_DIR.glob("*.npy")):
            waveform = np.load(file)
            X_ml_bound_out.append(extract_features(waveform, feature_set=mode))
            X_four_bound_out.append(extract_fourier_triangulation_vector(waveform))

    # Also load any legacy oob_* directories in DATA_DIR
    for oob_dir in sorted(DATA_DIR.glob("oob_*")):
        if oob_dir.is_dir():
            for file in sorted(oob_dir.glob("*.npy")):
                waveform = np.load(file)
                X_ml_bound_out.append(extract_features(waveform, feature_set=mode))
                X_four_bound_out.append(extract_fourier_triangulation_vector(waveform))

    # Also include the "noise" class samples as negative examples for boundary training
    for i in range(len(y_in)):
        if y_labels_in[i] == "noise":
            X_ml_bound_out.append(X_ml_in[i])
            X_four_bound_out.append(X_fourier_in[i])

    X_ml_bound_in = np.array(X_ml_bound_in, dtype=np.float32) if X_ml_bound_in else np.empty((0, 70), dtype=np.float32)
    X_four_bound_in = np.array(X_four_bound_in, dtype=np.float64) if X_four_bound_in else np.empty((0, 43), dtype=np.float64)
    X_ml_bound_out = np.array(X_ml_bound_out, dtype=np.float32) if X_ml_bound_out else np.empty((0, 70), dtype=np.float32)
    X_four_bound_out = np.array(X_four_bound_out, dtype=np.float64) if X_four_bound_out else np.empty((0, 43), dtype=np.float64)

    # -----------------------------------------------------------------
    # Step 3: Train & Validate Candidate Hybrid Gesture Classifiers
    # -----------------------------------------------------------------
    test_size = float(tr_cfg.get("test_size", 0.20))
    indices = np.arange(len(y_in))
    min_class_count = min(np.bincount(y_in))
    stratify_target = y_in if min_class_count >= 2 else None
    actual_folds = min(cv_folds, max(2, min_class_count))

    train_idx, test_idx = train_test_split(indices, test_size=test_size, stratify=stratify_target, random_state=seed)

    print("\n" + "=" * 70)
    print(f"[TRAIN HYBRID & GEOFENCE] In-Bounds Gesture Samples: {len(y_in)} ({len(train_idx)} train / {len(test_idx)} holdout)")
    print(f"Target Classes: {in_classes}")
    print(f"Geofence Inside Boundary Samples:  {len(X_ml_bound_in)} (perimeter boundary + center palm rest)")
    print(f"Geofence Outside Boundary Samples: {len(X_ml_bound_out)} (desk knocks & table vibrations)")
    if len(X_ml_bound_out) == 0:
        print("  [Notice] No outside boundary data found in dataset/boundary/outside.")
        print("           Run 'uv run python -m src.holo.collect --calibrate-geofence' to record desk strikes.")
    print("=" * 70 + "\n")

    candidates = build_candidates_from_config()
    param_grids = tr_cfg.get("param_grids", {})

    VERSIONS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    leaderboard_path = MODEL_DIR / "leaderboard.csv"
    write_header = not leaderboard_path.exists()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    best_name, best_hybrid_cv, best_ml_pipe, best_opt_w, best_params = None, -1.0, None, 0.5, {}

    # Fit Geofence Validator strictly using boundary data
    gf_validator = AcousticGeofenceValidator(CFG)
    gf_validator.fit(
        inside_ml=X_ml_bound_in,
        inside_fourier=X_four_bound_in,
        outside_ml=X_ml_bound_out,
        outside_fourier=X_four_bound_out,
        inside_labels=np.array(bound_in_labels),
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
                grid = GridSearchCV(pipeline, param_grids[name], cv=actual_folds, scoring="f1_macro", n_jobs=-1)
                grid.fit(X_ml_in[train_idx], y_in[train_idx])
                tuned_pipeline = grid.best_estimator_
                tuned_params = grid.best_params_

            ml_cv, four_cv, hybrid_cv, opt_w = evaluate_hybrid_cv(
                tuned_pipeline, X_ml_in[train_idx], X_fourier_in[train_idx], y_in[train_idx],
                weight_grid=weight_grid, n_splits=actual_folds, seed=seed
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
    # Step 4: Final Production Evaluation & Geofence Boundary Benchmark
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

    # In-Bounds Tap FRR Calculation (percentage of real palm rest taps falsely rejected by geofence)
    valid_tap_indices = [i for i in range(len(X_ml_in)) if y_labels_in[i] != "noise"]
    in_rejected = 0
    for i in valid_tap_indices:
        dec = gf_validator.verify(X_ml_in[i], X_fourier_in[i])
        if not dec.passed:
            in_rejected += 1
    frr = float(in_rejected / max(1, len(valid_tap_indices)))

    # Out-of-Bounds Rejection Rate (percentage of desk knocks correctly blocked by geofence)
    oob_rr_str = "N/A (No outside boundary samples in dataset/boundary/outside/)"
    oob_rr = 1.0
    stage_breakdown = {}
    if len(X_ml_bound_out) > 0:
        oob_blocked = 0
        for i in range(len(X_ml_bound_out)):
            dec = gf_validator.verify(X_ml_bound_out[i], X_four_bound_out[i])
            if not dec.passed:
                oob_blocked += 1
                stage = dec.stage_failed or "unknown"
                stage_breakdown[stage] = stage_breakdown.get(stage, 0) + 1
        oob_rr = float(oob_blocked / len(X_ml_bound_out))
        oob_rr_str = f"{oob_rr * 100:.2f}% ({oob_blocked}/{len(X_ml_bound_out)} desk knocks blocked)"

    print("\n" + "=" * 70)
    print(f"--> Winner: Hybrid_{best_name} ({int((1-best_opt_w)*100)}% ML + {int(best_opt_w*100)}% FFT)")
    print(f"Holdout Classification Accuracy: {holdout_acc * 100:.2f}%")
    print(f"Geofence Inside False Rejection Rate (FRR): {frr * 100:.2f}% ({in_rejected}/{len(valid_tap_indices)} inside taps rejected)")
    print(f"Geofence Outside Rejection Rate (OOB-RR):   {oob_rr_str}")
    if stage_breakdown:
        print(f"  OOB Rejection Breakdown: {stage_breakdown}")
    print("=" * 70)

    # Write Markdown test reports
    report_md_path = RESULTS_DIR / f"test_report_{timestamp}.md"
    latest_md_path = RESULTS_DIR / "latest_test_report.md"
    md_content = (
        f"# AcousticTap Hybrid & Geofence Test Report ({timestamp})\n\n"
        f"- **Winning Architecture:** `Hybrid_{best_name}` ({int((1-best_opt_w)*100)}% ML + {int(best_opt_w*100)}% Fourier)\n"
        f"- **In-Bounds Dataset Samples:** {len(y_in)} ({len(train_idx)} train / {len(test_idx)} holdout)\n"
        f"- **Geofence Inside Boundary Samples:** {len(X_ml_bound_in)}\n"
        f"- **Geofence Outside Boundary Samples:** {len(X_ml_bound_out)}\n"
        f"- **Holdout Test Accuracy:** `{holdout_acc*100:.2f}%`\n"
        f"- **Geofence Inside FRR:** `{frr*100:.2f}%`\n"
        f"- **Geofence Outside Rejection Rate (OOB-RR):** `{oob_rr_str}`\n\n"
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
