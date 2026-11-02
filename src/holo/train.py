from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path
from typing import Any
import joblib
import numpy as np
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

try:
    from .config import (
        AUTO_TUNE_WEIGHT,
        CFG,
        CLASSES,
        CV_FOLDS,
        DATA_DIR,
        FOURIER_WEIGHT,
        MODEL_DIR,
        MODEL_PATH,
        RANDOM_SEED,
        RESULTS_DIR,
        SAMPLE_RATE,
        TEST_SIZE,
        VERSIONS_DIR,
        WEIGHT_SEARCH_GRID,
    )
    from .features import extract_features
    from .triangulation import FourierTriangulator, extract_fourier_triangulation_vector
except (ImportError, ValueError):
    from config import (  # type: ignore
        AUTO_TUNE_WEIGHT,
        CFG,
        CLASSES,
        CV_FOLDS,
        DATA_DIR,
        FOURIER_WEIGHT,
        MODEL_DIR,
        MODEL_PATH,
        RANDOM_SEED,
        RESULTS_DIR,
        SAMPLE_RATE,
        TEST_SIZE,
        VERSIONS_DIR,
        WEIGHT_SEARCH_GRID,
    )
    from features import extract_features  # type: ignore
    from triangulation import FourierTriangulator, extract_fourier_triangulation_vector  # type: ignore


def load_dataset(
    data_dir: Path = DATA_DIR,
    sr: int = SAMPLE_RATE,
    classes: list[str] = CLASSES,
) -> tuple[list[np.ndarray], list[str]]:
    """
    Loads raw impulse waveform files with strict validation and error handling:
    wraps np.load() in try/except blocks to log and skip any corrupt, truncated, or 0-byte files.
    """
    waveforms: list[np.ndarray] = []
    labels: list[str] = []

    if data_dir.is_dir():
        for cls_name in classes:
            folder = data_dir / cls_name
            if not folder.is_dir():
                continue
            for fpath in sorted(folder.iterdir()):
                if fpath.name.startswith("."):
                    continue
                w: np.ndarray | None = None
                if fpath.suffix == ".npy":
                    try:
                        w_raw = np.load(fpath)
                        w = np.asarray(w_raw, dtype=np.float32).flatten()
                    except (ValueError, EOFError, OSError) as e:
                        print(f"[Warning] Corrupted or truncated sample skipped: {fpath} ({e})")
                        continue
                elif fpath.suffix in (".wav", ".flac"):
                    try:
                        import librosa
                        w_raw, _ = librosa.load(str(fpath), sr=sr, mono=True)
                        w = np.asarray(w_raw, dtype=np.float32).flatten()
                    except Exception as e:
                        print(f"[Warning] Could not load audio file: {fpath} ({e})")
                        continue

                if w is not None and len(w) >= 128:
                    # Sanitize any NaNs or Infs
                    w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
                    waveforms.append(w)
                    labels.append(cls_name)

    # Bootstrap physics-informed synthetic impulses if dataset has insufficient samples
    if len(waveforms) < 15:
        print("[Bootstrap] Insufficient recorded samples (<15). Synthesizing physics-informed bootstrap clips...")
        rng = np.random.default_rng(RANDOM_SEED)
        n_samples = int(0.14 * sr)
        t = np.arange(n_samples, dtype=np.float32) / sr
        onset_idx = n_samples // 4
        t_post = np.maximum(0.0, t - t[onset_idx])
        env = np.where(t >= t[onset_idx], np.exp(-t_post * 110.0), 0.0)

        for cls_name in classes:
            for _ in range(35):
                noise = rng.normal(0, 0.002, size=n_samples).astype(np.float32)
                if cls_name == "left_tap":
                    acoustic = env * (0.35 * np.sin(2 * np.pi * 1750 * t_post) + 0.25 * np.sin(2 * np.pi * 4200 * t_post))
                    ultra = 0.03 * np.sin(2 * np.pi * (19000 + 75 * env) * t) + 0.02 * np.sin(2 * np.pi * 20200 * t)
                    wave = acoustic + ultra + noise
                elif cls_name == "right_tap":
                    acoustic = env * (0.35 * np.sin(2 * np.pi * 1150 * t_post) + 0.25 * np.sin(2 * np.pi * 2850 * t_post))
                    ultra = 0.02 * np.sin(2 * np.pi * 19000 * t) + 0.03 * np.sin(2 * np.pi * (20200 + 75 * env) * t)
                    wave = acoustic + ultra + noise
                else:  # noise
                    wave = noise + (0.015 * rng.normal(0, 1, size=n_samples) if rng.random() > 0.5 else 0.0)
                waveforms.append(wave.astype(np.float32))
                labels.append(cls_name)

    return waveforms, labels


def build_candidate_models(seed: int = RANDOM_SEED) -> dict[str, Pipeline]:
    """Builds standard scikit-learn classifier candidate pipelines."""
    tr_cfg = CFG.get("training", {})
    m_cfg = tr_cfg.get("models", {})

    svm_cfg = m_cfg.get("SVM_RBF", {})
    svm_cw = svm_cfg.get("class_weight")
    if isinstance(svm_cw, dict):
        svm_cw = {int(k) if str(k).isdigit() else k: float(v) for k, v in svm_cw.items()}

    rf_cfg = m_cfg.get("RandomForest", {})
    et_cfg = m_cfg.get("ExtraTrees", {})

    return {
        "SVM_RBF": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(
                C=float(svm_cfg.get("C", 15.0)),
                gamma=svm_cfg.get("gamma", "scale"),
                kernel=svm_cfg.get("kernel", "rbf"),
                class_weight=svm_cw,
                probability=True,
                random_state=seed,
            )),
        ]),
        "RandomForest": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(
                n_estimators=int(rf_cfg.get("n_estimators", 250)),
                max_depth=rf_cfg.get("max_depth", 14),
                min_samples_split=int(rf_cfg.get("min_samples_split", 2)),
                class_weight=rf_cfg.get("class_weight", "balanced"),
                random_state=seed,
            )),
        ]),
        "ExtraTrees": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", ExtraTreesClassifier(
                n_estimators=int(et_cfg.get("n_estimators", 250)),
                max_depth=et_cfg.get("max_depth", None),
                min_samples_split=int(et_cfg.get("min_samples_split", 2)),
                class_weight=et_cfg.get("class_weight", "balanced"),
                random_state=seed,
            )),
        ]),
    }


def evaluate_hybrid_cv(
    ml_pipeline: Pipeline,
    X_ml: np.ndarray,
    X_fourier: np.ndarray,
    y: np.ndarray,
    weight_grid: list[float],
    n_splits: int = 5,
    seed: int = RANDOM_SEED,
) -> tuple[float, float, float, float]:
    """
    Performs out-of-fold cross validation fusing ML pipeline predictions with
    FourierTriangulator predictions over the search grid of fusion weights.
    Returns: (ml_macro_f1, fourier_macro_f1, best_hybrid_macro_f1, best_fourier_weight)
    """
    n_classes = len(np.unique(y))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof_ml = np.zeros((len(y), n_classes), dtype=np.float64)
    oof_fourier = np.zeros((len(y), n_classes), dtype=np.float64)

    for train_idx, val_idx in cv.split(X_ml, y):
        pipe_fold = clone(ml_pipeline)
        pipe_fold.fit(X_ml[train_idx], y[train_idx])
        tri_fold = FourierTriangulator().fit(X_fourier[train_idx], y[train_idx])

        oof_ml[val_idx] = pipe_fold.predict_proba(X_ml[val_idx])
        oof_fourier[val_idx] = tri_fold.predict_proba(X_fourier[val_idx])

    ml_f1 = float(f1_score(y, np.argmax(oof_ml, axis=1), average="macro", zero_division=0))
    fourier_f1 = float(f1_score(y, np.argmax(oof_fourier, axis=1), average="macro", zero_division=0))

    best_w = weight_grid[0]
    best_hyb_f1 = -1.0
    for w in weight_grid:
        p_hyb = (1.0 - w) * oof_ml + w * oof_fourier
        score = float(f1_score(y, np.argmax(p_hyb, axis=1), average="macro", zero_division=0))
        if score > best_hyb_f1:
            best_hyb_f1 = score
            best_w = float(w)

    return ml_f1, fourier_f1, best_hyb_f1, best_w


def train_and_benchmark(
    feature_set: str | None = None,
    fourier_weight: float | None = None,
) -> dict[str, Any]:
    """Trains, benchmarks, optimizes, and exports AcousticTap V3 production model."""
    print("=" * 70)
    print(" AcousticTap V3: Training Hybrid ML + Fourier Triangulation Architecture")
    print(f" Sample Rate: {SAMPLE_RATE} Hz | Carrier Isolation: 16 kHz LPF")
    print("=" * 70)

    mode = feature_set if feature_set is not None else CFG.get("features", {}).get("mode", "enhanced")
    if fourier_weight is not None:
        weight_grid = [float(fourier_weight)]
    elif AUTO_TUNE_WEIGHT:
        weight_grid = list(WEIGHT_SEARCH_GRID)
    else:
        weight_grid = [float(FOURIER_WEIGHT)]

    waveforms, str_labels = load_dataset(DATA_DIR, sr=SAMPLE_RATE, classes=CLASSES)
    if len(waveforms) == 0:
        raise ValueError(f"No valid audio samples found in {DATA_DIR}")

    label_to_idx = {name: idx for idx, name in enumerate(CLASSES)}
    valid_data = [(w, label_to_idx[lbl]) for w, lbl in zip(waveforms, str_labels) if lbl in label_to_idx]
    if len(valid_data) == 0:
        raise ValueError(f"No samples matched configured classes {CLASSES}")

    raw_waves = [d[0] for d in valid_data]
    y_raw = np.array([d[1] for d in valid_data], dtype=np.int64)

    # Deterministic feature extraction
    print(f"\nExtracting deterministic acoustic & Fourier features for {len(raw_waves)} samples...")
    X_ml_list = []
    X_four_list = []
    y_clean_list = []

    for idx, (w, label_idx) in enumerate(zip(raw_waves, y_raw)):
        f_ml = extract_features(w, sr=SAMPLE_RATE, feature_set=mode)
        f_four = extract_fourier_triangulation_vector(w, sr=SAMPLE_RATE)
        if not np.isnan(f_ml).any() and not np.isnan(f_four).any():
            X_ml_list.append(f_ml)
            X_four_list.append(f_four)
            y_clean_list.append(label_idx)

    X_ml = np.array(X_ml_list, dtype=np.float32)
    X_fourier = np.array(X_four_list, dtype=np.float32)
    y = np.array(y_clean_list, dtype=np.int64)

    print(f"-> ML Feature Matrix:      {X_ml.shape} (LPF @ 16 kHz)")
    print(f"-> Fourier Feature Matrix: {X_fourier.shape} (Spatial Resonance)")
    print(f"-> Class Distribution:     {[int(np.sum(y == i)) for i in range(len(CLASSES))]}")

    # Stratified Train/Holdout partition (80% train, 20% unseen test)
    min_count = int(np.min(np.unique(y, return_counts=True)[1]))
    n_splits = min(CV_FOLDS, max(2, min_count))

    train_idx, test_idx = train_test_split(
        np.arange(len(y)),
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_SEED,
    )

    X_ml_tr, X_ml_te = X_ml[train_idx], X_ml[test_idx]
    X_four_tr, X_four_te = X_fourier[train_idx], X_fourier[test_idx]
    y_tr, y_te = y[train_idx], y[test_idx]

    candidates = build_candidate_models(seed=RANDOM_SEED)
    param_grids = CFG.get("training", {}).get("param_grids", {})
    do_opt = bool(CFG.get("training", {}).get("optimize_hyperparameters", True))

    best_name = "ExtraTrees"
    best_hybrid_cv = -1.0
    best_opt_w = 0.50
    best_ml_pipe: Pipeline | None = None
    best_params: dict[str, Any] = {}
    leaderboard_entries: list[dict[str, Any]] = []

    print(f"\nBenchmarking {len(candidates)} Candidate Architectures with {n_splits}-Fold Stratified CV...")

    for name, pipeline in candidates.items():
        if do_opt and name in param_grids and len(param_grids[name]) > 0:
            grid = GridSearchCV(
                pipeline,
                param_grids[name],
                cv=n_splits,
                scoring="f1_macro",
                n_jobs=-1,
                error_score=0.0,
            )
            grid.fit(X_ml_tr, y_tr)
            tuned_pipeline = grid.best_estimator_
            tuned_params = grid.best_params_
        else:
            tuned_pipeline = pipeline
            tuned_params = {}

        ml_cv, fourier_cv, hybrid_cv, opt_w = evaluate_hybrid_cv(
            tuned_pipeline,
            X_ml_tr,
            X_four_tr,
            y_tr,
            weight_grid=weight_grid,
            n_splits=n_splits,
            seed=RANDOM_SEED,
        )

        # Evaluate on holdout split
        holdout_ml = clone(tuned_pipeline).fit(X_ml_tr, y_tr)
        holdout_tri = FourierTriangulator().fit(X_four_tr, y_tr)
        p_ml_te = holdout_ml.predict_proba(X_ml_te)
        p_four_te = holdout_tri.predict_proba(X_four_te)
        p_hyb_te = (1.0 - opt_w) * p_ml_te + opt_w * p_four_te
        preds_te = np.argmax(p_hyb_te, axis=1)

        holdout_acc = float(accuracy_score(y_te, preds_te))
        holdout_f1 = float(f1_score(y_te, preds_te, average="macro", zero_division=0))

        print(
            f"  {name:14s} | ML F1: {ml_cv:.4f} | FFT F1: {fourier_cv:.4f} | "
            f"Hybrid CV F1: {hybrid_cv:.4f} (opt_w={opt_w:.2f}) | Holdout F1: {holdout_f1:.4f}"
        )

        row = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model_name": f"Hybrid_{name}",
            "feature_set": mode,
            "train_samples": len(train_idx),
            "ml_cv_f1": round(ml_cv, 4),
            "fourier_cv_f1": round(fourier_cv, 4),
            "hybrid_cv_f1": round(hybrid_cv, 4),
            "holdout_acc": round(holdout_acc, 4),
            "holdout_f1": round(holdout_f1, 4),
            "fourier_weight": round(opt_w, 2),
        }
        leaderboard_entries.append(row)

        if hybrid_cv > best_hybrid_cv:
            best_name = name
            best_hybrid_cv = hybrid_cv
            best_ml_pipe = tuned_pipeline
            best_opt_w = opt_w
            best_params = tuned_params

    # Final Evaluation on Holdout Set for Report
    assert best_ml_pipe is not None
    win_pipe_holdout = clone(best_ml_pipe).fit(X_ml_tr, y_tr)
    win_tri_holdout = FourierTriangulator().fit(X_four_tr, y_tr)
    win_p_hyb = (1.0 - best_opt_w) * win_pipe_holdout.predict_proba(X_ml_te) + best_opt_w * win_tri_holdout.predict_proba(X_four_te)
    win_preds = np.argmax(win_p_hyb, axis=1)
    final_holdout_acc = float(accuracy_score(y_te, win_preds))
    final_holdout_f1 = float(f1_score(y_te, win_preds, average="macro", zero_division=0))

    # Safe classification report extraction avoiding KeyErrors
    report_dict = classification_report(
        y_te, win_preds, target_names=CLASSES, output_dict=True, zero_division=0
    )
    report_str = classification_report(
        y_te, win_preds, target_names=CLASSES, zero_division=0
    )
    cm = confusion_matrix(y_te, win_preds, labels=range(len(CLASSES)))

    print("\n" + "=" * 70)
    print(
        f" Winning Architecture: Hybrid_{best_name} "
        f"({round((1.0 - best_opt_w)*100)}% ML + {round(best_opt_w*100)}% Fourier)"
    )
    print(f" 5-Fold Cross-Validation Macro F1: {best_hybrid_cv:.4f}")
    print(f" Holdout Test Accuracy:            {final_holdout_acc * 100:.2f}%")
    print(f" Holdout Test Macro F1:            {final_holdout_f1:.4f}")
    print("=" * 70)
    print("\nHoldout Classification Report:\n" + report_str)
    print("Confusion Matrix:\n" + str(cm))

    # Log to leaderboard
    lb_path = MODEL_DIR / "leaderboard.csv"
    file_exists = lb_path.is_file() and lb_path.stat().st_size > 0
    with open(lb_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(leaderboard_entries[0].keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerows(leaderboard_entries)

    # Generate Markdown Test Report
    ts_now = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_md = (
        f"# AcousticTap V3 Hybrid Test Report ({ts_now})\n\n"
        f"- **Winning Architecture:** `Hybrid_{best_name}` ({int((1 - best_opt_w) * 100)}% ML + {int(best_opt_w * 100)}% Fourier Triangulation)\n"
        f"- **Best Tuned Parameters:** `{best_params}`\n"
        f"- **Dataset Partition:** {len(y)} samples ({len(train_idx)} train / {len(test_idx)} holdout)\n"
        f"- **Cross-Validation Macro F1:** `{best_hybrid_cv:.4f}`\n"
        f"- **Holdout Test Accuracy:** `{final_holdout_acc * 100:.2f}%`\n"
        f"- **Holdout Test Macro F1:** `{final_holdout_f1:.4f}`\n\n"
        f"## Holdout Classification Report\n```text\n{report_str}\n```\n\n"
        f"## Confusion Matrix (`{CLASSES}`)\n```text\n{cm}\n```\n"
    )
    (RESULTS_DIR / "latest_test_report.md").write_text(report_md, encoding="utf-8")
    (RESULTS_DIR / f"test_report_{ts_now}.md").write_text(report_md, encoding="utf-8")

    # Fit Production Model on 100% of the dataset
    final_ml = clone(best_ml_pipe).fit(X_ml, y)
    final_tri = FourierTriangulator().fit(X_fourier, y)

    bundle: dict[str, Any] = {
        "model": final_ml,
        "scaler": final_ml.named_steps.get("scaler") if hasattr(final_ml, "named_steps") else None,
        "feature_dim": int(X_ml.shape[1]),
        "fourier_dim": int(X_fourier.shape[1]),
        "classes": list(CLASSES),
        "fourier_weight": float(best_opt_w),
        "triangulator": final_tri.to_dict(),
        "model_name": f"Hybrid_{best_name}",
        "best_params": best_params,
        "feature_set": mode,
        "cv_f1": float(best_hybrid_cv),
        "test_f1": float(final_holdout_f1),
        "sample_rate": int(SAMPLE_RATE),
        "timestamp": ts_now,
    }

    joblib.dump(bundle, MODEL_PATH)
    version_file = VERSIONS_DIR / f"Hybrid_{best_name}_{mode}_{ts_now}.joblib"
    joblib.dump(bundle, version_file)

    print(f"\n[Artifact Saved] Production model saved to: {MODEL_PATH}")
    print(f"[Artifact Saved] Versioned copy saved to:    {version_file}")
    print(f"[Artifact Saved] Test report written to:     {RESULTS_DIR / 'latest_test_report.md'}")

    return bundle


def main() -> None:
    parser = argparse.ArgumentParser(description="Train AcousticTap V3 Hybrid ML + Fourier Model")
    parser.add_argument("--features", dest="features", choices=["baseline", "enhanced"], default=None)
    parser.add_argument("--fourier-weight", dest="fourier_weight", type=float, default=None)
    args = parser.parse_args()
    train_and_benchmark(feature_set=args.features, fourier_weight=args.fourier_weight)


if __name__ == "__main__":
    main()
