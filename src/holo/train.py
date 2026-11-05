import csv
import argparse
from datetime import datetime
import numpy as np
import joblib
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from .config import CFG, DATA_DIR, MODEL_DIR, VERSIONS_DIR, RESULTS_DIR, MODEL_PATH, CLASSES
from .features import extract_features
from .triangulation import extract_fourier_triangulation_vector, FourierTriangulator

def build_candidates_from_config():
    tr_cfg = CFG["training"]
    m_cfg = tr_cfg["models"]
    seed = int(tr_cfg["random_seed"])

    svm_cw = m_cfg["SVM_RBF"].get("class_weight")
    if isinstance(svm_cw, dict):
        svm_cw = {int(k): float(v) for k, v in svm_cw.items()}

    candidates = {
        "SVM_RBF": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(
                C=float(m_cfg["SVM_RBF"]["C"]),
                gamma=m_cfg["SVM_RBF"]["gamma"],
                kernel=m_cfg["SVM_RBF"].get("kernel", "rbf"),
                class_weight=svm_cw,
                probability=True,
                random_state=seed
            ))
        ]),
        "RandomForest": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(
                n_estimators=int(m_cfg["RandomForest"]["n_estimators"]),
                max_depth=m_cfg["RandomForest"]["max_depth"],
                min_samples_split=int(m_cfg["RandomForest"].get("min_samples_split", 2)),
                class_weight=m_cfg["RandomForest"].get("class_weight"),
                random_state=seed
            ))
        ]),
        "ExtraTrees": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", ExtraTreesClassifier(
                n_estimators=int(m_cfg["ExtraTrees"]["n_estimators"]),
                max_depth=m_cfg["ExtraTrees"]["max_depth"],
                min_samples_split=int(m_cfg["ExtraTrees"].get("min_samples_split", 2)),
                class_weight=m_cfg["ExtraTrees"].get("class_weight"),
                random_state=seed
            ))
        ]),
    }
    return candidates

def evaluate_hybrid_cv(ml_pipeline, X_ml, X_fourier, y, weight_grid, n_splits=5, seed=42):
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof_ml = np.zeros((len(y), len(np.unique(y))))
    oof_fourier = np.zeros((len(y), len(np.unique(y))))

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

def train_and_benchmark(feature_set: str | None = None, fourier_weight: float | None = None):
    tr_cfg = CFG["training"]
    hyb_cfg = CFG["hybrid_fusion"]
    mode = feature_set if feature_set is not None else CFG["features"]["mode"]
    cv_folds = int(tr_cfg["cv_folds"])
    seed = int(tr_cfg["random_seed"])
    do_opt = bool(tr_cfg.get("optimize_hyperparameters", True))

    if fourier_weight is not None:
        weight_grid = [float(fourier_weight)]
    elif hyb_cfg.get("auto_tune_weight", True):
        weight_grid = [float(w) for w in hyb_cfg["weight_search_grid"]]
    else:
        weight_grid = [float(hyb_cfg["fourier_weight"])]

    X_ml, X_fourier, y = [], [], []
    for idx, label in enumerate(CLASSES):
        folder = DATA_DIR / label
        if not folder.exists():
            continue
        for file in sorted(folder.glob("*.npy")):
            waveform = np.load(file)
            X_ml.append(extract_features(waveform, feature_set=mode))
            X_fourier.append(extract_fourier_triangulation_vector(waveform))
            y.append(idx)

    X_ml, X_fourier, y = np.array(X_ml), np.array(X_fourier), np.array(y)
    if len(np.unique(y)) < 2:
        raise ValueError("Collect samples for at least 2 classes inside ./dataset before training!")

    print(f"\n[TRAIN] Loaded {len(X)} total samples across {len(np.unique(y))} classes.")
    print(f"Feature vector dimension: {X.shape[1]} ({feature_set} mode)\n")

    # Strict isolation: 80% train / 20% holdout test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    print(f"Dataset split: {len(X_train)} training samples | {len(X_test)} holdout test samples\n")

    print(f"\n[TRAIN HYBRID] Total Samples: {len(y)} | Train: {len(train_idx)} | Holdout: {len(test_idx)}")
    print(f"ML Feature Dim: {X_ml.shape[1]} | Fourier Dim: {X_fourier.shape[1]}")
    print(f"Hyperparameter Tuning: {'ENABLED' if do_opt else 'DISABLED'} | Weight Grid: {weight_grid}\n")

    candidates = build_candidates_from_config()
    param_grids = tr_cfg.get("param_grids", {})

    VERSIONS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    leaderboard_path = MODEL_DIR / "leaderboard.csv"
    write_header = not leaderboard_path.exists()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    best_name, best_hybrid_cv, best_ml_pipe, best_opt_w, best_params = None, -1.0, None, 0.5, {}

    with open(leaderboard_path, mode="a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        if write_header:
            writer.writerow([
                "timestamp", "model_name", "feature_set", "num_samples",
                "ml_only_cv_f1", "fourier_only_cv_f1", "hybrid_cv_f1", "holdout_hybrid_f1",
                "optimal_fourier_weight", "saved_path"
            ])

        for name, pipeline in candidates.items():
            # Problem 1 Fix: Cross-validation runs strictly on X_train / y_train
            cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="f1_macro")
            cv_mean = float(np.mean(cv_scores))
            cv_std = float(np.std(cv_scores))

            # Fit candidate strictly on training split and evaluate unseen holdout
            pipeline.fit(X_train, y_train)
            test_preds = pipeline.predict(X_test)
            holdout_f1 = float(f1_score(y_test, test_preds, average="macro"))

            version_filename = f"{name}_{feature_set}_{timestamp}.joblib"
            version_path = versions_dir / version_filename

            joblib.dump({
                "model": full_pipe,
                "triangulator": tri_full,
                "fourier_weight": opt_w,
                "model_name": f"Hybrid_{name}",
                "best_params": tuned_params,
                "feature_set": mode,
                "classes": CLASSES,
                "cv_f1": cv_mean,
                "holdout_f1": holdout_f1
            }, version_path)

            writer.writerow([
                timestamp, name, feature_set, len(X_train),
                f"{cv_mean:.4f}", f"{cv_std:.4f}", f"{holdout_f1:.4f}",
                str(version_path.relative_to(MODEL_DIR.parent))
            ])

            print(f"  {name:14s} | Train 5-Fold F1: {cv_mean:.4f} (+/- {cv_std:.4f}) | Holdout F1: {holdout_f1:.4f}")
            print(f"                 -> Saved version: models/versions/{version_filename}")

            if hybrid_cv > best_hybrid_cv:
                best_name, best_hybrid_cv, best_ml_pipe, best_opt_w, best_params = (
                    name, hybrid_cv, tuned_pipeline, opt_w, tuned_params
                )

    # Final holdout evaluation of winning pipeline
    best_preds = best_pipeline.predict(X_test)
    print(f"\n--> Winner: {best_name} (Train 5-Fold Macro F1: {best_cv_score:.4f})")
    print("\nHoldout Test Set Classification Report (Evaluated on untouched test split):")
    print(classification_report(y_test, best_preds, target_names=CLASSES))
    print("Confusion Matrix (Rows = Actual, Columns = Predicted):")
    print(cm)

    report_md_path = RESULTS_DIR / f"test_report_{timestamp}.md"
    latest_md_path = RESULTS_DIR / "latest_test_report.md"
    md_content = (
        f"# AcousticTap Hybrid Test Report ({timestamp})\n\n"
        f"- **Winning Architecture:** `Hybrid_{best_name}` ({int((1-best_opt_w)*100)}% ML + {int(best_opt_w*100)}% Fourier Transform Triangulation)\n"
        f"- **Best Tuned Hyperparameters:** `{best_params}`\n"
        f"- **Total Dataset Samples:** {len(y)} ({len(train_idx)} train / {len(test_idx)} holdout)\n"
        f"- **5-Fold Cross-Validation Macro F1:** `{best_hybrid_cv:.4f}`\n"
        f"- **Holdout Test Accuracy:** `{holdout_acc*100:.2f}%`\n\n"
        f"## Holdout Classification Report\n```text\n{report_str}\n```\n\n"
        f"## Confusion Matrix (`{CLASSES}`)\n```text\n{cm}\n```\n"
    )
    report_md_path.write_text(md_content)
    latest_md_path.write_text(md_content)

    # Production model refit on full (X, y) for deployment binary
    best_pipeline.fit(X, y)
    joblib.dump({
        "model": final_ml,
        "triangulator": final_tri,
        "fourier_weight": best_opt_w,
        "model_name": f"Hybrid_{best_name}",
        "best_params": best_params,
        "feature_set": mode,
        "classes": CLASSES,
        "cv_f1": best_hybrid_cv,
        "timestamp": timestamp
    }, MODEL_PATH)

    print(f"\nSaved production model -> {MODEL_PATH}")
    print(f"Saved test report      -> {latest_md_path}")

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train and optimize Hybrid AcousticTap models via config/config.yaml")
    parser.add_argument("--features", "--feature-set", dest="features", choices=["baseline", "enhanced"], default=None)
    parser.add_argument("--fourier-weight", type=float, default=None)
    return parser

if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    train_and_benchmark(args.features, args.fourier_weight)
