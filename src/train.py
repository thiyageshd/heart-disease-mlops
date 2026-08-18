"""
train.py
--------
Phase 2: feature engineering, model development, and experiment tracking.

What this does:
  1. Loads and cleans data via preprocess.py (shared with the API for zero skew).
  2. Stratified train/test split (respects the mild class imbalance).
  3. For each candidate model (Logistic Regression, Random Forest):
       - wraps preprocessor + classifier in one Pipeline
       - tunes hyper-parameters with stratified k-fold GridSearchCV (scoring=ROC-AUC)
       - evaluates on the held-out test set (accuracy, precision, recall, ROC-AUC)
       - logs params, metrics, ROC curve, and confusion matrix to MLflow
  4. Selects the best model by test ROC-AUC and saves the full fitted pipeline
     to models/heart_pipeline.joblib (+ logs it as the registered MLflow model).

Run:
    python src/train.py
Then inspect:
    mlflow ui        # -> http://127.0.0.1:5000
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline

from preprocess import build_preprocessor, load_clean

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "heart_disease_raw.csv"
MODELS = ROOT / "models"
FIGS = ROOT / "report" / "figures"
MODELS.mkdir(exist_ok=True)
FIGS.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

# ------------------------------------------------------------------ model zoo
def candidate_models() -> dict[str, tuple[object, dict]]:
    """Return {name: (estimator, param_grid)}. Grids are small and sensible."""
    return {
        "logistic_regression": (
            LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
            {
                "clf__C": [0.05, 0.1, 0.5, 1.0, 5.0],
                "clf__class_weight": [None, "balanced"],
            },
        ),
        "random_forest": (
            RandomForestClassifier(random_state=RANDOM_STATE),
            {
                "clf__n_estimators": [200, 400],
                "clf__max_depth": [None, 4, 6, 8],
                "clf__min_samples_leaf": [1, 2, 4],
                "clf__class_weight": [None, "balanced"],
            },
        ),
    }


def make_pipeline(estimator) -> Pipeline:
    return Pipeline(steps=[
        ("preprocess", build_preprocessor()),
        ("clf", estimator),
    ])


def evaluate(pipe, X_test, y_test) -> dict:
    y_pred = pipe.predict(X_test)
    # probability of the positive class for ROC-AUC
    y_proba = pipe.predict_proba(X_test)[:, 1]
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }, y_pred, y_proba


def log_plots(name, pipe, X_test, y_test, y_pred):
    # ROC curve
    fig, ax = plt.subplots(figsize=(5.5, 5))
    RocCurveDisplay.from_estimator(pipe, X_test, y_test, ax=ax, name=name)
    ax.plot([0, 1], [0, 1], "--", color="grey", linewidth=1)
    ax.set_title(f"ROC curve — {name}")
    roc_path = FIGS / f"roc_{name}.png"
    fig.tight_layout(); fig.savefig(roc_path, dpi=130); plt.close(fig)

    # Confusion matrix
    fig, ax = plt.subplots(figsize=(4.8, 4.4))
    cm = confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay(cm, display_labels=["No disease", "Disease"]).plot(
        ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Confusion matrix — {name}")
    cm_path = FIGS / f"cm_{name}.png"
    fig.tight_layout(); fig.savefig(cm_path, dpi=130); plt.close(fig)

    return roc_path, cm_path


def main():
    # SQLite tracking backend (modern MLflow recommendation; supports the full
    # feature set incl. the model registry). Artifacts land under ./mlartifacts.
    mlflow.set_tracking_uri(f"sqlite:///{ROOT / 'mlflow.db'}")
    os.makedirs(ROOT / "mlartifacts", exist_ok=True)
    mlflow.set_experiment("heart-disease-classification")

    X, y, clean_report = load_clean(str(DATA))
    print(f"Loaded X={X.shape}, target balance={y.value_counts().to_dict()}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
    print(f"Train={X_train.shape[0]} Test={X_test.shape[0]}")

    results = {}
    for name, (estimator, grid) in candidate_models().items():
        print(f"\n=== Tuning {name} ===")
        with mlflow.start_run(run_name=name):
            pipe = make_pipeline(estimator)
            search = GridSearchCV(pipe, grid, scoring="roc_auc", cv=CV, n_jobs=-1)
            search.fit(X_train, y_train)
            best = search.best_estimator_

            metrics, y_pred, _ = evaluate(best, X_test, y_test)
            cv_best_auc = float(search.best_score_)

            # ---- MLflow logging
            mlflow.log_params({k.replace("clf__", ""): v
                               for k, v in search.best_params_.items()})
            mlflow.log_param("model_type", name)
            mlflow.log_param("cv_folds", CV.get_n_splits())
            mlflow.log_metric("cv_roc_auc", cv_best_auc)
            for m, v in metrics.items():
                mlflow.log_metric(f"test_{m}", v)

            roc_path, cm_path = log_plots(name, best, X_test, y_test, y_pred)
            mlflow.log_artifact(str(roc_path), artifact_path="plots")
            mlflow.log_artifact(str(cm_path), artifact_path="plots")

            # classification report as an artifact
            rep_txt = classification_report(
                y_test, y_pred, target_names=["No disease", "Disease"])
            rep_path = FIGS.parent / f"classification_report_{name}.txt"
            rep_path.write_text(rep_txt)
            mlflow.log_artifact(str(rep_path))

            mlflow.sklearn.log_model(
                best, artifact_path="model",
                serialization_format="cloudpickle",
            )

            print(f"  CV ROC-AUC: {cv_best_auc:.4f} | best params: {search.best_params_}")
            print("  Test:", {k: round(v, 4) for k, v in metrics.items()})

            results[name] = {
                "estimator": best,
                "metrics": metrics,
                "cv_roc_auc": cv_best_auc,
                "best_params": search.best_params_,
            }

    # ------------------------------------------------------------ pick winner
    winner = max(results, key=lambda n: results[n]["metrics"]["roc_auc"])
    best_pipe = results[winner]["estimator"]
    print(f"\n>>> Best model: {winner} "
          f"(test ROC-AUC={results[winner]['metrics']['roc_auc']:.4f})")

    # Save the full fitted pipeline (preprocess + clf) for the API.
    model_path = MODELS / "heart_pipeline.joblib"
    joblib.dump(best_pipe, model_path)

    # Save a metadata sidecar the API/report can read.
    meta = {
        "winner": winner,
        "metrics": {k: round(v, 4) for k, v in results[winner]["metrics"].items()},
        "best_params": {k.replace("clf__", ""): v
                        for k, v in results[winner]["best_params"].items()},
        "feature_order": list(X.columns),
        "clean_notes": clean_report.notes,
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0]),
    }
    (MODELS / "model_metadata.json").write_text(json.dumps(meta, indent=2))

    print(f"Saved pipeline -> {model_path}")
    print(f"Saved metadata -> {MODELS / 'model_metadata.json'}")

    # Comparison table for the report
    print("\n--- Model comparison (test set) ---")
    print(f"{'model':22} {'acc':>6} {'prec':>6} {'rec':>6} {'f1':>6} {'auc':>6}")
    for n, r in results.items():
        m = r["metrics"]
        print(f"{n:22} {m['accuracy']:.3f} {m['precision']:.3f} "
              f"{m['recall']:.3f} {m['f1']:.3f} {m['roc_auc']:.3f}")


if __name__ == "__main__":
    main()
