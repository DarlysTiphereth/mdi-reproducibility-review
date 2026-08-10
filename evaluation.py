"""Anomaly detection and classifier comparisons required by Reviewer 2."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (balanced_accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from config import Config


def evaluate_isolation_forest(fleet: pd.DataFrame, cfg: Config):
    model = IsolationForest(contamination=cfg.anomaly_fraction, random_state=cfg.seed)
    raw_prediction = model.fit_predict(fleet[["latitude", "longitude"]])
    out = fleet.copy()
    out["predicted_anomaly"] = (raw_prediction == -1).astype(int)
    y_true = out.is_injected_anomaly.to_numpy()
    y_pred = out.predicted_anomaly.to_numpy()
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    metrics = pd.DataFrame([{
        "model": "Isolation Forest",
        "n": len(out), "positive_definition": "injected anomaly",
        "true_negative": int(tn), "false_positive": int(fp),
        "false_negative": int(fn), "true_positive": int(tp),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }])
    return out, metrics


def _models(seed: int):
    return {
        "SVM (linear)": Pipeline([("scale", StandardScaler()),
                                  ("model", SVC(kernel="linear", C=1.0))]),
        "Logistic regression": Pipeline([("scale", StandardScaler()),
                                          ("model", LogisticRegression(C=1.0, max_iter=1000,
                                                                       random_state=seed))]),
        "Decision tree": DecisionTreeClassifier(random_state=seed),
        "Random forest": RandomForestClassifier(n_estimators=100, random_state=seed),
    }


def compare_classifiers(zones: pd.DataFrame, cfg: Config):
    X = zones[["P", "I", "O"]].to_numpy()
    y = zones["is_critical"].to_numpy()
    cv = StratifiedKFold(n_splits=cfg.cv_folds, shuffle=True, random_state=cfg.seed)
    fold_rows, confusion_rows = [], []
    for model_name, model in _models(cfg.seed).items():
        for fold, (train_idx, test_idx) in enumerate(cv.split(X, y), start=1):
            model.fit(X[train_idx], y[train_idx])
            pred = model.predict(X[test_idx])
            tn, fp, fn, tp = confusion_matrix(y[test_idx], pred, labels=[0, 1]).ravel()
            fold_rows.append({
                "model": model_name, "fold": fold,
                "n_train": len(train_idx), "n_test": len(test_idx),
                "precision_critical": precision_score(y[test_idx], pred, zero_division=0),
                "recall_critical": recall_score(y[test_idx], pred, zero_division=0),
                "f1_critical": f1_score(y[test_idx], pred, zero_division=0),
                "macro_f1": f1_score(y[test_idx], pred, average="macro", zero_division=0),
                "balanced_accuracy": balanced_accuracy_score(y[test_idx], pred),
            })
            confusion_rows.append({"model": model_name, "fold": fold,
                                   "tn": int(tn), "fp": int(fp),
                                   "fn": int(fn), "tp": int(tp)})
    folds = pd.DataFrame(fold_rows)
    summary = folds.groupby("model", as_index=False).agg(
        folds=("fold", "count"),
        precision_mean=("precision_critical", "mean"),
        precision_std=("precision_critical", "std"),
        recall_mean=("recall_critical", "mean"),
        recall_std=("recall_critical", "std"),
        f1_critical_mean=("f1_critical", "mean"),
        f1_critical_std=("f1_critical", "std"),
        macro_f1_mean=("macro_f1", "mean"),
        macro_f1_std=("macro_f1", "std"),
        balanced_accuracy_mean=("balanced_accuracy", "mean"),
        balanced_accuracy_std=("balanced_accuracy", "std"),
    )
    return folds, summary, pd.DataFrame(confusion_rows)

