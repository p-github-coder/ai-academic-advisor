"""Training, evaluation and (de)serialisation of the advisor models.

The artefact written to disk (`models/advisor_model.pkl`) is a dict "bundle":

    {
        "model":           fitted classifier (has predict_proba) -> standing,
        "regressor":       fitted regressor -> expected semester GPA,
        "feature_names":   list[str]  - column order both models expect,
        "classes":         list[str]  - label order of predict_proba,
        "metrics":         dict       - hold-out accuracy, macro-F1, MAE, R2,
        "sklearn_version": str        - version used for training,
    }

The Colab notebook writes the same structure, so either source of the .pkl works.
"""

from __future__ import annotations

import joblib
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

from advisor.data import FEATURES, TARGET_CLASS, TARGET_GPA

DEFAULT_MODEL_PATH = "models/advisor_model.pkl"


def build_model(seed: int = 42) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=200,
        min_samples_leaf=8,
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
    )


def build_regressor(seed: int = 42) -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=150,
        min_samples_leaf=8,
        random_state=seed,
        n_jobs=-1,
    )


def split(df: pd.DataFrame, seed: int = 42):
    """The one train/test split used everywhere (also in the Colab notebook)."""
    return train_test_split(
        df[FEATURES],
        df[[TARGET_CLASS, TARGET_GPA]],
        test_size=0.2,
        stratify=df[TARGET_CLASS],
        random_state=seed,
    )


def train_bundle(df: pd.DataFrame, seed: int = 42) -> dict:
    """Fit both models on `df`, evaluate on a stratified hold-out split, return a bundle."""
    X_train, X_test, y_train, y_test = split(df, seed)
    clf = build_model(seed).fit(X_train, y_train[TARGET_CLASS])
    reg = build_regressor(seed).fit(X_train, y_train[TARGET_GPA])

    pred = clf.predict(X_test)
    gpa_pred = reg.predict(X_test)
    metrics = {
        "accuracy": round(float(accuracy_score(y_test[TARGET_CLASS], pred)), 4),
        "macro_f1": round(float(f1_score(y_test[TARGET_CLASS], pred, average="macro")), 4),
        "gpa_mae": round(float(mean_absolute_error(y_test[TARGET_GPA], gpa_pred)), 4),
        "gpa_r2": round(float(r2_score(y_test[TARGET_GPA], gpa_pred)), 4),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
    }
    return {
        "model": clf,
        "regressor": reg,
        "feature_names": list(FEATURES),
        "classes": [str(c) for c in clf.classes_],
        "metrics": metrics,
        "sklearn_version": sklearn.__version__,
    }


def save_bundle(bundle: dict, path: str = DEFAULT_MODEL_PATH) -> None:
    joblib.dump(bundle, path, compress=3)


def load_bundle(path: str = DEFAULT_MODEL_PATH) -> dict:
    bundle = joblib.load(path)
    missing = {"model", "feature_names", "classes"} - set(bundle)
    if missing:
        raise ValueError(f"{path} is not a valid advisor bundle (missing {sorted(missing)})")
    return bundle
