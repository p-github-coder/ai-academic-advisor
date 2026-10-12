"""Training, evaluation and (de)serialisation of the standing-prediction model.

The artefact written to disk (`models/advisor_model.pkl`) is a dict "bundle":

    {
        "model":           fitted scikit-learn estimator (has predict_proba),
        "feature_names":   list[str]  - column order the model expects,
        "classes":         list[str]  - label order of predict_proba,
        "metrics":         dict       - hold-out accuracy / macro-F1,
        "sklearn_version": str        - version used for training,
    }

The Colab notebook writes the same structure, so either source of the .pkl works.
"""

from __future__ import annotations

import joblib
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from advisor.data import FEATURES, TARGET

DEFAULT_MODEL_PATH = "models/advisor_model.pkl"


def build_model(seed: int = 42) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=200,
        min_samples_leaf=8,
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
    )


def train_bundle(df: pd.DataFrame, seed: int = 42) -> dict:
    """Fit the model on `df`, evaluate on a stratified hold-out split, return a bundle."""
    X, y = df[FEATURES], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=seed
    )
    model = build_model(seed).fit(X_train, y_train)
    pred = model.predict(X_test)
    metrics = {
        "accuracy": round(float(accuracy_score(y_test, pred)), 4),
        "macro_f1": round(float(f1_score(y_test, pred, average="macro")), 4),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
    }
    return {
        "model": model,
        "feature_names": list(FEATURES),
        "classes": [str(c) for c in model.classes_],
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
