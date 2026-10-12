"""Regenerate notebooks/train_advisor_model.ipynb from the repo sources.

    python scripts/build_notebook.py

The notebook embeds advisor/data.py verbatim (via %writefile) so the data generator the
notebook uses can never drift from the one the app and tests use.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "notebooks" / "train_advisor_model.ipynb"
DATA_PY = (ROOT / "advisor" / "data.py").read_text()

cells: list[dict] = []


def md(text: str) -> None:
    cells.append(
        {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(True)}
    )


def code(text: str) -> None:
    cells.append(
        {
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": text.strip("\n").splitlines(True),
        }
    )


md(
    """
# Academic Advisor: model training (CSC 303)

This notebook trains the model behind the Academic Advisor web app. It runs end to end on
Google Colab (Runtime > Run all) and produces **`advisor_model.pkl`**, which you place in the
`models/` folder of the repo and serve with Flask.

**Pipeline:** synthetic student records -> exploratory look -> compare 3 models with
cross-validation -> evaluate the winner on a hold-out set -> save the `.pkl` bundle -> download.

> The data is synthetic (generated from plausible rules plus noise). It demonstrates the
> workflow; for real use, replace it with anonymised records from your department.
"""
)

md("## 1. Setup")
code(
    r"""
import sys, sklearn, pandas as pd, numpy as np
print("Python", sys.version.split()[0], "| scikit-learn", sklearn.__version__, "| pandas", pd.__version__)
"""
)
code(
    """
!mkdir -p advisor models data
!touch advisor/__init__.py
"""
)

md("## 2. Data generator\nThis cell writes the same `advisor/data.py` used by the web app.")
code("%%writefile advisor/data.py\n" + DATA_PY)

md("## 3. Generate and inspect the dataset")
code(
    r"""
import matplotlib.pyplot as plt
from advisor.data import FEATURES, TARGET, CLASSES, generate_dataset

df = generate_dataset(n=4000, seed=42)
df.to_csv("data/students.csv", index=False)
print(df.shape)
df.head()
"""
)
code(
    r"""
print(df[TARGET].value_counts().reindex(CLASSES))
df.describe().round(2).T[["mean", "std", "min", "max"]]
"""
)
code(
    r"""
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
df[TARGET].value_counts().reindex(CLASSES).plot.bar(ax=ax[0], color="#2a4bd7")
ax[0].set_title("Next-semester standing (target)")
ax[0].set_xlabel(""); ax[0].tick_params(axis="x", rotation=0)

corr = df[FEATURES].corr()
im = ax[1].imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
ax[1].set_xticks(range(len(FEATURES)), FEATURES, rotation=90)
ax[1].set_yticks(range(len(FEATURES)), FEATURES)
ax[1].set_title("Feature correlations")
fig.colorbar(im, ax=ax[1], fraction=0.046)
plt.tight_layout(); plt.show()
"""
)

md("## 4. Compare candidate models (5-fold cross-validation)")
code(
    r"""
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X, y = df[FEATURES], df[TARGET]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

def rf():  # must match advisor/model.py build_model()
    return RandomForestClassifier(n_estimators=200, min_samples_leaf=8,
                                  class_weight="balanced", random_state=42, n_jobs=-1)

candidates = {
    "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")),
    "Random Forest": rf(),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
rows = []
for name, model in candidates.items():
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1_macro")
    rows.append({"model": name, "macro_f1_mean": scores.mean().round(4), "macro_f1_std": scores.std().round(4)})
pd.DataFrame(rows).sort_values("macro_f1_mean", ascending=False)
"""
)
md(
    """
The scores are close, so the **Random Forest** is used: it handles non-linear interactions, needs no
feature scaling, exposes feature importances for the app, and stays small once compressed.
"""
)

md("## 5. Train the final model and evaluate on the hold-out set")
code(
    r"""
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report, f1_score

model = rf().fit(X_train, y_train)
pred = model.predict(X_test)
print(classification_report(y_test, pred, labels=CLASSES))

fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
ConfusionMatrixDisplay.from_predictions(y_test, pred, labels=CLASSES, ax=ax[0], cmap="Blues", colorbar=False)
ax[0].set_title("Confusion matrix (hold-out)")
imp = pd.Series(model.feature_importances_, index=FEATURES).sort_values()
imp.plot.barh(ax=ax[1], color="#2a4bd7"); ax[1].set_title("Feature importance")
plt.tight_layout(); plt.show()
"""
)

md("## 6. Save the `.pkl` bundle and verify it reloads")
code(
    r"""
import joblib

bundle = {
    "model": model,
    "feature_names": list(FEATURES),
    "classes": [str(c) for c in model.classes_],
    "metrics": {
        "accuracy": round(float(accuracy_score(y_test, pred)), 4),
        "macro_f1": round(float(f1_score(y_test, pred, average="macro")), 4),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
    },
    "sklearn_version": sklearn.__version__,
}
joblib.dump(bundle, "models/advisor_model.pkl", compress=3)

# Sanity check: reload from disk and predict for one hypothetical student.
loaded = joblib.load("models/advisor_model.pkl")
student = pd.DataFrame([{
    "cgpa": 3.2, "last_gpa": 2.9, "attendance_pct": 70, "study_hours_week": 8,
    "assignment_pct": 75, "carryovers": 1, "credit_load": 24, "sleep_hours": 6, "extracurricular_hours": 10,
}])[loaded["feature_names"]]
print({c: round(float(p), 3) for c, p in zip(loaded["classes"], loaded["model"].predict_proba(student)[0])})
print("Saved models/advisor_model.pkl", loaded["metrics"])
"""
)

md("## 7. Download the model")
code(
    r"""
try:
    from google.colab import files
    files.download("models/advisor_model.pkl")
except ImportError:
    print("Not running on Colab; the file is at models/advisor_model.pkl")
"""
)
md(
    """
**Next:** copy the downloaded `advisor_model.pkl` into the repo's `models/` folder, run
`python app.py`, and test it in the browser. If loading fails with a scikit-learn version error,
install the version printed in section 1 locally (`pip install scikit-learn==<version>`).
"""
)

notebook = {
    "cells": [{**c, "id": f"cell{i:02d}"} for i, c in enumerate(cells)],
    "metadata": {
        "colab": {"provenance": [], "name": "train_advisor_model.ipynb"},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(notebook, indent=1) + "\n")
print(f"Wrote {OUT.relative_to(ROOT)} ({len(cells)} cells)")
