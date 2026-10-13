"""Regenerate notebooks/train_advisor_model.ipynb from the repo sources.

    python scripts/build_notebook.py

The notebook embeds advisor/data.py and advisor/model.py verbatim (via %%writefile), so the
code Colab trains with can never drift from the code the web app and tests use.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "notebooks" / "train_advisor_model.ipynb"
DATA_PY = (ROOT / "advisor" / "data.py").read_text()
MODEL_PY = (ROOT / "advisor" / "model.py").read_text()

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

This notebook trains the models behind the Academic Advisor web app. It runs end to end on
Google Colab (**Runtime > Run all**) and produces **`advisor_model.pkl`**, which you place in the
`models/` folder of the repo and serve with Flask.

**Pipeline:** synthetic student records -> exploratory look -> compare candidate models with
cross-validation -> train the final classifier (standing) and regressor (semester GPA) ->
evaluate on a hold-out set -> save the `.pkl` bundle -> download.

**What the models learn from:** academic level, CGPA, last semester GPA, average course score,
credit load, average difficulty and confidence, study hours and days, attendance, assignment
habits, repeat courses and the gap to the target CGPA. Everything else the student enters
(study method, challenges, exam date...) drives the rule-based part of the advisor instead.

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

md(
    "## 2. Project code\n"
    "These two cells write the same `advisor/data.py` and `advisor/model.py` used by the web app."
)
code("%%writefile advisor/data.py\n" + DATA_PY)
code("%%writefile advisor/model.py\n" + MODEL_PY)

md("## 3. Generate and inspect the dataset")
code(
    r"""
import matplotlib.pyplot as plt
from advisor.data import FEATURES, TARGET_CLASS, TARGET_GPA, CLASSES, generate_dataset

df = generate_dataset(n=4000, seed=42)
df.to_csv("data/students.csv", index=False)
print(df.shape)
df.head()
"""
)
code(
    r"""
print(df[TARGET_CLASS].value_counts().reindex(CLASSES))
df[FEATURES + [TARGET_GPA]].describe().round(2).T[["mean", "std", "min", "max"]]
"""
)
code(
    r"""
fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))
df[TARGET_CLASS].value_counts().reindex(CLASSES).plot.bar(ax=ax[0], color="#d6283c")
ax[0].set_title("Next-semester standing (class target)")
ax[0].set_xlabel(""); ax[0].tick_params(axis="x", rotation=0)

ax[1].hist(df[TARGET_GPA], bins=30, color="#0f172a")
ax[1].set_title("Next-semester GPA (regression target)"); ax[1].set_xlabel("GPA")

corr = df[FEATURES + [TARGET_GPA]].corr()
im = ax[2].imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
ax[2].set_xticks(range(len(corr)), corr.columns, rotation=90, fontsize=7)
ax[2].set_yticks(range(len(corr)), corr.columns, fontsize=7)
ax[2].set_title("Correlations")
fig.colorbar(im, ax=ax[2], fraction=0.046)
plt.tight_layout(); plt.show()
"""
)

md("## 4. Compare candidate models (5-fold cross-validation)")
code(
    r"""
from sklearn.ensemble import (GradientBoostingClassifier, GradientBoostingRegressor,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from advisor.model import build_model, build_regressor, save_bundle, split, train_bundle

# The same split the final model uses (and the web app's training code uses).
X_train, X_test, y_train, y_test = split(df, seed=42)

classifiers = {
    "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced")),
    "Random Forest (chosen)": build_model(),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
}
rows = []
for name, model in classifiers.items():
    s = cross_val_score(model, X_train, y_train[TARGET_CLASS],
                        cv=StratifiedKFold(5, shuffle=True, random_state=42), scoring="f1_macro")
    rows.append({"task": "standing (classifier)", "model": name, "score": round(s.mean(), 4), "metric": "macro-F1"})

regressors = {
    "Ridge Regression": make_pipeline(StandardScaler(), Ridge()),
    "Random Forest (chosen)": build_regressor(),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42),
}
for name, model in regressors.items():
    s = cross_val_score(model, X_train, y_train[TARGET_GPA],
                        cv=KFold(5, shuffle=True, random_state=42), scoring="neg_mean_absolute_error")
    rows.append({"task": "semester GPA (regressor)", "model": name, "score": round(-s.mean(), 4), "metric": "MAE (lower is better)"})
pd.DataFrame(rows)
"""
)
md(
    """
The scores are close. The **Random Forest** is used for both tasks: it captures non-linear
interactions (for example, heavy credit load only hurts when study time is low), needs no feature
scaling, exposes feature importances for the app, and stays small once compressed. The classifier
gives the standing label and its probabilities; the regressor gives the expected semester GPA that
feeds the CGPA projection in the app.
"""
)

md("## 5. Train the final models and evaluate on the hold-out set")
code(
    r"""
import sklearn
from sklearn.metrics import ConfusionMatrixDisplay, classification_report

bundle = train_bundle(df, seed=42)
model, regressor = bundle["model"], bundle["regressor"]
pred = model.predict(X_test)
gpa_pred = regressor.predict(X_test)

print(classification_report(y_test[TARGET_CLASS], pred, labels=CLASSES))
print({k: v for k, v in bundle["metrics"].items()})

fig, ax = plt.subplots(1, 3, figsize=(17, 4.6))
ConfusionMatrixDisplay.from_predictions(y_test[TARGET_CLASS], pred, labels=CLASSES, ax=ax[0],
                                        cmap="Blues", colorbar=False)
ax[0].set_title("Standing: confusion matrix (hold-out)")

ax[1].scatter(y_test[TARGET_GPA], gpa_pred, s=8, alpha=0.5, color="#d6283c")
ax[1].plot([0, 5], [0, 5], color="#0f172a", lw=1)
ax[1].set_xlabel("Actual GPA"); ax[1].set_ylabel("Predicted GPA"); ax[1].set_title("Semester GPA: predicted vs actual")

imp = pd.Series(model.feature_importances_, index=FEATURES).sort_values()
imp.plot.barh(ax=ax[2], color="#0f172a"); ax[2].set_title("What the classifier relies on")
plt.tight_layout(); plt.show()
"""
)

md("## 6. Save the `.pkl` bundle and verify it reloads")
code(
    r"""
import joblib

save_bundle(bundle, "models/advisor_model.pkl")

# Sanity check: reload from disk and predict for one hypothetical student.
loaded = joblib.load("models/advisor_model.pkl")
student = pd.DataFrame([{
    "level": 3, "cgpa": 3.6, "last_gpa": 3.4, "avg_score": 63, "credit_load": 18,
    "avg_difficulty": 2.4, "avg_confidence": 2.8, "study_hours_day": 3, "study_days_week": 5,
    "attendance_pct": 82, "assignment_rate": 0.6, "carryovers": 1, "goal_gap": 0.2,
}])[loaded["feature_names"]]
probs = loaded["model"].predict_proba(student)[0]
print({c: round(float(p), 3) for c, p in zip(loaded["classes"], probs)})
print("Expected semester GPA:", round(float(loaded["regressor"].predict(student)[0]), 2))
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
**Next:** copy the downloaded `advisor_model.pkl` into the repo's `models/` folder (replace the old
one), run `python app.py`, and test it in the browser. If loading fails with a scikit-learn version
error, the app retrains a fresh model automatically; or install the version printed in section 1
(`pip install scikit-learn==<version>`) to use the Colab file as is.
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
