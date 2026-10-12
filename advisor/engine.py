"""Hybrid advisor: combines the ML model's prediction with the rule engine."""

from __future__ import annotations

import pandas as pd

from advisor.data import degree_class
from advisor.rules import Advice, apply_rules, validate_profile

STANDING_BLURB = {
    "Excellent": "The model expects a very strong semester (GPA 4.50+).",
    "Good": "The model expects a solid semester (GPA 3.50 to 4.49).",
    "Fair": "The model expects an average semester (GPA 2.40 to 3.49).",
    "At Risk": "The model expects a weak semester (GPA below 2.40). Act early.",
}


class Advisor:
    """Wraps a loaded model bundle and turns student profiles into advice."""

    def __init__(self, bundle: dict):
        self.model = bundle["model"]
        self.feature_names: list[str] = list(bundle["feature_names"])
        self.metrics: dict = bundle.get("metrics", {})
        self.sklearn_version: str = bundle.get("sklearn_version", "unknown")

    def top_factors(self, k: int = 4) -> list[dict]:
        """Global feature importances: what the model leans on most (not per-student)."""
        importances = getattr(self.model, "feature_importances_", None)
        if importances is None:
            return []
        pairs = sorted(
            zip(self.feature_names, importances, strict=True), key=lambda t: t[1], reverse=True
        )
        return [{"feature": f, "importance": round(float(i), 3)} for f, i in pairs[:k]]

    def advise(self, raw: dict) -> dict:
        """Validate `raw`, predict, and return a JSON-serialisable result.

        Returns {"ok": False, "errors": {...}} when validation fails.
        """
        profile, errors = validate_profile(raw)
        if errors:
            return {"ok": False, "errors": errors}

        row = pd.DataFrame([[profile[f] for f in self.feature_names]], columns=self.feature_names)
        proba = self.model.predict_proba(row)[0]
        classes = [str(c) for c in self.model.classes_]
        probabilities = {c: round(float(p), 4) for c, p in zip(classes, proba, strict=True)}
        standing = max(probabilities, key=probabilities.get)

        advice = apply_rules(profile)
        if standing == "At Risk" and not any(a.priority == "high" for a in advice):
            advice.insert(
                0,
                Advice(
                    "high",
                    "Model alert",
                    "No single warning sign stands out, but the combination of your inputs "
                    "looks risky. Meet your course adviser this week.",
                    source="model",
                ),
            )

        return {
            "ok": True,
            "predicted_standing": standing,
            "standing_note": STANDING_BLURB.get(standing, ""),
            "probabilities": probabilities,
            "current_class": degree_class(profile["cgpa"]),
            "advice": [a.as_dict() for a in advice],
            "model_focus": self.top_factors(),
            "profile": profile,
        }
