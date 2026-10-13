"""Hybrid advisor: ML predictions + planner + rule engine, assembled into one result."""

from __future__ import annotations

from datetime import date

import pandas as pd

from advisor.data import degree_class
from advisor.planner import build_timetable, rank_courses, track_goal
from advisor.profile import derive_features, parse_profile
from advisor.rules import Advice, apply_rules

STANDING_BLURB = {
    "Excellent": "The model expects a very strong semester (GPA 4.50 or higher).",
    "Good": "The model expects a solid semester (GPA 3.50 to 4.49).",
    "Fair": "The model expects an average semester (GPA 2.40 to 3.49).",
    "At Risk": "The model expects a weak semester (GPA below 2.40). Act early.",
}
# Used only if a model file has no regressor: expected GPA from the class probabilities.
CLASS_MIDPOINT = {"At Risk": 1.8, "Fair": 3.0, "Good": 4.0, "Excellent": 4.75}


class Advisor:
    """Wraps a loaded model bundle and turns student profiles into a full advising plan."""

    def __init__(self, bundle: dict):
        self.model = bundle["model"]
        self.regressor = bundle.get("regressor")
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

    def advise(self, raw: dict, today: date | None = None) -> dict:
        """Validate `raw`, predict, and return a JSON-serialisable plan.

        Returns {"ok": False, "errors": {...}} when validation fails.
        """
        profile, errors = parse_profile(raw)
        if errors:
            return {"ok": False, "errors": errors}

        features, notes = derive_features(profile)
        row = pd.DataFrame([[features[f] for f in self.feature_names]], columns=self.feature_names)

        proba = self.model.predict_proba(row)[0]
        classes = [str(c) for c in self.model.classes_]
        probabilities = {c: round(float(p), 4) for c, p in zip(classes, proba, strict=True)}
        standing = max(probabilities, key=probabilities.get)

        if self.regressor is not None:
            predicted_gpa = float(self.regressor.predict(row)[0])
        else:
            predicted_gpa = sum(CLASS_MIDPOINT[c] * p for c, p in probabilities.items())
        predicted_gpa = round(min(5.0, max(0.0, predicted_gpa)), 2)

        ranked = rank_courses(profile)
        plan = build_timetable(profile, ranked, today)
        goal = track_goal(profile, predicted_gpa)
        advice = apply_rules(profile, ranked, plan, goal)

        if standing == "At Risk" and not any(a.priority == "high" for a in advice):
            advice.insert(
                0,
                Advice(
                    "high",
                    "Model alert",
                    "No single warning sign stands out, but the combination of your inputs looks "
                    "risky. Meet your course adviser this week.",
                    source="model",
                ),
            )
        if notes["score_estimated"]:
            advice.append(
                Advice(
                    "low",
                    "Scores",
                    "You entered no course scores, so the model estimated your average from last "
                    "semester's GPA. Add scores for a sharper prediction.",
                    source="model",
                )
            )

        return {
            "ok": True,
            "summary": {
                "predicted_standing": standing,
                "standing_note": STANDING_BLURB.get(standing, ""),
                "probabilities": probabilities,
                "predicted_gpa": predicted_gpa,
                "predicted_class": degree_class(predicted_gpa),
                "confidence": round(probabilities[standing], 2),
                "score_estimated": notes["score_estimated"],
                "credit_load": features["credit_load"],
            },
            "goal": goal,
            "courses": ranked,
            "timetable": plan,
            "advice": [a.as_dict() for a in advice],
            "outlook": [
                {"label": "Last semester", "value": profile["last_gpa"], "kind": "gpa"},
                {"label": "Current CGPA", "value": profile["cgpa"], "kind": "cgpa"},
                {"label": "Predicted this semester", "value": predicted_gpa, "kind": "predicted"},
                {"label": "Projected CGPA", "value": goal["projected_cgpa"], "kind": "projected"},
            ],
            "model_focus": self.top_factors(),
            "profile": {
                "level": profile["level"],
                "department": profile["department"],
                "semester": profile["semester"],
            },
        }
