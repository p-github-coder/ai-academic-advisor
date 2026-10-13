"""Synthetic student-record generator and shared domain constants.

No real student records are used anywhere in this project. The generator encodes
common-sense relationships (attendance, study time, confidence, difficulty, carry-overs,
workload, assignment habits) so that a model has something realistic to learn.
Swap `generate_dataset()` for a loader of real, anonymised records when available.

The model only ever sees the numeric FEATURES below, in this order. Everything else the
student types (department, study method, challenges, exam date...) drives the rule-based
part of the advisor instead.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Features the ML model is trained on (order matters and is stored in the .pkl).
FEATURES = [
    "level",  # 1..5 (100L..500L)
    "cgpa",
    "last_gpa",
    "avg_score",  # credit-weighted average course score, %
    "credit_load",  # total credit units this semester
    "avg_difficulty",  # 1 easy .. 3 difficult (credit-weighted)
    "avg_confidence",  # 1 low .. 5 high (credit-weighted)
    "study_hours_day",
    "study_days_week",
    "attendance_pct",
    "assignment_rate",  # 0.9 usually, 0.6 sometimes, 0.3 rarely
    "carryovers",  # courses being repeated
    "goal_gap",  # max(0, target CGPA - current CGPA)
]

TARGET_CLASS = "next_standing"
TARGET_GPA = "next_gpa"
# Kept for older callers that only need the class label column name.
TARGET = TARGET_CLASS

# Ordered from worst to best.
CLASSES = ["At Risk", "Fair", "Good", "Excellent"]

DIFFICULTY = {"easy": 1, "moderate": 2, "difficult": 3}
ASSIGNMENT_RATE = {"usually": 0.9, "sometimes": 0.6, "rarely": 0.3}

# Nigerian 5-point scale (lower bound, label). Check your department's handbook.
DEGREE_CLASSES = [
    (4.50, "First Class"),
    (3.50, "Second Class Upper"),
    (2.40, "Second Class Lower"),
    (1.50, "Third Class"),
    (1.00, "Pass"),
]


def degree_class(cgpa: float) -> str:
    """Return the degree class for a CGPA on the 5-point scale."""
    for lower, label in DEGREE_CLASSES:
        if cgpa >= lower:
            return label
    return "Below Pass"


def gpa_to_standing(gpa: float) -> str:
    """Map a semester GPA to the standing label the model predicts."""
    if gpa >= 4.5:
        return "Excellent"
    if gpa >= 3.5:
        return "Good"
    if gpa >= 2.4:
        return "Fair"
    return "At Risk"


def estimate_score_from_gpa(gpa: float) -> float:
    """Typical average course score (%) for a GPA. Used when a student enters no scores."""
    return float(np.clip(30 + 8.5 * gpa, 20, 100))


def generate_dataset(n: int = 4000, seed: int = 42) -> pd.DataFrame:
    """Create `n` synthetic student records with next-semester GPA and standing."""
    rng = np.random.default_rng(seed)

    ability = rng.normal(0.0, 1.0, n)  # hidden driver shared by several features

    level = rng.integers(1, 6, n)
    cgpa = np.clip(3.4 + 0.7 * ability + rng.normal(0, 0.25, n), 1.0, 5.0)
    last_gpa = np.clip(cgpa + rng.normal(0, 0.45, n), 0.0, 5.0)
    avg_score = np.clip(30 + 8.5 * last_gpa + rng.normal(0, 6, n), 20, 100)
    credit_load = rng.choice([15, 18, 21, 24, 27], size=n, p=[0.10, 0.30, 0.30, 0.22, 0.08])
    avg_difficulty = np.clip(2.0 - 0.25 * ability + rng.normal(0, 0.4, n), 1, 3)
    avg_confidence = np.clip(3.0 + 0.6 * ability + rng.normal(0, 0.7, n), 1, 5)
    study_hours_day = np.clip(2.2 + 0.5 * ability + rng.normal(0, 1.0, n), 0, 10)
    study_days_week = np.clip(np.round(4.5 + 0.6 * ability + rng.normal(0, 1.2, n)), 1, 7)
    attendance = np.clip(78 + 5 * ability + rng.normal(0, 11, n), 30, 100)
    habit = 0.5 * ability + rng.normal(0, 0.8, n)
    assignment_rate = np.where(habit > 0.1, 0.9, np.where(habit > -0.8, 0.6, 0.3))
    carryovers = np.clip(rng.poisson(np.clip(1.0 - 0.8 * ability, 0.05, None)), 0, 6)
    target = np.clip(cgpa + np.abs(rng.normal(0.3, 0.4, n)), 1.0, 5.0)
    goal_gap = target - cgpa

    weekly_study = study_hours_day * study_days_week

    # Hidden "true" process for next-semester GPA, plus noise.
    next_gpa = (
        0.30
        + 0.55 * cgpa
        + 0.25 * last_gpa
        + 0.010 * (avg_score - 60)
        + 0.012 * (attendance - 75)
        + 0.025 * (weekly_study - 10)
        + 0.10 * (avg_confidence - 3)
        - 0.12 * (avg_difficulty - 2)
        + 0.25 * (assignment_rate - 0.6)
        - 0.12 * carryovers
        - 0.040 * np.maximum(credit_load - 21, 0)
        + rng.normal(0, 0.30, n)
    )
    next_gpa = np.round(np.clip(next_gpa, 0.0, 5.0), 2)  # round first so label and GPA agree

    df = pd.DataFrame(
        {
            "level": level.astype(int),
            "cgpa": cgpa.round(2),
            "last_gpa": last_gpa.round(2),
            "avg_score": avg_score.round(1),
            "credit_load": credit_load.astype(int),
            "avg_difficulty": avg_difficulty.round(2),
            "avg_confidence": avg_confidence.round(2),
            "study_hours_day": study_hours_day.round(1),
            "study_days_week": study_days_week.astype(int),
            "attendance_pct": attendance.round(0),
            "assignment_rate": assignment_rate,
            "carryovers": carryovers.astype(int),
            "goal_gap": goal_gap.round(2),
        }
    )
    df[TARGET_GPA] = next_gpa
    df[TARGET_CLASS] = [gpa_to_standing(g) for g in next_gpa]
    return df


if __name__ == "__main__":
    out = generate_dataset()
    out.to_csv("data/students.csv", index=False)
    print(out[TARGET_CLASS].value_counts())
    print(f"Wrote {len(out)} rows to data/students.csv")
