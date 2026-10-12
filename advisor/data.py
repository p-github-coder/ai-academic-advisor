"""Synthetic student-record generator and shared domain constants.

No real student records are used anywhere in this project. The generator encodes
common-sense relationships (attendance, study time, carry-overs, workload and sleep
all influence next-semester GPA) so that a model has something realistic to learn.
Swap `generate_dataset()` for a loader of real, anonymised records when available.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Features the ML model is trained on (order matters and is stored in the .pkl).
FEATURES = [
    "cgpa",
    "last_gpa",
    "attendance_pct",
    "study_hours_week",
    "assignment_pct",
    "carryovers",
    "credit_load",
    "sleep_hours",
    "extracurricular_hours",
]

TARGET = "next_standing"

# Ordered from worst to best.
CLASSES = ["At Risk", "Fair", "Good", "Excellent"]

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


def generate_dataset(n: int = 4000, seed: int = 42) -> pd.DataFrame:
    """Create `n` synthetic student records with the next-semester standing label."""
    rng = np.random.default_rng(seed)

    ability = rng.normal(0.0, 1.0, n)  # hidden driver shared by several features

    cgpa = np.clip(3.4 + 0.75 * ability + rng.normal(0, 0.25, n), 1.0, 5.0)
    last_gpa = np.clip(cgpa + rng.normal(0, 0.45, n), 0.0, 5.0)
    attendance = np.clip(78 + 5 * ability + rng.normal(0, 11, n), 30, 100)
    study = np.clip(13 + 3 * ability + rng.normal(0, 5, n), 0, 40)
    assignment = np.clip(80 + 6 * ability + rng.normal(0, 10, n), 20, 100)
    carryovers = np.clip(rng.poisson(np.clip(1.0 - 0.8 * ability, 0.05, None)), 0, 6)
    credit_load = rng.choice([15, 18, 21, 24, 27], size=n, p=[0.10, 0.30, 0.30, 0.22, 0.08])
    sleep = np.clip(rng.normal(6.8, 1.2, n), 3, 10)
    extracurricular = np.clip(rng.gamma(2.0, 3.0, n), 0, 25)

    # Hidden "true" process for next-semester GPA, plus noise.
    next_gpa = (
        0.35
        + 0.60 * cgpa
        + 0.25 * last_gpa
        + 0.012 * (attendance - 75)
        + 0.020 * (study - 13)
        + 0.008 * (assignment - 80)
        - 0.12 * carryovers
        - 0.040 * np.maximum(credit_load - 21, 0)
        + 0.08 * np.clip(sleep - 6.0, -3, 2)
        - 0.025 * np.maximum(extracurricular - 10, 0)
        + rng.normal(0, 0.30, n)
    )
    next_gpa = np.clip(next_gpa, 0.0, 5.0)

    df = pd.DataFrame(
        {
            "cgpa": cgpa.round(2),
            "last_gpa": last_gpa.round(2),
            "attendance_pct": attendance.round(0),
            "study_hours_week": study.round(1),
            "assignment_pct": assignment.round(0),
            "carryovers": carryovers.astype(int),
            "credit_load": credit_load.astype(int),
            "sleep_hours": sleep.round(1),
            "extracurricular_hours": extracurricular.round(1),
        }
    )
    df[TARGET] = [gpa_to_standing(g) for g in next_gpa]
    return df


if __name__ == "__main__":
    out = generate_dataset()
    out.to_csv("data/students.csv", index=False)
    print(out[TARGET].value_counts())
    print(f"Wrote {len(out)} rows to data/students.csv")
