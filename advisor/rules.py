"""Rule-based (expert-system) half of the advisor, plus input validation.

Rules are explicit and explainable: each one fires on a clear condition and returns
advice a student can act on. Thresholds are common conventions on a 5-point scale and
should be adjusted to your institution's handbook.
"""

from __future__ import annotations

from dataclasses import dataclass

from advisor.data import DEGREE_CLASSES, FEATURES, degree_class

PRIORITY_ORDER = {"high": 0, "medium": 1, "low": 2}


@dataclass(frozen=True)
class Field:
    label: str
    minimum: float
    maximum: float
    default: float
    step: float
    help: str
    integer: bool = False


# Order here is the order shown in the web form. `credits_earned` is used by the
# rules only (the CGPA planner); the ML model uses the FEATURES list.
FIELDS: dict[str, Field] = {
    "cgpa": Field("Current CGPA", 0, 5, 3.5, 0.01, "Cumulative GPA on the 5.0 scale"),
    "last_gpa": Field("Last semester GPA", 0, 5, 3.5, 0.01, "GPA from your most recent semester"),
    "credits_earned": Field(
        "Credit units completed", 0, 200, 60, 1, "Total units passed so far", integer=True
    ),
    "credit_load": Field(
        "Credit units this semester", 12, 30, 21, 1, "Units you are registered for", integer=True
    ),
    "attendance_pct": Field("Class attendance (%)", 0, 100, 80, 1, "Average across your courses"),
    "study_hours_week": Field("Study hours per week", 0, 60, 14, 0.5, "Outside lectures"),
    "assignment_pct": Field("Assignments submitted (%)", 0, 100, 85, 1, "On time and complete"),
    "carryovers": Field("Carry-over courses", 0, 10, 0, 1, "Courses you must resit", integer=True),
    "sleep_hours": Field("Sleep per night (hours)", 0, 14, 7, 0.5, "Typical weeknight"),
    "extracurricular_hours": Field(
        "Extracurricular hours per week", 0, 60, 5, 0.5, "Clubs, jobs, NACOS duties, etc."
    ),
}


def validate_profile(raw: dict) -> tuple[dict, dict]:
    """Coerce `raw` into numbers and range-check it. Returns (clean, errors)."""
    clean: dict[str, float] = {}
    errors: dict[str, str] = {}
    for name, spec in FIELDS.items():
        value = raw.get(name)
        if value is None or str(value).strip() == "":
            errors[name] = "This field is required."
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            errors[name] = "Enter a number."
            continue
        if number != number or number in (float("inf"), float("-inf")):
            errors[name] = "Enter a finite number."
        elif not spec.minimum <= number <= spec.maximum:
            errors[name] = f"Must be between {spec.minimum:g} and {spec.maximum:g}."
        else:
            clean[name] = int(round(number)) if spec.integer else number
    return clean, errors


@dataclass(frozen=True)
class Advice:
    priority: str  # "high" | "medium" | "low"
    category: str
    message: str
    source: str = "rule"

    def as_dict(self) -> dict:
        return {
            "priority": self.priority,
            "category": self.category,
            "message": self.message,
            "source": self.source,
        }


def required_semester_gpa(cgpa: float, credits_earned: float, credit_load: float, target: float):
    """Semester GPA needed this term to lift the CGPA to `target`.

    Solves (cgpa*E + g*L) / (E + L) = target for g. Returns None if there is no history.
    """
    if credits_earned <= 0 or credit_load <= 0:
        return None
    return (target * (credits_earned + credit_load) - cgpa * credits_earned) / credit_load


def target_planner(p: dict) -> Advice | None:
    """Tell the student what it takes to reach the next degree class."""
    cgpa = p["cgpa"]
    higher = [(low, name) for low, name in DEGREE_CLASSES if low > cgpa]
    if not higher:
        return Advice(
            "low", "Goal", "You are in the First Class range. Protect it by staying consistent."
        )
    low, name = higher[-1]  # the nearest class above the current one
    needed = required_semester_gpa(cgpa, p["credits_earned"], p["credit_load"], low)
    if needed is None:
        return None
    if needed > 5.0:
        return Advice(
            "medium",
            "Goal",
            f"Reaching {name} ({low:.2f}+) cannot be done in a single semester "
            f"(you would need a GPA of {needed:.2f}). Aim for steady gains over several terms.",
        )
    return Advice(
        "medium" if needed <= 4.5 else "high",
        "Goal",
        f"To lift your CGPA to {low:.2f} ({name}) after this semester you need a "
        f"semester GPA of about {max(needed, 0):.2f}.",
    )


def apply_rules(p: dict) -> list[Advice]:
    """Run every rule against a validated profile and return advice sorted by priority."""
    out: list[Advice] = []

    if p["attendance_pct"] < 75:
        out.append(
            Advice(
                "high",
                "Attendance",
                f"Attendance of {p['attendance_pct']:g}% is under 75%, a common threshold for "
                "exam eligibility. Prioritise getting to class and confirm your department's rule.",
            )
        )
    elif p["attendance_pct"] < 85:
        out.append(
            Advice("low", "Attendance", "Attendance is fine; pushing past 85% helps retention.")
        )

    if p["carryovers"] > 0:
        out.append(
            Advice(
                "high",
                "Carry-overs",
                f"You have {p['carryovers']} carry-over course(s). Register them first, "
                "attend their lectures and book time with the course lecturer.",
            )
        )

    if p["credit_load"] > 24:
        out.append(
            Advice(
                "high",
                "Workload",
                f"{p['credit_load']} units is above the usual 24-unit ceiling. Confirm approval "
                "and consider dropping optional courses.",
            )
        )
    elif p["credit_load"] >= 22 and p["study_hours_week"] < p["credit_load"]:
        out.append(
            Advice(
                "medium",
                "Workload",
                "A heavy load with limited study time is risky. Plan for about one study hour "
                "per credit unit each week.",
            )
        )

    if p["study_hours_week"] < p["credit_load"] * 0.5:
        out.append(
            Advice(
                "high",
                "Study time",
                f"{p['study_hours_week']:g} study hours a week is low for "
                f"{p['credit_load']} units. "
                "Block out fixed daily study slots.",
            )
        )
    elif p["study_hours_week"] < p["credit_load"]:
        out.append(
            Advice(
                "medium",
                "Study time",
                "Raise study time toward roughly one hour per registered credit unit.",
            )
        )

    if p["assignment_pct"] < 70:
        out.append(
            Advice(
                "high",
                "Coursework",
                f"Only {p['assignment_pct']:g}% of assignments are submitted. Continuous "
                "assessment marks are the easiest to recover.",
            )
        )

    if p["sleep_hours"] < 6:
        out.append(
            Advice(
                "medium",
                "Wellbeing",
                f"{p['sleep_hours']:g} hours of sleep is short; memory and focus suffer. "
                "Aim for 7 to 8 hours, especially in exam weeks.",
            )
        )

    if p["extracurricular_hours"] > 15:
        out.append(
            Advice(
                "medium",
                "Balance",
                f"{p['extracurricular_hours']:g} extracurricular hours a week is a lot. "
                "Delegate or schedule around lectures and tests.",
            )
        )

    if p["last_gpa"] < p["cgpa"] - 0.5:
        out.append(
            Advice(
                "high",
                "Trend",
                "Your last semester fell well below your CGPA. Find out what changed and "
                "speak to your course adviser early.",
            )
        )
    elif p["last_gpa"] > p["cgpa"] + 0.3:
        out.append(Advice("low", "Trend", "Your results are trending up. Keep the same routine."))

    planner = target_planner(p)
    if planner:
        out.append(planner)

    out.sort(key=lambda a: PRIORITY_ORDER[a.priority])
    return out


__all__ = [
    "Advice",
    "FEATURES",
    "FIELDS",
    "apply_rules",
    "degree_class",
    "required_semester_gpa",
    "target_planner",
    "validate_profile",
]
