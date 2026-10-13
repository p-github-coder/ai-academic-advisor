"""Student profile: input options, validation, and derivation of the model's features.

The browser sends one nested JSON object (see `EXAMPLE`). `parse_profile` coerces and
range-checks it and returns `(clean, errors)`. Errors are keyed by field path such as
"cgpa" or "courses.2.credits" so the page can show each message next to its input.
"""

from __future__ import annotations

import math
from datetime import date

from advisor.data import (
    ASSIGNMENT_RATE,
    DIFFICULTY,
    FEATURES,
    estimate_score_from_gpa,
)

LEVELS = [100, 200, 300, 400, 500]
SEMESTERS = ["First", "Second"]
GOAL_TYPES = {
    "improve": "Improve my CGPA",
    "maintain": "Maintain my CGPA",
    "recover": "Recover from poor results",
}
PERIODS = ["Morning", "Afternoon", "Evening", "Night"]
METHODS = {
    "reading": "Reading notes or textbooks",
    "videos": "Watching videos",
    "practice": "Practice questions",
    "group": "Group study",
}
CHALLENGES = {
    "none": "No major challenge",
    "understanding": "Understanding concepts",
    "remembering": "Remembering information",
    "exam_questions": "Answering exam questions",
    "time": "Managing my time",
}
MATERIALS = {
    "notes": "Lecture notes",
    "textbooks": "Textbooks",
    "past_questions": "Past questions",
}
ASSIGNMENT_LABELS = {"usually": "Usually", "sometimes": "Sometimes", "rarely": "Rarely"}
MAX_COURSES = 12


def _num(raw, errors, key, lo, hi, *, integer=False, required=True, default=None):
    """Coerce raw[key] to a number within [lo, hi]; record an error otherwise."""
    value = raw.get(key)
    if value is None or str(value).strip() == "":
        if required:
            errors[key] = "This field is required."
            return None
        return default
    try:
        number = float(value)
    except (TypeError, ValueError):
        errors[key] = "Enter a number."
        return None
    if not math.isfinite(number):
        errors[key] = "Enter a finite number."
        return None
    if not lo <= number <= hi:
        errors[key] = f"Must be between {lo:g} and {hi:g}."
        return None
    return int(round(number)) if integer else number


def _choice(raw, errors, key, allowed, *, default=None):
    value = raw.get(key)
    if value is None or str(value).strip() == "":
        return default
    value = str(value).strip()
    if value not in allowed:
        errors[key] = "Pick one of the listed options."
        return default
    return value


def _checks(raw, errors, key, allowed):
    value = raw.get(key) or []
    if not isinstance(value, list) or any(v not in allowed for v in value):
        errors[key] = "Pick only the listed options."
        return []
    return list(dict.fromkeys(value))


def parse_profile(raw: dict) -> tuple[dict, dict]:
    """Validate a raw profile. Returns (clean, errors); `clean` is only usable if no errors."""
    errors: dict[str, str] = {}
    if not isinstance(raw, dict):
        return {}, {"_": "Send a JSON object."}

    # 1. Student profile
    level = _num(raw, errors, "level", 100, 500, integer=True)
    if level is not None and level not in LEVELS:
        errors["level"] = "Pick a level from 100 to 500."
    department = str(raw.get("department") or "").strip()[:80]
    if not department:
        errors["department"] = "Enter your department or course of study."
    semester = _choice(raw, errors, "semester", SEMESTERS, default="First")
    cgpa = _num(raw, errors, "cgpa", 0, 5)
    last_gpa = _num(raw, errors, "last_gpa", 0, 5)
    current_gpa = _num(raw, errors, "current_gpa", 0, 5, required=False)
    target_cgpa = _num(raw, errors, "target_cgpa", 0, 5)
    goal_type = _choice(raw, errors, "goal_type", GOAL_TYPES, default="improve")
    credits_earned = _num(
        raw, errors, "credits_earned", 0, 240, integer=True, required=False, default=0
    )
    credits_required = _num(
        raw, errors, "credits_required", 30, 240, integer=True, required=False, default=120
    )
    if (
        credits_earned is not None
        and credits_required is not None
        and "credits_earned" not in errors
        and credits_earned > credits_required
    ):
        errors["credits_earned"] = "Cannot be more than the credits needed to graduate."

    # 2. Courses
    courses = []
    raw_courses = raw.get("courses")
    if not isinstance(raw_courses, list) or not raw_courses:
        errors["courses"] = "Add at least one course."
    elif len(raw_courses) > MAX_COURSES:
        errors["courses"] = f"Add at most {MAX_COURSES} courses."
    else:
        seen = set()
        for i, rc in enumerate(raw_courses):
            rc = rc if isinstance(rc, dict) else {}
            prefix = f"courses.{i}."
            sub: dict[str, str] = {}
            code = str(rc.get("code") or "").strip().upper()[:20]
            if not code:
                sub["code"] = "Enter a course code."
            elif code in seen:
                sub["code"] = "Duplicate course."
            seen.add(code)
            credits = _num(rc, sub, "credits", 1, 6, integer=True)
            score = _num(rc, sub, "score", 0, 100, required=False, default=None)
            difficulty = _choice(rc, sub, "difficulty", DIFFICULTY, default="moderate")
            confidence = _num(rc, sub, "confidence", 1, 5, integer=True, required=False, default=3)
            errors.update({prefix + k: v for k, v in sub.items()})
            courses.append(
                {
                    "code": code,
                    "credits": credits,
                    "score": score,
                    "difficulty": difficulty,
                    "confidence": confidence,
                    "repeat": bool(rc.get("repeat")),
                }
            )

    # 3. Study habits
    hours_per_day = _num(raw, errors, "hours_per_day", 0, 16)
    days_per_week = _num(
        raw, errors, "days_per_week", 1, 7, integer=True, required=False, default=5
    )
    preferred_period = _choice(raw, errors, "preferred_period", PERIODS, default="Evening")
    methods = _checks(raw, errors, "methods", METHODS)
    attendance = _num(raw, errors, "attendance_pct", 0, 100, required=False, default=80)
    assignment = _choice(raw, errors, "assignment", ASSIGNMENT_RATE, default="usually")
    exam_date = None
    if str(raw.get("exam_date") or "").strip():
        try:
            exam_date = date.fromisoformat(str(raw["exam_date"]).strip()).isoformat()
        except ValueError:
            errors["exam_date"] = "Use a valid date."

    # 4. Challenges
    challenge = _choice(raw, errors, "challenge", CHALLENGES, default="none")
    misses_deadlines = bool(raw.get("misses_deadlines"))
    materials = _checks(raw, errors, "materials", MATERIALS)
    priority_course = str(raw.get("priority_course") or "").strip().upper()[:20] or None

    clean = {
        "level": level,
        "department": department,
        "semester": semester,
        "cgpa": cgpa,
        "last_gpa": last_gpa,
        "current_gpa": current_gpa,
        "target_cgpa": target_cgpa,
        "goal_type": goal_type,
        "credits_earned": credits_earned,
        "credits_required": credits_required,
        "courses": courses,
        "hours_per_day": hours_per_day,
        "days_per_week": days_per_week,
        "preferred_period": preferred_period,
        "methods": methods,
        "attendance_pct": attendance,
        "assignment": assignment,
        "exam_date": exam_date,
        "challenge": challenge,
        "misses_deadlines": misses_deadlines,
        "materials": materials,
        "priority_course": priority_course,
    }
    return clean, errors


def derive_features(p: dict) -> tuple[dict, dict]:
    """Turn a clean profile into the model's numeric features.

    Returns (features, notes). `notes["score_estimated"]` is True when the student entered
    no scores and the average score was estimated from last semester's GPA.
    """
    courses = p["courses"]
    total = sum(c["credits"] for c in courses)

    def weighted(key, convert=lambda v: v):
        return sum(convert(c[key]) * c["credits"] for c in courses) / total

    scored = [c for c in courses if c["score"] is not None]
    if scored:
        weight = sum(c["credits"] for c in scored)
        avg_score = sum(c["score"] * c["credits"] for c in scored) / weight
        estimated = False
    else:
        avg_score = estimate_score_from_gpa(p["last_gpa"])
        estimated = True

    features = {
        "level": p["level"] // 100,
        "cgpa": p["cgpa"],
        "last_gpa": p["last_gpa"],
        "avg_score": avg_score,
        "credit_load": total,
        "avg_difficulty": weighted("difficulty", lambda d: DIFFICULTY[d]),
        "avg_confidence": weighted("confidence"),
        "study_hours_day": p["hours_per_day"],
        "study_days_week": p["days_per_week"],
        "attendance_pct": p["attendance_pct"],
        "assignment_rate": ASSIGNMENT_RATE[p["assignment"]],
        "carryovers": sum(1 for c in courses if c["repeat"]),
        "goal_gap": max(0.0, p["target_cgpa"] - p["cgpa"]),
    }
    assert list(features) == FEATURES
    return features, {"score_estimated": estimated}


EXAMPLE = {
    "level": 300,
    "department": "Computer Science",
    "semester": "First",
    "cgpa": 3.6,
    "last_gpa": 3.4,
    "current_gpa": "",
    "target_cgpa": 3.8,
    "goal_type": "improve",
    "credits_earned": 72,
    "credits_required": 120,
    "courses": [
        {"code": "CSC 301", "credits": 3, "score": 58, "difficulty": "difficult", "confidence": 2},
        {"code": "CSC 303", "credits": 3, "score": 66, "difficulty": "moderate", "confidence": 3},
        {"code": "CSC 305", "credits": 4, "score": "", "difficulty": "difficult", "confidence": 2},
        {"code": "MTH 301", "credits": 3, "score": 72, "difficulty": "moderate", "confidence": 4},
        {"code": "GST 311", "credits": 2, "score": 81, "difficulty": "easy", "confidence": 5},
        {
            "code": "CSC 207",
            "credits": 3,
            "score": 45,
            "difficulty": "difficult",
            "confidence": 2,
            "repeat": True,
        },
    ],
    "hours_per_day": 3,
    "days_per_week": 5,
    "preferred_period": "Evening",
    "methods": ["reading", "videos"],
    "attendance_pct": 82,
    "assignment": "sometimes",
    "exam_date": "",
    "challenge": "exam_questions",
    "misses_deadlines": False,
    "materials": ["notes", "textbooks"],
    "priority_course": "CSC 301",
}
