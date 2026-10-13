"""Course priority ranking, weekly study timetable and goal tracking.

These are transparent calculations (no ML): every number can be traced to the student's
own inputs, which keeps the expert-system half of the advisor explainable.
"""

from __future__ import annotations

from datetime import date

from advisor.data import DIFFICULTY, degree_class

DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
# Which weekdays to study on, for 1..7 study days a week (spread out, rest days kept).
DAY_PATTERNS = {
    1: [5],
    2: [1, 5],
    3: [0, 2, 4],
    4: [0, 1, 3, 5],
    5: [0, 1, 2, 3, 4],
    6: [0, 1, 2, 3, 4, 5],
    7: [0, 1, 2, 3, 4, 5, 6],
}
PASS_MARK = 70  # a score of 70% or more is an A on the common 5-point scale


def _clip(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


# ---------------------------------------------------------------- course priority


def rank_courses(p: dict) -> list[dict]:
    """Rank courses by how much attention they need (0-100) with the reasons why."""
    ranked = []
    for c in p["courses"]:
        reasons = []
        difficulty = (DIFFICULTY[c["difficulty"]] - 1) / 2
        low_confidence = (5 - c["confidence"]) / 4
        credit_weight = c["credits"] / 6

        if c["score"] is None:
            score_gap = 0.5  # unknown: assume a moderate gap rather than ignoring it
        else:
            score_gap = _clip((PASS_MARK - c["score"]) / 40)
            if c["score"] < 50:
                reasons.append(f"score of {c['score']:g}% is below 50%")
            elif c["score"] < 60:
                reasons.append(f"score of {c['score']:g}% is below 60%")

        if c["difficulty"] == "difficult":
            reasons.append("you rated it difficult")
        if c["confidence"] <= 2:
            reasons.append("low confidence")
        if c["credits"] >= 4:
            reasons.append(f"{c['credits']} credit units carry a lot of weight")
        if c["repeat"]:
            reasons.append("repeat course")
        is_priority = p["priority_course"] == c["code"]
        if is_priority:
            reasons.append("your top priority")

        raw = (
            0.25 * difficulty
            + 0.25 * low_confidence
            + 0.25 * score_gap
            + 0.15 * credit_weight
            + (0.10 if c["repeat"] else 0.0)
            + (0.10 if is_priority else 0.0)
        )
        priority = round(_clip(raw) * 100)
        label = "Urgent" if priority >= 60 else "Needs attention" if priority >= 35 else "On track"
        ranked.append(
            {
                "code": c["code"],
                "credits": c["credits"],
                "score": c["score"],
                "difficulty": c["difficulty"],
                "confidence": c["confidence"],
                "repeat": c["repeat"],
                "priority": priority,
                "label": label,
                "reasons": reasons or ["no warning signs"],
            }
        )
    ranked.sort(key=lambda r: (-r["priority"], -r["credits"], r["code"]))
    for i, r in enumerate(ranked, 1):
        r["rank"] = i
    return ranked


# ---------------------------------------------------------------- study timetable


def _split_hours(ranked: list[dict], weekly: float) -> dict[str, float]:
    """Share `weekly` hours across courses in proportion to priority (halves of an hour)."""
    weights = {r["code"]: r["priority"] + 10 for r in ranked}  # +10: nobody gets zero
    total = sum(weights.values())
    units = int(round(weekly * 2))  # half-hour units
    shares = {k: w / total * units for k, w in weights.items()}
    alloc = {k: int(v) for k, v in shares.items()}
    leftover = units - sum(alloc.values())
    # Hand out the leftover half-hours to the largest fractional remainders.
    for k in sorted(shares, key=lambda k: shares[k] - alloc[k], reverse=True)[: max(leftover, 0)]:
        alloc[k] += 1
    return {k: v / 2 for k, v in alloc.items()}


def build_timetable(p: dict, ranked: list[dict], today: date | None = None) -> dict:
    today = today or date.today()
    hours_day = p["hours_per_day"]
    days_week = p["days_per_week"]
    weekly = round(hours_day * days_week * 2) / 2
    credit_load = sum(c["credits"] for c in p["courses"])
    recommended = float(credit_load)  # rule of thumb: about 1 study hour per credit unit a week

    exam = None
    if p["exam_date"]:
        days_left = (date.fromisoformat(p["exam_date"]) - today).days
        exam = {"date": p["exam_date"], "days_left": days_left, "passed": days_left < 0}

    if weekly <= 0:
        return {
            "weekly_hours": 0,
            "recommended_hours": recommended,
            "gap": recommended,
            "per_course": [],
            "days": [],
            "period": p["preferred_period"],
            "exam": exam,
        }

    per_course = _split_hours(ranked, weekly)
    remaining = dict(per_course)
    study_days = DAY_PATTERNS[days_week]
    days = []
    for d in range(7):
        sessions = []
        if d in study_days:
            left = hours_day
            used = set()
            while left > 0 and any(v > 0 for v in remaining.values()):
                # Prefer the course with most hours left that has not been used today yet.
                candidates = [k for k, v in remaining.items() if v > 0 and k not in used] or [
                    k for k, v in remaining.items() if v > 0
                ]
                code = max(candidates, key=lambda k: remaining[k])
                chunk = min(left, remaining[code], 2.0)
                sessions.append({"course": code, "hours": chunk})
                remaining[code] -= chunk
                left -= chunk
                used.add(code)
        days.append({"day": DAYS[d], "sessions": sessions, "rest": d not in study_days})

    return {
        "weekly_hours": weekly,
        "recommended_hours": recommended,
        "gap": round(recommended - weekly, 1),
        "per_course": [{"course": k, "hours": v} for k, v in per_course.items()],
        "days": days,
        "period": p["preferred_period"],
        "exam": exam,
    }


# ---------------------------------------------------------------- goal tracking


def required_semester_gpa(cgpa: float, credits_earned: float, credit_load: float, target: float):
    """Semester GPA needed this term to lift the CGPA to `target`.

    Solves (cgpa*E + g*L) / (E + L) = target for g. Returns None if there is no history.
    """
    if credits_earned <= 0 or credit_load <= 0:
        return None
    return (target * (credits_earned + credit_load) - cgpa * credits_earned) / credit_load


def project_cgpa(cgpa: float, credits_earned: float, credit_load: float, semester_gpa: float):
    """CGPA after this semester if the student earns `semester_gpa`."""
    if credits_earned <= 0:
        return round(semester_gpa, 2)
    total = credits_earned + credit_load
    return round((cgpa * credits_earned + semester_gpa * credit_load) / total, 2)


def track_goal(p: dict, predicted_gpa: float) -> dict:
    credit_load = sum(c["credits"] for c in p["courses"])
    earned, required = p["credits_earned"], p["credits_required"]
    cgpa, target = p["cgpa"], p["target_cgpa"]

    projected = project_cgpa(cgpa, earned, credit_load, predicted_gpa)
    needed = required_semester_gpa(cgpa, earned, credit_load, target)

    if target <= cgpa:
        status = "on_track" if projected >= target - 0.005 else "at_risk"
    elif projected >= target - 0.005:
        status = "on_track"
    elif needed is not None and needed > 5.0:
        status = "out_of_reach"
    else:
        status = "stretch"

    remaining_credits = max(required - earned, 0)
    return {
        "cgpa": cgpa,
        "target": target,
        "goal_type": p["goal_type"],
        "projected_cgpa": projected,
        "current_class": degree_class(cgpa),
        "projected_class": degree_class(projected),
        "target_class": degree_class(target),
        "needed_gpa": None if needed is None else round(needed, 2),
        "status": status,
        "progress_pct": round(_clip(cgpa / target if target else 1.0) * 100),
        "credits_earned": earned,
        "credits_required": required,
        "credits_pct": round(_clip(earned / required if required else 0) * 100),
        "remaining_credits": remaining_credits,
        "semesters_left": (
            None if credit_load <= 0 else max(1, -(-remaining_credits // credit_load))
        ),
    }
