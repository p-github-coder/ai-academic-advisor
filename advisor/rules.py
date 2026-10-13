"""Rule-based (expert-system) recommendations.

Each rule fires on a clear condition and returns advice a student can act on. Thresholds are
common conventions on a 5-point scale; adjust them to your institution's handbook.
"""

from __future__ import annotations

from dataclasses import dataclass

PRIORITY_ORDER = {"high": 0, "medium": 1, "low": 2}


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


def apply_rules(p: dict, ranked: list[dict], plan: dict, goal: dict) -> list[Advice]:
    """Run every rule against a validated profile and return advice sorted by priority."""
    out: list[Advice] = []
    add = out.append
    credit_load = sum(c["credits"] for c in p["courses"])
    repeats = [c["code"] for c in p["courses"] if c["repeat"]]

    # Attendance
    if p["attendance_pct"] < 75:
        add(
            Advice(
                "high",
                "Attendance",
                f"Attendance of {p['attendance_pct']:g}% is under 75%, a common threshold for exam "
                "eligibility. Prioritise getting to class and confirm your department's rule.",
            )
        )
    elif p["attendance_pct"] < 85:
        add(Advice("low", "Attendance", "Attendance is fine; pushing past 85% helps retention."))

    # Repeat courses
    if repeats:
        add(
            Advice(
                "high",
                "Repeat courses",
                f"You are repeating {', '.join(repeats)}. Register them first, attend every "
                "lecture and book time with the course lecturer early.",
            )
        )

    # Workload and study time
    if credit_load > 24:
        add(
            Advice(
                "high",
                "Workload",
                f"{credit_load} units is above the usual 24-unit ceiling. Confirm approval and "
                "consider dropping optional courses.",
            )
        )
    weekly, recommended = plan["weekly_hours"], plan["recommended_hours"]
    if weekly < recommended * 0.5:
        add(
            Advice(
                "high",
                "Study time",
                f"{weekly:g} study hours a week is low for {credit_load} units. Aim for about "
                f"{recommended:g} (one hour per credit unit) and block fixed daily slots.",
            )
        )
    elif weekly < recommended:
        add(
            Advice(
                "medium",
                "Study time",
                f"You plan {weekly:g} hours a week; about {recommended:g} (one per credit unit) is "
                "a safer target.",
            )
        )
    if p["hours_per_day"] > 6:
        add(
            Advice(
                "medium",
                "Study time",
                "Study blocks over 6 hours a day are hard to sustain. Use 50-minute sessions with "
                "10-minute breaks.",
            )
        )

    # Coursework habits
    if p["assignment"] == "rarely":
        add(
            Advice(
                "high",
                "Coursework",
                "You rarely complete assignments. Continuous assessment marks are the easiest to "
                "recover, so submit every one from now on.",
            )
        )
    elif p["assignment"] == "sometimes":
        add(
            Advice(
                "medium",
                "Coursework",
                "Make completing every assignment a rule; they add marks before exams begin.",
            )
        )
    if p["misses_deadlines"]:
        add(
            Advice(
                "medium",
                "Deadlines",
                "Put every deadline in a phone calendar with a reminder two days before.",
            )
        )

    # Study methods and materials
    if p["methods"] and set(p["methods"]) <= {"reading", "videos"}:
        add(
            Advice(
                "medium",
                "Study method",
                "Reading and videos alone give a false sense of progress. Add practice questions "
                "and test yourself without notes.",
            )
        )
    if "past_questions" not in p["materials"]:
        add(
            Advice(
                "medium",
                "Materials",
                "Get past questions for each course from your class representative or department; "
                "they show how lecturers set exams.",
            )
        )
    if "notes" not in p["materials"]:
        add(
            Advice(
                "medium",
                "Materials",
                "Collect complete lecture notes from classmates or lecturers before revising.",
            )
        )

    # Biggest challenge
    challenge_tips = {
        "understanding": (
            "Understanding concepts",
            "Explain each topic aloud in your own words. Where you get stuck, ask the lecturer or "
            "a coursemate and attend tutorials.",
        ),
        "remembering": (
            "Remembering information",
            "Use active recall and spaced repetition: quiz yourself after 1 day, 3 days and 1 week "
            "instead of rereading.",
        ),
        "exam_questions": (
            "Answering exam questions",
            "Do timed past questions twice a week, then compare your answers with the marking "
            "scheme or notes.",
        ),
        "time": (
            "Time management",
            "Plan the week on Sunday using the timetable below and review it each night.",
        ),
    }
    if p["challenge"] in challenge_tips:
        category, message = challenge_tips[p["challenge"]]
        add(Advice("medium", category, message))

    # Exam countdown
    exam = plan["exam"]
    if exam and not exam["passed"]:
        d = exam["days_left"]
        if d <= 14:
            add(
                Advice(
                    "high",
                    "Exams",
                    f"Exams start in {d} day{'s' if d != 1 else ''}. Switch to past questions and "
                    "revise your highest-priority courses first.",
                )
            )
        elif d <= 42:
            add(
                Advice(
                    "medium",
                    "Exams",
                    f"{d} days to exams. Finish covering the syllabus within the next "
                    f"{max(d - 14, 7)} days, then spend the last two weeks on practice.",
                )
            )

    # Focus course
    top = [r for r in ranked if r["label"] == "Urgent"][:2]
    if top:
        names = " and ".join(r["code"] for r in top)
        add(Advice("high", "Course focus", f"Start with {names}; they need the most attention."))

    # Trend
    if p["last_gpa"] < p["cgpa"] - 0.5:
        add(
            Advice(
                "high",
                "Trend",
                "Your last semester fell well below your CGPA. Find out what changed and speak to "
                "your course adviser early.",
            )
        )
    elif p["last_gpa"] > p["cgpa"] + 0.3:
        add(Advice("low", "Trend", "Your results are trending up. Keep the same routine."))

    # Goal tracking
    needed = goal["needed_gpa"]
    if goal["status"] == "out_of_reach" and needed is not None:
        add(
            Advice(
                "medium",
                "Goal",
                f"Reaching a CGPA of {goal['target']:g} in one semester would need a GPA of "
                f"{needed:.2f}, above the 5.00 maximum. Spread the climb over several semesters.",
            )
        )
    elif goal["status"] == "stretch" and needed is not None:
        add(
            Advice(
                "medium",
                "Goal",
                f"To reach a CGPA of {goal['target']:g} you need about {needed:.2f} this "
                f"semester; the model projects {goal['projected_cgpa']:.2f}. Close the gap "
                "with the urgent courses.",
            )
        )
    elif goal["status"] == "on_track":
        add(
            Advice(
                "low",
                "Goal",
                f"At your current pace the model projects a CGPA of {goal['projected_cgpa']:.2f}, "
                "which meets your target. Stay consistent.",
            )
        )
    elif goal["status"] == "at_risk":
        add(
            Advice(
                "high",
                "Goal",
                f"The model projects your CGPA dropping to {goal['projected_cgpa']:.2f}, below "
                f"your target of {goal['target']:g}. Fix the urgent courses to protect it.",
            )
        )

    if p["goal_type"] == "recover":
        add(
            Advice(
                "medium",
                "Recovery",
                "Recovering is realistic when you target your weakest courses first. Fix repeat "
                "courses and low-scoring high-credit courses before anything else.",
            )
        )

    out.sort(key=lambda a: PRIORITY_ORDER[a.priority])
    return out
