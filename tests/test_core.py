"""Unit tests: data generation, profile validation, planner, rules, model bundle."""

import copy
import tempfile
import unittest
from datetime import date
from pathlib import Path

from advisor.data import (
    CLASSES,
    FEATURES,
    TARGET_CLASS,
    TARGET_GPA,
    degree_class,
    generate_dataset,
    gpa_to_standing,
)
from advisor.engine import Advisor
from advisor.model import load_bundle, save_bundle, train_bundle
from advisor.planner import (
    build_timetable,
    project_cgpa,
    rank_courses,
    required_semester_gpa,
)
from advisor.profile import EXAMPLE, derive_features, parse_profile

TODAY = date(2026, 10, 13)


def profile(**overrides):
    base = copy.deepcopy(EXAMPLE)
    base.update(overrides)
    return base


def clean(**overrides):
    parsed, errors = parse_profile(profile(**overrides))
    assert errors == {}, errors
    return parsed


class DataTests(unittest.TestCase):
    def test_shape_columns_and_labels(self):
        df = generate_dataset(300, seed=1)
        self.assertEqual(len(df), 300)
        self.assertEqual(list(df.columns), [*FEATURES, TARGET_GPA, TARGET_CLASS])
        self.assertTrue(set(df[TARGET_CLASS]) <= set(CLASSES))

    def test_deterministic_for_same_seed(self):
        self.assertTrue(generate_dataset(100, seed=7).equals(generate_dataset(100, seed=7)))

    def test_values_stay_in_range(self):
        df = generate_dataset(500, seed=3)
        self.assertTrue(df["cgpa"].between(0, 5).all())
        self.assertTrue(df[TARGET_GPA].between(0, 5).all())
        self.assertTrue(df["attendance_pct"].between(0, 100).all())
        self.assertTrue(df["level"].between(1, 5).all())

    def test_label_matches_gpa(self):
        df = generate_dataset(200, seed=2)
        self.assertTrue((df[TARGET_GPA].map(gpa_to_standing) == df[TARGET_CLASS]).all())

    def test_class_boundaries(self):
        self.assertEqual(degree_class(4.50), "First Class")
        self.assertEqual(degree_class(4.49), "Second Class Upper")
        self.assertEqual(degree_class(2.40), "Second Class Lower")
        self.assertEqual(degree_class(0.5), "Below Pass")
        self.assertEqual(gpa_to_standing(2.39), "At Risk")
        self.assertEqual(gpa_to_standing(4.5), "Excellent")


class ProfileTests(unittest.TestCase):
    def test_example_is_valid(self):
        _, errors = parse_profile(EXAMPLE)
        self.assertEqual(errors, {})

    def test_required_fields_and_ranges(self):
        _, errors = parse_profile(profile(cgpa=7, last_gpa="abc", department="", level=""))
        for key in ("cgpa", "last_gpa", "department", "level"):
            self.assertIn(key, errors)

    def test_rejects_nan_inf_and_bad_level(self):
        _, errors = parse_profile(profile(cgpa="nan", target_cgpa="inf", level=250))
        self.assertEqual(
            {"cgpa", "target_cgpa", "level"} & set(errors), {"cgpa", "target_cgpa", "level"}
        )

    def test_course_errors_are_addressed_by_index(self):
        courses = [
            {"code": "CSC 1", "credits": 3},
            {"code": "", "credits": 9, "score": 120},
            {"code": "csc 1", "credits": 3},
        ]
        _, errors = parse_profile(profile(courses=courses))
        self.assertIn("courses.1.code", errors)
        self.assertIn("courses.1.credits", errors)
        self.assertIn("courses.1.score", errors)
        self.assertIn("courses.2.code", errors)  # duplicate, case-insensitive

    def test_needs_a_course_and_caps_the_list(self):
        self.assertIn("courses", parse_profile(profile(courses=[]))[1])
        many = [{"code": f"C{i}", "credits": 2} for i in range(13)]
        self.assertIn("courses", parse_profile(profile(courses=many))[1])

    def test_optional_fields_can_be_blank(self):
        parsed, errors = parse_profile(
            profile(current_gpa="", exam_date="", credits_earned="", priority_course="")
        )
        self.assertEqual(errors, {})
        self.assertIsNone(parsed["current_gpa"])
        self.assertIsNone(parsed["exam_date"])
        self.assertEqual(parsed["credits_earned"], 0)

    def test_bad_choices_and_dates(self):
        _, errors = parse_profile(
            profile(assignment="never", exam_date="31/12/2026", methods=["telepathy"])
        )
        self.assertEqual(
            {"assignment", "exam_date", "methods"} & set(errors),
            {"assignment", "exam_date", "methods"},
        )

    def test_credits_earned_cannot_exceed_requirement(self):
        _, errors = parse_profile(profile(credits_earned=130, credits_required=120))
        self.assertIn("credits_earned", errors)

    def test_non_dict_payload(self):
        self.assertIn("_", parse_profile([1, 2])[1])

    def test_features_follow_model_order_and_weights(self):
        features, notes = derive_features(clean())
        self.assertEqual(list(features), FEATURES)
        self.assertEqual(features["level"], 3)
        self.assertEqual(features["credit_load"], 18)
        self.assertEqual(features["carryovers"], 1)
        self.assertAlmostEqual(features["goal_gap"], 0.2)
        self.assertFalse(notes["score_estimated"])

    def test_score_is_estimated_when_none_entered(self):
        courses = [{"code": "A 1", "credits": 3}, {"code": "B 2", "credits": 2}]
        features, notes = derive_features(clean(courses=courses))
        self.assertTrue(notes["score_estimated"])
        self.assertAlmostEqual(features["avg_score"], 30 + 8.5 * 3.4)


class PlannerTests(unittest.TestCase):
    def test_goal_math(self):
        # (3.0*60 + g*20) / 80 = 3.5  ->  g = 5.0
        self.assertAlmostEqual(required_semester_gpa(3.0, 60, 20, 3.5), 5.0)
        self.assertIsNone(required_semester_gpa(3.0, 0, 20, 3.5))
        self.assertEqual(project_cgpa(3.0, 60, 20, 4.0), 3.25)
        self.assertEqual(project_cgpa(3.0, 0, 20, 4.0), 4.0)

    def test_weak_heavy_repeat_course_ranks_first(self):
        ranked = rank_courses(clean())
        self.assertEqual(ranked[0]["code"], "CSC 207")
        self.assertEqual(ranked[-1]["code"], "GST 311")
        self.assertEqual([r["rank"] for r in ranked], list(range(1, len(ranked) + 1)))
        self.assertTrue(all(0 <= r["priority"] <= 100 for r in ranked))

    def test_priority_course_gets_a_boost(self):
        plain = {r["code"]: r["priority"] for r in rank_courses(clean(priority_course=""))}
        boosted = {r["code"]: r["priority"] for r in rank_courses(clean(priority_course="MTH 301"))}
        self.assertGreater(boosted["MTH 301"], plain["MTH 301"])

    def test_timetable_adds_up(self):
        p = clean(hours_per_day=3, days_per_week=5)
        plan = build_timetable(p, rank_courses(p), TODAY)
        self.assertEqual(plan["weekly_hours"], 15)
        self.assertAlmostEqual(sum(c["hours"] for c in plan["per_course"]), 15)
        scheduled = sum(s["hours"] for d in plan["days"] for s in d["sessions"])
        self.assertAlmostEqual(scheduled, 15)
        self.assertEqual(sum(1 for d in plan["days"] if d["sessions"]), 5)
        self.assertEqual(len(plan["days"]), 7)

    def test_timetable_gives_urgent_courses_more_time(self):
        p = clean()
        plan = build_timetable(p, rank_courses(p), TODAY)
        hours = {c["course"]: c["hours"] for c in plan["per_course"]}
        self.assertGreater(hours["CSC 207"], hours["GST 311"])

    def test_zero_study_hours_gives_no_sessions(self):
        p = clean(hours_per_day=0)
        plan = build_timetable(p, rank_courses(p), TODAY)
        self.assertEqual(plan["days"], [])

    def test_exam_countdown(self):
        p = clean(exam_date="2026-10-27")
        self.assertEqual(build_timetable(p, rank_courses(p), TODAY)["exam"]["days_left"], 14)
        p = clean(exam_date="2026-10-01")
        self.assertTrue(build_timetable(p, rank_courses(p), TODAY)["exam"]["passed"])


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = train_bundle(generate_dataset(1500, seed=5), seed=5)
        cls.advisor = Advisor(cls.bundle)

    def test_bundle_keys_and_quality(self):
        for key in ("model", "regressor", "feature_names", "classes", "metrics", "sklearn_version"):
            self.assertIn(key, self.bundle)
        self.assertEqual(self.bundle["feature_names"], FEATURES)
        self.assertGreater(self.bundle["metrics"]["accuracy"], 0.6)
        self.assertLess(self.bundle["metrics"]["gpa_mae"], 0.5)

    def test_pickle_round_trip_gives_identical_plans(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "m.pkl"
            save_bundle(self.bundle, str(path))
            loaded = Advisor(load_bundle(str(path)))
        a = self.advisor.advise(EXAMPLE, TODAY)
        b = loaded.advise(EXAMPLE, TODAY)
        self.assertEqual(a["summary"], b["summary"])

    def test_load_rejects_wrong_format(self):
        import joblib

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.pkl"
            joblib.dump({"oops": 1}, path)
            with self.assertRaises(ValueError):
                load_bundle(str(path))

    def test_weak_profile_is_riskier_than_strong(self):
        weak = self.advisor.advise(
            profile(
                cgpa=1.8,
                last_gpa=1.5,
                target_cgpa=3.0,
                attendance_pct=50,
                hours_per_day=0.5,
                assignment="rarely",
                courses=[
                    {
                        "code": "A 1",
                        "credits": 6,
                        "score": 35,
                        "difficulty": "difficult",
                        "confidence": 1,
                        "repeat": True,
                    },
                    {
                        "code": "B 2",
                        "credits": 6,
                        "score": 38,
                        "difficulty": "difficult",
                        "confidence": 1,
                        "repeat": True,
                    },
                    {
                        "code": "C 3",
                        "credits": 6,
                        "score": 40,
                        "difficulty": "difficult",
                        "confidence": 2,
                    },
                    {
                        "code": "D 4",
                        "credits": 6,
                        "score": 42,
                        "difficulty": "difficult",
                        "confidence": 2,
                    },
                ],
            ),
            TODAY,
        )
        strong = self.advisor.advise(
            profile(
                cgpa=4.7,
                last_gpa=4.8,
                target_cgpa=4.8,
                attendance_pct=97,
                hours_per_day=4,
                assignment="usually",
                courses=[
                    {
                        "code": "A 1",
                        "credits": 3,
                        "score": 90,
                        "difficulty": "easy",
                        "confidence": 5,
                    },
                    {
                        "code": "B 2",
                        "credits": 3,
                        "score": 88,
                        "difficulty": "moderate",
                        "confidence": 5,
                    },
                    {
                        "code": "C 3",
                        "credits": 3,
                        "score": 85,
                        "difficulty": "easy",
                        "confidence": 4,
                    },
                ],
            ),
            TODAY,
        )
        self.assertEqual(weak["summary"]["predicted_standing"], "At Risk")
        self.assertGreater(strong["summary"]["predicted_gpa"], weak["summary"]["predicted_gpa"] + 1)
        self.assertGreater(
            weak["summary"]["probabilities"]["At Risk"],
            strong["summary"]["probabilities"]["At Risk"],
        )
        self.assertTrue(any(a["priority"] == "high" for a in weak["advice"]))
        self.assertFalse(any(a["priority"] == "high" for a in strong["advice"]))

    def test_full_result_structure(self):
        r = self.advisor.advise(EXAMPLE, TODAY)
        self.assertTrue(r["ok"])
        for key in ("summary", "goal", "courses", "timetable", "advice", "outlook", "model_focus"):
            self.assertIn(key, r)
        self.assertAlmostEqual(sum(r["summary"]["probabilities"].values()), 1.0, places=2)
        self.assertEqual(len(r["outlook"]), 4)
        order = [a["priority"] for a in r["advice"] if a["source"] == "rule"]
        self.assertEqual(order, sorted(order, key=["high", "medium", "low"].index))

    def test_invalid_input_returns_errors_not_exceptions(self):
        result = self.advisor.advise({"cgpa": "x"})
        self.assertFalse(result["ok"])
        self.assertIn("cgpa", result["errors"])

    def test_missing_scores_are_flagged(self):
        courses = [{"code": "A 1", "credits": 3}, {"code": "B 2", "credits": 3}]
        r = self.advisor.advise(profile(courses=courses), TODAY)
        self.assertTrue(r["summary"]["score_estimated"])

    def test_goal_status_values(self):
        out = self.advisor.advise(profile(target_cgpa=3.0), TODAY)
        self.assertEqual(out["goal"]["status"], "on_track")
        out = self.advisor.advise(profile(target_cgpa=5.0, cgpa=3.0, credits_earned=100), TODAY)
        self.assertEqual(out["goal"]["status"], "out_of_reach")


if __name__ == "__main__":
    unittest.main()
