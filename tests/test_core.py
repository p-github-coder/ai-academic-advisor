"""Unit tests for data generation, rules, and the model bundle (unittest; pytest runs these too)."""

import tempfile
import unittest
from pathlib import Path

from advisor.data import CLASSES, FEATURES, TARGET, degree_class, generate_dataset, gpa_to_standing
from advisor.engine import Advisor
from advisor.model import load_bundle, save_bundle, train_bundle
from advisor.rules import FIELDS, apply_rules, required_semester_gpa, validate_profile


def profile(**overrides):
    base = {name: f.default for name, f in FIELDS.items()}
    base.update(overrides)
    return base


class DataTests(unittest.TestCase):
    def test_shape_columns_and_labels(self):
        df = generate_dataset(300, seed=1)
        self.assertEqual(len(df), 300)
        self.assertEqual(list(df.columns), [*FEATURES, TARGET])
        self.assertTrue(set(df[TARGET]) <= set(CLASSES))

    def test_deterministic_for_same_seed(self):
        a = generate_dataset(100, seed=7)
        b = generate_dataset(100, seed=7)
        self.assertTrue(a.equals(b))

    def test_values_stay_in_range(self):
        df = generate_dataset(500, seed=3)
        self.assertTrue(df["cgpa"].between(0, 5).all())
        self.assertTrue(df["attendance_pct"].between(0, 100).all())

    def test_degree_class_boundaries(self):
        self.assertEqual(degree_class(4.50), "First Class")
        self.assertEqual(degree_class(4.49), "Second Class Upper")
        self.assertEqual(degree_class(2.40), "Second Class Lower")
        self.assertEqual(degree_class(0.5), "Below Pass")

    def test_standing_boundaries(self):
        self.assertEqual(gpa_to_standing(2.39), "At Risk")
        self.assertEqual(gpa_to_standing(2.40), "Fair")
        self.assertEqual(gpa_to_standing(4.5), "Excellent")


class ValidationTests(unittest.TestCase):
    def test_defaults_are_valid(self):
        clean, errors = validate_profile(profile())
        self.assertEqual(errors, {})
        self.assertEqual(set(clean), set(FIELDS))

    def test_rejects_out_of_range_missing_and_non_numeric(self):
        _, errors = validate_profile(profile(cgpa=7, attendance_pct="abc"))
        self.assertIn("cgpa", errors)
        self.assertIn("attendance_pct", errors)
        raw = profile()
        del raw["sleep_hours"]
        _, errors = validate_profile(raw)
        self.assertIn("sleep_hours", errors)

    def test_rejects_nan_and_inf(self):
        _, errors = validate_profile(profile(cgpa="nan", last_gpa="inf"))
        self.assertIn("cgpa", errors)
        self.assertIn("last_gpa", errors)

    def test_integer_fields_are_rounded(self):
        clean, _ = validate_profile(profile(carryovers="2.0"))
        self.assertEqual(clean["carryovers"], 2)


class RuleTests(unittest.TestCase):
    def clean(self, **o):
        clean, errors = validate_profile(profile(**o))
        self.assertEqual(errors, {})
        return clean

    def test_planner_math(self):
        # (3.0*60 + g*20) / 80 = 3.5  ->  g = 5.0
        self.assertAlmostEqual(required_semester_gpa(3.0, 60, 20, 3.5), 5.0)
        self.assertIsNone(required_semester_gpa(3.0, 0, 20, 3.5))

    def test_weak_student_gets_high_priority_advice(self):
        weak = self.clean(
            cgpa=2.0,
            last_gpa=1.4,
            attendance_pct=50,
            carryovers=3,
            credit_load=27,
            study_hours_week=4,
            assignment_pct=40,
            sleep_hours=4,
        )
        categories = {a.category for a in apply_rules(weak) if a.priority == "high"}
        self.assertTrue({"Attendance", "Carry-overs", "Workload", "Study time"} <= categories)

    def test_advice_is_sorted_high_first(self):
        weak = self.clean(attendance_pct=50, carryovers=2)
        order = [a.priority for a in apply_rules(weak)]
        self.assertEqual(order, sorted(order, key=["high", "medium", "low"].index))

    def test_strong_student_has_no_high_priority_advice(self):
        strong = self.clean(
            cgpa=4.6,
            last_gpa=4.7,
            attendance_pct=95,
            study_hours_week=25,
            assignment_pct=98,
            carryovers=0,
            credit_load=18,
            sleep_hours=7.5,
        )
        self.assertFalse([a for a in apply_rules(strong) if a.priority == "high"])


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = train_bundle(generate_dataset(1200, seed=5), seed=5)

    def test_bundle_has_expected_keys_and_decent_accuracy(self):
        for key in ("model", "feature_names", "classes", "metrics", "sklearn_version"):
            self.assertIn(key, self.bundle)
        self.assertEqual(self.bundle["feature_names"], FEATURES)
        self.assertGreater(self.bundle["metrics"]["accuracy"], 0.6)

    def test_pickle_round_trip_gives_identical_predictions(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "m.pkl"
            save_bundle(self.bundle, str(path))
            loaded = load_bundle(str(path))
        a, b = Advisor(self.bundle), Advisor(loaded)
        self.assertEqual(a.advise(profile())["probabilities"], b.advise(profile())["probabilities"])

    def test_load_rejects_wrong_format(self):
        import joblib

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.pkl"
            joblib.dump({"oops": 1}, path)
            with self.assertRaises(ValueError):
                load_bundle(str(path))

    def test_probabilities_sum_to_one_and_ordering_is_sensible(self):
        adv = Advisor(self.bundle)
        weak = adv.advise(
            profile(
                cgpa=1.8,
                last_gpa=1.5,
                attendance_pct=50,
                study_hours_week=4,
                assignment_pct=45,
                carryovers=4,
                sleep_hours=4.5,
            )
        )
        strong = adv.advise(
            profile(
                cgpa=4.7,
                last_gpa=4.8,
                attendance_pct=97,
                study_hours_week=25,
                assignment_pct=99,
                carryovers=0,
                sleep_hours=8,
            )
        )
        self.assertAlmostEqual(sum(weak["probabilities"].values()), 1.0, places=2)
        self.assertGreater(weak["probabilities"]["At Risk"], strong["probabilities"]["At Risk"])
        self.assertEqual(weak["predicted_standing"], "At Risk")

    def test_invalid_input_returns_errors_not_exceptions(self):
        result = Advisor(self.bundle).advise({"cgpa": "x"})
        self.assertFalse(result["ok"])
        self.assertIn("cgpa", result["errors"])


if __name__ == "__main__":
    unittest.main()
