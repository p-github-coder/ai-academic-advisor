"""Integration tests: the .pkl is loaded by the Flask app and exercised over HTTP."""

import tempfile
import unittest
from pathlib import Path

from advisor.data import generate_dataset
from advisor.model import save_bundle, train_bundle
from advisor.rules import FIELDS
from app import create_app

GOOD = {name: f.default for name, f in FIELDS.items()}
WEAK = {
    **GOOD,
    "cgpa": 1.9,
    "last_gpa": 1.4,
    "attendance_pct": 50,
    "study_hours_week": 4,
    "assignment_pct": 45,
    "carryovers": 4,
    "sleep_hours": 4.5,
    "credit_load": 27,
}


class AppTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        path = Path(cls.tmp.name) / "model.pkl"
        save_bundle(train_bundle(generate_dataset(1200, seed=11), seed=11), str(path))
        cls.client = create_app(path).test_client()

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_health_reports_model_loaded(self):
        data = self.client.get("/api/health").get_json()
        self.assertEqual(data["status"], "ok")
        self.assertTrue(data["model_loaded"])

    def test_home_page_renders_form(self):
        res = self.client.get("/")
        self.assertEqual(res.status_code, 200)
        self.assertIn(b"Get advice", res.data)
        for name in FIELDS:
            self.assertIn(f'name="{name}"'.encode(), res.data)

    def test_api_returns_prediction_and_advice(self):
        res = self.client.post("/api/advise", json=WEAK)
        body = res.get_json()
        self.assertEqual(res.status_code, 200)
        self.assertEqual(body["predicted_standing"], "At Risk")
        self.assertTrue(any(a["priority"] == "high" for a in body["advice"]))

    def test_api_rejects_bad_input_with_400(self):
        res = self.client.post("/api/advise", json={**GOOD, "cgpa": 12})
        self.assertEqual(res.status_code, 400)
        self.assertIn("cgpa", res.get_json()["errors"])

    def test_api_rejects_non_json(self):
        res = self.client.post("/api/advise", data="nope", content_type="text/plain")
        self.assertEqual(res.status_code, 400)

    def test_form_post_shows_result_page(self):
        res = self.client.post("/", data={k: str(v) for k, v in WEAK.items()})
        self.assertEqual(res.status_code, 200)
        self.assertIn(b"Recommendations", res.data)
        self.assertIn(b"At Risk", res.data)

    def test_form_post_shows_field_errors(self):
        res = self.client.post("/", data={**{k: str(v) for k, v in GOOD.items()}, "cgpa": "9"})
        self.assertEqual(res.status_code, 400)
        self.assertIn(b"Must be between", res.data)

    def test_missing_model_degrades_gracefully(self):
        client = create_app("does/not/exist.pkl").test_client()
        self.assertEqual(client.get("/api/health").get_json()["status"], "degraded")
        self.assertEqual(client.post("/api/advise", json=GOOD).status_code, 503)
        self.assertEqual(client.get("/").status_code, 200)


if __name__ == "__main__":
    unittest.main()
