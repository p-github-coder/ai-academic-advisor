"""Integration tests: the .pkl is loaded by the Flask app and exercised over HTTP."""

import copy
import tempfile
import unittest
from pathlib import Path

from advisor.data import generate_dataset
from advisor.model import load_bundle, save_bundle, train_bundle
from advisor.profile import EXAMPLE
from app import create_app


class AppTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.path = Path(cls.tmp.name) / "model.pkl"
        save_bundle(train_bundle(generate_dataset(1500, seed=11), seed=11), str(cls.path))
        cls.client = create_app(cls.path).test_client()

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_health_reports_model_loaded(self):
        data = self.client.get("/api/health").get_json()
        self.assertEqual(data["status"], "ok")
        self.assertTrue(data["model_loaded"])
        self.assertIn("gpa_mae", data["metrics"])

    def test_home_page_has_the_form_and_assets(self):
        res = self.client.get("/")
        self.assertEqual(res.status_code, 200)
        for text in (b"Build my plan", b"About you", b"Study habits", b"app.js", b"style.css"):
            self.assertIn(text, res.data)
        for name in (b'name="cgpa"', b'name="target_cgpa"', b'name="hours_per_day"'):
            self.assertIn(name, res.data)

    def test_static_files_are_served(self):
        self.assertEqual(self.client.get("/static/app.js").status_code, 200)
        self.assertEqual(self.client.get("/static/style.css").status_code, 200)

    def test_example_endpoint_round_trips_through_advise(self):
        example = self.client.get("/api/example").get_json()
        res = self.client.post("/api/advise", json=example)
        self.assertEqual(res.status_code, 200)
        body = res.get_json()
        self.assertTrue(body["ok"])
        self.assertEqual(body["courses"][0]["rank"], 1)
        self.assertGreater(len(body["timetable"]["days"]), 0)

    def test_api_rejects_bad_input_with_field_errors(self):
        bad = copy.deepcopy(EXAMPLE)
        bad["cgpa"] = 12
        bad["courses"][0]["credits"] = 99
        res = self.client.post("/api/advise", json=bad)
        self.assertEqual(res.status_code, 400)
        errors = res.get_json()["errors"]
        self.assertIn("cgpa", errors)
        self.assertIn("courses.0.credits", errors)

    def test_api_rejects_non_json(self):
        res = self.client.post("/api/advise", data="nope", content_type="text/plain")
        self.assertEqual(res.status_code, 400)
        self.assertIn("_", res.get_json()["errors"])

    def test_missing_model_degrades_gracefully_without_autotrain(self):
        client = create_app("does/not/exist.pkl", auto_train=False).test_client()
        self.assertEqual(client.get("/api/health").get_json()["status"], "degraded")
        self.assertEqual(client.post("/api/advise", json=EXAMPLE).status_code, 503)
        self.assertEqual(client.get("/").status_code, 200)

    def test_damaged_model_is_retrained_automatically(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "advisor_model.pkl"
            path.write_bytes(self.path.read_bytes()[:5000])  # truncated, like a bad download
            client = create_app(path).test_client()
            health = client.get("/api/health").get_json()
            self.assertEqual(health["status"], "ok")
            self.assertIn("trained automatically", health["notice"])
            self.assertEqual(client.post("/api/advise", json=EXAMPLE).status_code, 200)
            load_bundle(str(path))  # a fresh, readable file was written back


if __name__ == "__main__":
    unittest.main()
