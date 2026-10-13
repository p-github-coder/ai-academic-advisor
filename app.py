"""Flask web app that serves the academic advisor.

Run:   python app.py            (development server on http://127.0.0.1:5000)
Or:    flask --app app run
Prod:  gunicorn "app:create_app()"

If models/advisor_model.pkl is missing or unreadable (for example a damaged download or a
scikit-learn version mismatch) the app retrains it automatically from the synthetic data
generator, which takes a few seconds, and saves a fresh copy.
"""

from __future__ import annotations

import os
from pathlib import Path

from flask import Flask, jsonify, render_template, request

from advisor import __version__
from advisor.data import generate_dataset
from advisor.engine import Advisor
from advisor.model import DEFAULT_MODEL_PATH, load_bundle, save_bundle, train_bundle
from advisor.profile import (
    ASSIGNMENT_LABELS,
    CHALLENGES,
    EXAMPLE,
    GOAL_TYPES,
    LEVELS,
    MATERIALS,
    METHODS,
    PERIODS,
    SEMESTERS,
)

ROOT = Path(__file__).resolve().parent

OPTIONS = {
    "levels": LEVELS,
    "semesters": SEMESTERS,
    "goal_types": GOAL_TYPES,
    "periods": PERIODS,
    "methods": METHODS,
    "challenges": CHALLENGES,
    "materials": MATERIALS,
    "assignment": ASSIGNMENT_LABELS,
}


def _load_advisor(path: Path, auto_train: bool) -> tuple[Advisor | None, str | None, str | None]:
    """Return (advisor, error, notice). `notice` explains an automatic retrain."""
    try:
        return Advisor(load_bundle(str(path))), None, None
    except Exception as exc:  # missing, damaged, wrong format, sklearn version mismatch...
        problem = (
            f"Model file not found at {path}."
            if isinstance(exc, FileNotFoundError)
            else f"Could not load the model at {path}: {exc}"
        )
        if not auto_train:
            return None, problem + " Run `python train.py` first.", None
    try:
        bundle = train_bundle(generate_dataset())
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            save_bundle(bundle, str(path))
        except OSError:
            pass  # read-only disk: keep the freshly trained model in memory only
        notice = f"{problem} A new model was trained automatically."
        print(notice)
        return Advisor(bundle), None, notice
    except Exception as exc:
        return None, f"{problem} Automatic retraining failed: {exc}", None


def create_app(model_path: str | os.PathLike | None = None, auto_train: bool = True) -> Flask:
    app = Flask(__name__)
    path = Path(model_path or os.environ.get("ADVISOR_MODEL", ROOT / DEFAULT_MODEL_PATH))
    advisor, load_error, notice = _load_advisor(path, auto_train)

    @app.get("/")
    def index():
        return render_template("index.html", options=OPTIONS, load_error=load_error)

    @app.post("/api/advise")
    def api_advise():
        if advisor is None:
            return jsonify({"ok": False, "error": load_error}), 503
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return jsonify({"ok": False, "errors": {"_": "Send a JSON object."}}), 400
        result = advisor.advise(payload)
        return jsonify(result), (200 if result["ok"] else 400)

    @app.get("/api/example")
    def api_example():
        return jsonify(EXAMPLE)

    @app.get("/api/health")
    def health():
        return jsonify(
            {
                "status": "ok" if advisor else "degraded",
                "model_loaded": advisor is not None,
                "version": __version__,
                "metrics": advisor.metrics if advisor else None,
                "notice": notice,
                "error": load_error,
            }
        )

    return app


if __name__ == "__main__":
    create_app().run(debug=os.environ.get("FLASK_DEBUG") == "1")
