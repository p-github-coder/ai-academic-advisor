"""Flask web app that serves the academic advisor.

Run:   python app.py            (development server on http://127.0.0.1:5000)
Or:    flask --app app run
Prod:  gunicorn "app:create_app()"
"""

from __future__ import annotations

import os
from pathlib import Path

from flask import Flask, jsonify, render_template, request

from advisor import __version__
from advisor.engine import Advisor
from advisor.model import DEFAULT_MODEL_PATH, load_bundle
from advisor.rules import FIELDS

ROOT = Path(__file__).resolve().parent


def create_app(model_path: str | os.PathLike | None = None) -> Flask:
    app = Flask(__name__)
    path = Path(model_path or os.environ.get("ADVISOR_MODEL", ROOT / DEFAULT_MODEL_PATH))

    advisor: Advisor | None = None
    load_error: str | None = None
    try:
        advisor = Advisor(load_bundle(str(path)))
    except FileNotFoundError:
        load_error = f"Model file not found at {path}. Run `python train.py` first."
    except Exception as exc:  # corrupt file, sklearn version mismatch, wrong format...
        load_error = f"Could not load model from {path}: {exc}"

    @app.get("/")
    def index():
        return render_template(
            "index.html",
            fields=FIELDS,
            values={k: f.default for k, f in FIELDS.items()},
            errors={},
            result=None,
            load_error=load_error,
        )

    @app.post("/")
    def index_post():
        if advisor is None:
            return (
                render_template(
                    "index.html",
                    fields=FIELDS,
                    values=request.form,
                    errors={},
                    result=None,
                    load_error=load_error,
                ),
                503,
            )
        result = advisor.advise(request.form.to_dict())
        errors = {} if result["ok"] else result["errors"]
        return (
            render_template(
                "index.html",
                fields=FIELDS,
                values=request.form,
                errors=errors,
                result=result if result["ok"] else None,
                load_error=None,
            ),
            200 if result["ok"] else 400,
        )

    @app.post("/api/advise")
    def api_advise():
        if advisor is None:
            return jsonify({"ok": False, "error": load_error}), 503
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return jsonify({"ok": False, "error": "Send a JSON object."}), 400
        result = advisor.advise(payload)
        return jsonify(result), (200 if result["ok"] else 400)

    @app.get("/api/health")
    def health():
        return jsonify(
            {
                "status": "ok" if advisor else "degraded",
                "model_loaded": advisor is not None,
                "version": __version__,
                "metrics": advisor.metrics if advisor else None,
                "error": load_error,
            }
        )

    return app


if __name__ == "__main__":
    create_app().run(debug=os.environ.get("FLASK_DEBUG") == "1")
