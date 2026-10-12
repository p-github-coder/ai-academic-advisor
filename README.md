# Academic Advisor

A hybrid **expert-system + machine-learning** academic advisor, built for CSC 303.
A student enters their study habits and results; the app predicts next-semester standing
(`At Risk`, `Fair`, `Good`, `Excellent`) and returns prioritised, explainable advice.

| Layer | Technique | Where |
|---|---|---|
| AI (learned) | Random Forest classifier, trained on Google Colab, saved as `.pkl` | `notebooks/`, `models/advisor_model.pkl` |
| AI (symbolic) | Rule-based expert system + CGPA target planner | `advisor/rules.py` |
| Hybrid | Model prediction + rules merged into one answer | `advisor/engine.py` |
| Software engineering | Modules, input validation, tests, linting, CI, Git history | everywhere |
| Deployment | Flask web app + JSON API loading the `.pkl` | `app.py` |

> **Data is synthetic.** No real student records are used. The generator encodes plausible
> relationships (attendance, study time, carry-overs, workload, sleep) plus noise, so reported
> accuracy (~79%) shows the workflow works, not how well it predicts real students. Swap in
> anonymised departmental data via `python train.py --csv your.csv` (see `advisor/data.py` for columns).

## Project layout

```
advisor/        data.py (generator + constants) · model.py (train/save/load) · rules.py (expert rules,
                validation, CGPA planner) · engine.py (hybrid advisor)
app.py          Flask app: web form, POST /api/advise, GET /api/health
train.py        Local training CLI -> models/advisor_model.pkl
notebooks/      train_advisor_model.ipynb  (Google Colab)
models/         advisor_model.pkl
tests/          unit + integration tests (26)
scripts/        build_notebook.py (regenerates the notebook from repo sources)
.vscode/        Python-optimised VS Code settings, debug configs, extension list
.github/        CI workflow (lint + tests on every push)
```

## 1. Prerequisites: Git, GitHub, VS Code

**Git** (check first: `git --version`)
- Windows: install from https://git-scm.com/download/win (accept defaults). macOS: `xcode-select --install`.
- Set your identity once:
  ```
  git config --global user.name "Your Name"
  git config --global user.email "you@example.com"
  ```

**GitHub**: create a free account at https://github.com, then authenticate (HTTPS + browser, or
`gh auth login` if you install GitHub CLI).

**VS Code**: install from https://code.visualstudio.com, open this folder (`File > Open Folder`),
and accept the prompt to install the recommended extensions (Python, Pylance, Ruff, Jupyter, GitHub
Pull Requests, GitLens). `.vscode/settings.json` already enables format-on-save with Ruff, pytest
discovery and the project virtual environment.

## 2. Set up and run locally

```bash
python -m venv .venv
# Windows PowerShell:  .venv\Scripts\Activate.ps1      macOS/Linux:  source .venv/bin/activate
pip install -r requirements.txt
python app.py            # open http://127.0.0.1:5000
```

The repo already contains a trained `models/advisor_model.pkl`, so the app works immediately.
To retrain locally instead of using Colab: `python train.py`.

## 3. Train on Google Colab and deploy the `.pkl`

1. Go to https://colab.research.google.com > **File > Upload notebook** > choose
   `notebooks/train_advisor_model.ipynb` (or open it from GitHub once pushed).
2. **Runtime > Run all.** It generates data, compares three models with cross-validation,
   evaluates the Random Forest on a hold-out set, and saves `advisor_model.pkl` (it also downloads it).
3. Put the downloaded file in `models/` (replace the existing one) and restart the app.
4. If loading fails with a version error, install the scikit-learn version printed at the top of
   the notebook: `pip install scikit-learn==<that version>`.

## 4. Test it

```bash
python -m unittest discover -s tests -t . -v     # no extra install needed
# or, with pytest:  pip install pytest && pytest -q
```

Tests cover data generation, validation, each rule, planner arithmetic, `.pkl` save/load round trip,
the Flask routes (form, JSON API, error handling, missing-model fallback) and a sanity check that a
weak profile is rated riskier than a strong one.

Manual check in the browser: submit the defaults (expect `Fair`/`Good`), then try low attendance,
3 carry-overs and a 27-unit load (expect `At Risk` with high-priority advice).

**API example**

```bash
curl -X POST http://127.0.0.1:5000/api/advise -H "Content-Type: application/json" -d '{
  "cgpa": 2.1, "last_gpa": 1.6, "credits_earned": 90, "credit_load": 27,
  "attendance_pct": 55, "study_hours_week": 5, "assignment_pct": 50,
  "carryovers": 3, "sleep_hours": 4.5, "extracurricular_hours": 18 }'
```

## 5. Push to GitHub

```bash
# create an empty repo named academic-advisor on github.com (no README), then:
git remote add origin https://github.com/<your-username>/academic-advisor.git
git branch -M main
git push -u origin main
```

With GitHub CLI instead: `gh repo create academic-advisor --public --source=. --push`.

If the commit author shows the wrong name, fix it once with
`git commit --amend --reset-author --no-edit` (after setting `user.name`/`user.email`) for the latest
commit, or `git rebase -r --root --exec "git commit --amend --reset-author --no-edit"` for all of them.

## Design notes and limits

- **Why hybrid?** The model captures subtle combinations of factors; the rules give transparent,
  actionable reasons a student can check. If the model says `At Risk` but no rule fires, the engine
  adds a model-sourced alert so the prediction is never silent.
- **Grading scale:** Nigerian 5-point scale (First 4.50+, 2:1 3.50+, 2:2 2.40+, Third 1.50+, Pass 1.00+).
  Thresholds such as 75% attendance and a 24-unit load ceiling are common conventions; confirm them
  against your institution's handbook (`advisor/rules.py`).
- **Explainability:** the UI shows model confidence per class and the model's globally most
  influential inputs. These are not per-student attributions.
- **Not an official ruling.** Guidance only; see your course adviser for decisions.
- **Pickle safety:** only load `.pkl` files you trained or trust; unpickling can execute code.
