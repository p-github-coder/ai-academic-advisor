# Academic Advisor

A hybrid **expert-system + machine-learning** academic advisor, built for CSC 303.
A student enters their profile, courses, study habits and goals. The app returns:

- **Performance assessment**: predicted next-semester standing (`At Risk`, `Fair`, `Good`, `Excellent`), expected GPA and projected CGPA
- **Course priority ranking**: which courses need the most attention, and why
- **Weekly study timetable**: available hours shared across courses by priority
- **Goal tracking**: the GPA needed this semester to reach the target CGPA, credit progress
- **Personalised recommendations**: ordered by urgency

| Layer | Technique | Where |
|---|---|---|
| AI (learned) | Random Forest classifier (standing) and regressor (semester GPA), trained on Google Colab, saved as `.pkl` | `notebooks/`, `models/advisor_model.pkl` |
| AI (symbolic) | Rule-based recommendations, course scoring, timetable and goal planner | `advisor/rules.py`, `advisor/planner.py` |
| Hybrid | Model output and rules merged into one plan | `advisor/engine.py` |
| Software engineering | Modules, input validation, 39 tests, linting, CI, Git history | everywhere |
| Deployment | Flask web app and JSON API loading the `.pkl` | `app.py`, `templates/`, `static/` |

> **Data is synthetic.** No real student records are used. The generator encodes plausible
> relationships (attendance, study time, confidence, difficulty, repeat courses, workload) plus noise,
> so the reported accuracy (about 78%) shows the workflow works, not how well it predicts real
> students. Swap in anonymised departmental data with `python train.py --csv your.csv`.

**What the ML model uses and what the rules use.** The model sees 13 numeric features: level, CGPA,
last semester GPA, average score, credit load, average difficulty, average confidence, study hours
per day, study days per week, attendance, assignment habits, repeat courses and the gap to the target
CGPA. Department, study method, challenges, exam date and materials drive the rule-based advice,
because there is no training data for them. If no scores are entered, the average score is estimated
from last semester's GPA and the app says so.

## Project layout

```
advisor/   data.py (generator, constants) · model.py (train/save/load) · profile.py (inputs,
           validation, feature derivation) · planner.py (priority, timetable, goal) ·
           rules.py (recommendations) · engine.py (hybrid advisor)
app.py     Flask app: web page, POST /api/advise, GET /api/example, GET /api/health
train.py   Local training CLI -> models/advisor_model.pkl
notebooks/ train_advisor_model.ipynb  (Google Colab)
models/    advisor_model.pkl
templates/ static/   dashboard page, styles, JavaScript
tests/     unit + integration tests (39)
scripts/   build_notebook.py (regenerates the notebook from the repo sources)
.vscode/   Python-optimised VS Code settings, debug configs, extension list
.github/   CI workflow (lint + tests on every push)
```

## 1. Prerequisites: Git, GitHub, VS Code

**Git** (check first: `git --version`). Windows: https://git-scm.com/download/win (accept defaults). Set your identity once:
```
git config --global user.name "Your Name"
git config --global user.email "you@example.com"
```
**GitHub**: create a free account at https://github.com.
**VS Code**: https://code.visualstudio.com. Open this folder (`File > Open Folder`) and accept the
prompt to install the recommended extensions (Python, Pylance, Ruff, Jupyter, GitHub Pull Requests,
GitLens). `.vscode/settings.json` enables format-on-save, pytest discovery and the project environment.

## 2. Set up and run locally

```bash
python -m venv .venv
# Windows PowerShell:  .venv\Scripts\Activate.ps1      macOS/Linux:  source .venv/bin/activate
pip install flask scikit-learn pandas numpy joblib
python app.py            # open http://127.0.0.1:5000
```

Click **Fill with example** then **Build my plan** to see a full result. If `models/advisor_model.pkl`
is missing, damaged or built with an incompatible scikit-learn, the app trains a new one
automatically on start-up (a few seconds) and saves it. You can also run `python train.py` yourself.

## 3. Train on Google Colab and deploy the `.pkl`

1. Go to https://colab.research.google.com > **File > Upload notebook** > choose `notebooks/train_advisor_model.ipynb`.
2. **Runtime > Run all.** It generates data, compares models with cross-validation, trains the
   classifier and regressor, evaluates them on a hold-out set and saves `advisor_model.pkl` (it also downloads it).
3. Put the downloaded file in `models/` (replace the existing one) and restart the app.

## 4. Test it

```bash
python -m unittest discover -s tests -t . -v     # no extra install needed
# or, with pytest:  pip install pytest && pytest -q
```

Tests cover data generation, validation, course ranking, timetable arithmetic, goal maths, the `.pkl`
round trip, every API route, error handling, and automatic recovery from a damaged model file.

**API example**

```bash
curl http://127.0.0.1:5000/api/example > example.json
curl -X POST http://127.0.0.1:5000/api/advise -H "Content-Type: application/json" -d @example.json
```

## 5. Push to GitHub

```bash
# create an empty repo named academic-advisor on github.com (no README), then:
git remote add origin https://github.com/<your-username>/academic-advisor.git
git branch -M main
git push -u origin main
```
With GitHub CLI instead: `gh repo create academic-advisor --public --source=. --push`.

## Design notes and limits

- **Why hybrid?** The models capture combinations of factors; the rules give transparent, actionable
  reasons a student can check. If the model says `At Risk` but no rule fires, the engine adds a
  model-sourced alert so the prediction is never silent.
- **Grading scale:** Nigerian 5-point scale (First 4.50+, 2:1 3.50+, 2:2 2.40+, Third 1.50+, Pass 1.00+).
  Thresholds such as 75% attendance, a 24-unit ceiling and one study hour per unit are common
  conventions; confirm them against your institution's handbook (`advisor/rules.py`).
- **Explainability:** the page shows the probability of each standing, the reasons behind every course
  priority, and the model's globally most influential inputs (not per-student attributions).
- **Privacy:** nothing is stored. Each request is processed in memory and discarded.
- **Not an official ruling.** Guidance only; see your course adviser for decisions.
- **Pickle safety:** only load `.pkl` files you trained or trust; unpickling can execute code.
