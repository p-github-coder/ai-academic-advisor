"""Train the advisor model locally and write models/advisor_model.pkl.

    python train.py                      # generate synthetic data and train
    python train.py --csv data/my.csv    # train on your own CSV (FEATURES + next_standing)

The Colab notebook in notebooks/ produces the same kind of .pkl; either works with the app.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from advisor.data import FEATURES, TARGET, generate_dataset
from advisor.model import DEFAULT_MODEL_PATH, save_bundle, train_bundle


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--csv", help="CSV with feature columns and a next_standing column")
    parser.add_argument("--out", default=DEFAULT_MODEL_PATH, help="where to write the .pkl")
    parser.add_argument("--rows", type=int, default=4000, help="synthetic rows if no --csv")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.csv:
        df = pd.read_csv(args.csv)
        missing = [c for c in [*FEATURES, TARGET] if c not in df.columns]
        if missing:
            raise SystemExit(f"{args.csv} is missing columns: {missing}")
    else:
        df = generate_dataset(args.rows, args.seed)
        Path("data").mkdir(exist_ok=True)
        df.to_csv("data/students.csv", index=False)

    bundle = train_bundle(df, args.seed)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    save_bundle(bundle, args.out)
    m = bundle["metrics"]
    print(f"Saved {args.out}  accuracy={m['accuracy']}  macro_f1={m['macro_f1']}")


if __name__ == "__main__":
    main()
