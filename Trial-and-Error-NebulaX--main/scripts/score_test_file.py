"""Score any one .xlsx case file (training, validation, or the eventual real
test file -- any file matching this pipeline's schema) with the settled
model (Elliptic Envelope) and print/write a full ranking of its 8 cars.

Fits the standardizer and model on ALL 5 labeled files in data/train/
(reusing the category vocabulary already fit on them -- never refit on the
file being scored), scores the given file, and writes file_id,ranked_cars
in the exact PS3 submission format.

Usage:
    python scripts/score_test_file.py path/to/any_case.xlsx
    python scripts/score_test_file.py path/to/any_case.xlsx \
        --aggregation median --out results/my_submission.csv

--aggregation defaults to p90, the best-by-primary_metric choice from the
LOCO evaluation (see README.md's results table) -- pass --aggregation
median for the alternative aggregation also reported there.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from acv_fault import config, data_loading, preprocessing, submission

MODEL_NAME = "elliptic_envelope"  # the sole settled model -- see models/__init__.py


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("xlsx_path", type=Path, help="Path to the .xlsx case file to score")
    parser.add_argument("--aggregation", default="p90", choices=["median", "p90"])
    parser.add_argument(
        "--out", type=Path, default=config.RESULTS_DIR / "test_submission.csv", help="Output CSV path"
    )
    args = parser.parse_args()

    if not args.xlsx_path.exists():
        raise FileNotFoundError(f"{args.xlsx_path} does not exist")

    print(f"Loading + preparing the 5 labeled training files from {config.TRAIN_DIR} ...")
    full_long, _labels_df, _load_reports = data_loading.load_all_cases()
    prepared, report = preprocessing.prepare_global(full_long)

    print(f"\nScoring {args.xlsx_path} with model={MODEL_NAME!r}, aggregation={args.aggregation!r} ...")
    ranking = submission.score_new_file(args.xlsx_path, prepared, report, MODEL_NAME, args.aggregation)

    file_id = args.xlsx_path.name
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_df = submission.write_submission_csv({file_id: ranking}, args.out)

    print(f"\nRanking (most to least likely faulty): {' > '.join(ranking)}")
    print(f"\nWrote {args.out} (PS3 submission format):")
    print(out_df.to_string(index=False))


if __name__ == "__main__":
    main()
