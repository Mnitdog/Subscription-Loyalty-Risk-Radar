from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from . import config, data_prep
from .evaluate_models import main as evaluate_main
from .loyalty_scoring import score_all_customers
from .train_frequency_model import train_frequency_model
from .train_subscription_model import train_subscription_model


def cmd_prepare_data(args: argparse.Namespace) -> None:
    df_clean = data_prep.load_clean()
    print(f"Prepared clean dataset with {len(df_clean)} rows.")


def cmd_train_all(args: argparse.Namespace) -> None:
    print("Training subscription model...")
    train_subscription_model()
    print("Training frequency model...")
    train_frequency_model()
    print("Done.")


def cmd_evaluate(args: argparse.Namespace) -> None:
    evaluate_main()


def cmd_score_customers(args: argparse.Namespace) -> None:
    df_scored = score_all_customers()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_scored.to_parquet(output_path, index=False)
    print(f"Saved scored customers to {output_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Subscription-Loyalty-Risk-Radar CLI"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_prepare = subparsers.add_parser("prepare-data", help="Clean and cache the dataset.")
    p_prepare.set_defaults(func=cmd_prepare_data)

    p_train = subparsers.add_parser("train-all", help="Train all models.")
    p_train.set_defaults(func=cmd_train_all)

    p_eval = subparsers.add_parser("evaluate", help="Evaluate trained models.")
    p_eval.set_defaults(func=cmd_evaluate)

    p_score = subparsers.add_parser("score-customers", help="Score all customers.")
    p_score.add_argument(
        "--output",
        type=str,
        default="data/processed/scored_customers.parquet",
        help="Output parquet file path.",
    )
    p_score.set_defaults(func=cmd_score_customers)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
