from __future__ import annotations

import argparse
import sys

from .model import predict, train


def _print_prediction(result: dict[str, float]) -> None:
    team_a = result["team_a"]
    team_b = result["team_b"]
    rows = [
        (f"{team_a} win", result["win"]),
        ("Draw", result["draw"]),
        (f"{team_b} win", result["loss"]),
    ]
    pick_label, pick_probability = max(rows, key=lambda item: item[1])
    print("=" * 56)
    print(f"{team_a} vs {team_b}")
    print("=" * 56)
    for label, probability in rows:
        print(f"{label:<28} {probability * 100:6.1f}%")
    print("-" * 56)
    print(f"PICK: {pick_label} ({pick_probability * 100:.1f}%)")
    print("=" * 56)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="worldcup-predict",
        description=(
            "Train the model or predict neutral FIFA World Cup-style match "
            "probabilities."
        ),
        usage=(
            "worldcup-predict train [--force-download]\n"
            "       worldcup-predict <team_a> <team_b>"
        ),
    )
    parser.add_argument(
        "team_a",
        nargs="?",
        help="First team, prediction perspective.",
    )
    parser.add_argument(
        "team_b",
        nargs="?",
        help="Second team.",
    )
    return parser


def build_train_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="worldcup-predict train",
        description="Download data and train the model.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Refresh cached CSV files before training.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]

    if argv and argv[0] == "train":
        args = build_train_parser().parse_args(argv[1:])
        result = train(force_download=args.force_download)
        print(
            "Trained on "
            f"{result.rows} team-perspective rows. "
            f"Removed {result.removed_missing_scores} rows with missing scores. "
            f"Validation accuracy: {result.accuracy:.3f}. "
            f"Log loss: {result.log_loss:.3f}."
        )
        return 0

    parser = build_parser()
    args = parser.parse_args(argv)

    if args.team_a and args.team_b:
        try:
            _print_prediction(predict(args.team_a, args.team_b))
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 2
        return 0

    if args.team_a or args.team_b:
        parser.error("prediction requires exactly two team names")

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
