"""
Create deterministic train / validation / test splits for a JSONL dataset.

Rows are assigned in a round-robin pattern that approximates the requested
fractions *without* ever shuffling.  For example, with --val-fraction 0.33
and --test-fraction 0.33 the assignment pattern is:

    row 0 → train
    row 1 → validation
    row 2 → test
    row 3 → train
    row 4 → validation
    ...

Usage
-----

python make_splits.py --input math_500_and_MMLU_pro.jsonl

This writes:

    train.jsonl
    validation.jsonl
    test.jsonl

to the chosen output directory (current directory by default).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, TextIO


def deterministic_split(
    input_file: Path,
    output_dir: Path,
    val_frac: float,
    test_frac: float,
) -> None:
    # --- Sanity checks ----------------------------------------------------- #
    if not (0.0 <= val_frac < 1.0 and 0.0 <= test_frac < 1.0):
        raise ValueError("Fractions must satisfy 0 ≤ fraction < 1.")
    if val_frac + test_frac >= 1.0:
        raise ValueError("val_frac + test_frac must be < 1.")

    output_dir.mkdir(parents=True, exist_ok=True)

    destinations: Dict[str, TextIO] = {
        "train": (output_dir / "train.jsonl").open("w", encoding="utf-8"),
        "validation": (output_dir / "validation.jsonl").open("w", encoding="utf-8"),
        "test": (output_dir / "test.jsonl").open("w", encoding="utf-8"),
    }
    counts = {k: 0 for k in destinations}

    # Accumulators implement a Bresenham-style “evenly spaced” selection
    val_acc = 0.0
    test_acc = 0.0

    with input_file.open("r", encoding="utf-8") as src:
        for line in src:
            if not line.strip():          # skip blank / whitespace-only lines
                continue

            val_acc += val_frac
            test_acc += test_frac

            if val_acc >= 1.0:
                split = "validation"
                val_acc -= 1.0
            elif test_acc >= 1.0:
                split = "test"
                test_acc -= 1.0
            else:
                split = "train"

            destinations[split].write(line)
            counts[split] += 1

    for fp in destinations.values():
        fp.close()

    total = sum(counts.values())
    print(f"Finished writing splits to {output_dir.resolve()}")
    for name, n in counts.items():
        print(f"{name.title():>10}: {n:>6}  ({n/total:6.2%})")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Deterministically split a JSONL dataset into train/validation/test "
            "without shuffling."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to the combined JSONL file (e.g. math_500_and_MMLU_pro.jsonl)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("."),
        help="Directory to write split files (default: current directory)",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.10,
        help="Fraction of rows for the validation set (default: 0.10)",
    )
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=0.10,
        help="Fraction of rows for the test set (default: 0.10)",
    )
    args = parser.parse_args()

    deterministic_split(
        input_file=args.input,
        output_dir=args.output_dir,
        val_frac=args.val_fraction,
        test_frac=args.test_fraction,
    )


if __name__ == "__main__":
    main()
