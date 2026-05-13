#!/usr/bin/env python
"""Check fixed split files for label counts and cold-start entity leakage.

The script is intentionally conservative: it reports missing files instead of
failing silently, and it only computes overlap checks for complete
train/validation/test split directories.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = {"SMILES", "Protein", "Y"}
SPLIT_FILES = ("train.csv", "val.csv", "test.csv")


def read_split(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    return frame


def summarize_frame(frame: pd.DataFrame) -> str:
    labels = frame["Y"].value_counts().to_dict()
    return (
        f"rows={len(frame)}; "
        f"pos={int(labels.get(1, 0))}; neg={int(labels.get(0, 0))}; "
        f"drugs={frame['SMILES'].nunique()}; proteins={frame['Protein'].nunique()}"
    )


def check_split_dir(split_dir: Path, root: Path) -> list[str]:
    display_dir = split_dir.relative_to(root) if split_dir.is_relative_to(root) else split_dir
    lines = [f"## {display_dir.as_posix()}"]
    paths = {name: split_dir / name for name in SPLIT_FILES}
    missing = [name for name, path in paths.items() if not path.exists()]
    if missing:
        lines.append(f"missing={','.join(missing)}")
        for split_name, path in paths.items():
            if path.exists():
                frame = read_split(path)
                lines.append(f"{split_name.removesuffix('.csv')}: {summarize_frame(frame)}")
        return lines

    frames = {name.removesuffix(".csv"): read_split(path) for name, path in paths.items()}
    for split_name in ("train", "val", "test"):
        lines.append(f"{split_name}: {summarize_frame(frames[split_name])}")

    train_drugs = set(frames["train"]["SMILES"])
    train_proteins = set(frames["train"]["Protein"])
    test_drugs = set(frames["test"]["SMILES"])
    test_proteins = set(frames["test"]["Protein"])

    name = split_dir.name.lower()
    if "cold_drug" in name:
        lines.append(f"cold_drug_train_test_drug_overlap={len(train_drugs & test_drugs)}")
    if "cold_protein" in name:
        lines.append(f"cold_protein_train_test_protein_overlap={len(train_proteins & test_proteins)}")

    val_drug_overlap = len(set(frames["val"]["SMILES"]) & test_drugs)
    val_protein_overlap = len(set(frames["val"]["Protein"]) & test_proteins)
    lines.append(f"val_test_drug_overlap={val_drug_overlap}")
    lines.append(f"val_test_protein_overlap={val_protein_overlap}")
    return lines


def find_candidate_dirs(root: Path) -> list[Path]:
    dirs: set[Path] = set()
    for csv_path in root.rglob("*.csv"):
        if csv_path.name in SPLIT_FILES:
            dirs.add(csv_path.parent)
    return sorted(dirs)


def find_orphan_test_files(root: Path, split_dirs: list[Path]) -> list[Path]:
    split_test_paths = {split_dir / "test.csv" for split_dir in split_dirs}
    orphan_tests: list[Path] = []
    for csv_path in root.rglob("*_test.csv"):
        if csv_path not in split_test_paths:
            orphan_tests.append(csv_path)
    return sorted(orphan_tests)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."), help="Dataset root directory")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("split_integrity_report.md"),
        help="Markdown report path",
    )
    args = parser.parse_args()

    root = args.root.resolve()
    lines = ["# Split Integrity Report", "", f"root={args.root}", ""]

    dirs = find_candidate_dirs(root)
    if not dirs:
        lines.append("No split directories with train/val/test CSV files were found.")
    for split_dir in dirs:
        lines.extend(check_split_dir(split_dir, root))
        lines.append("")

    orphan_tests = find_orphan_test_files(root, dirs)
    if orphan_tests:
        lines.append("## Single test files without local train/val companions")
        for path in orphan_tests:
            frame = read_split(path)
            display_path = path.relative_to(root) if path.is_relative_to(root) else path
            lines.append(f"{display_path.as_posix()}: {summarize_frame(frame)}")
        lines.append("")

    args.output.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
