#!/usr/bin/env python
"""Prepare cached fixed-split inputs for the adapted DO-GMA baseline."""

from __future__ import annotations

import argparse
import os

from data_utils import build_cache, split_cache_paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare DO-GMA adapted cache")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", required=True, choices=["random", "cold_drug", "cold_protein"])
    parser.add_argument(
        "--data_root",
        default=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "datasets")),
        help="Root directory containing <dataset>/<split>/train,val,test.csv",
    )
    parser.add_argument("--output_root", default=os.path.join(os.path.dirname(__file__), "data_adapted"))
    parser.add_argument("--max_protein_len", type=int, default=1200)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    cache_paths = split_cache_paths(args.output_root, args.dataset, args.split)
    if os.path.exists(cache_paths.cache_pt) and not args.force:
        print(f"[SKIP] Cache already exists: {cache_paths.cache_pt}")
        return

    cache_paths = build_cache(
        dataset=args.dataset,
        split=args.split,
        data_root=args.data_root,
        output_root=args.output_root,
        max_protein_len=args.max_protein_len,
    )
    print(f"[DONE] Saved DO-GMA cache to {cache_paths.cache_pt}")


if __name__ == "__main__":
    main()
