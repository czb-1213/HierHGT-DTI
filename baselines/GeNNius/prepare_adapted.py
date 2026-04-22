#!/usr/bin/env python
"""Build GeNNius caches for a HierHGT-DTI fixed split."""

import argparse
import os

from data_utils import build_cache, split_cache_paths


def main():
    parser = argparse.ArgumentParser(description="Prepare GeNNius adapted cache")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", required=True, choices=["random", "cold_drug", "cold_protein"])
    parser.add_argument(
        "--data_root",
        default=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "datasets")),
        help="Root directory containing <dataset>/<split>/train,val,test.csv",
    )
    parser.add_argument(
        "--output_root",
        default=os.path.join(os.path.dirname(__file__), "data_adapted"),
        help="Cache root directory",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild cache even if it already exists",
    )
    args = parser.parse_args()

    paths = split_cache_paths(args.output_root, args.dataset, args.split)
    if os.path.exists(paths.cache_pt) and not args.force:
        print(f"[SKIP] Cache already exists at {paths.cache_pt}")
        return

    paths = build_cache(
        dataset=args.dataset,
        split=args.split,
        data_root=args.data_root,
        output_root=args.output_root,
    )
    print(f"[DONE] Saved GeNNius cache to {paths.cache_pt}")
    print(f"[DONE] Saved metadata to {paths.metadata_json}")


if __name__ == "__main__":
    main()
