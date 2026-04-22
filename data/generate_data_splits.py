import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# =========================
# Config
# =========================
RANDOM_SEED = 42
TRAIN_RATIO, VAL_RATIO, TEST_RATIO = 0.7, 0.1, 0.2  # 7:1:2
MAX_RESAMPLE_TRIES = 200  # resample upper bound

np.random.seed(RANDOM_SEED)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR)

DATASET_SPECS = [
    ("BioSnap", os.path.join(DATA_DIR, "BioSnap", "full.csv")),
    ("DrugBank", os.path.join(DATA_DIR, "DrugBank", "full.csv")),
]

REQUIRED_COLS = ["SMILES", "Protein", "Y"]


# =========================
# Helpers
# =========================
def read_and_validate(csv_path: str, dataset_name: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"[{dataset_name}] File not found: {csv_path}")

    df = pd.read_csv(csv_path)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"[{dataset_name}] Missing columns {missing}. Got: {df.columns.tolist()}")

    df = df[REQUIRED_COLS].dropna().copy()

    # Ensure Y is binary 0/1
    df["Y"] = pd.to_numeric(df["Y"], errors="coerce")
    df = df.dropna(subset=["Y"]).copy()
    unique_y = set(df["Y"].unique().tolist())
    if not unique_y.issubset({0, 1}):
        raise ValueError(f"[{dataset_name}] Y must be 0/1. Found: {sorted(list(unique_y))[:20]}")
    df["Y"] = df["Y"].astype(int)

    # Pair-level de-dup
    before = len(df)
    df = df.drop_duplicates(subset=["SMILES", "Protein"], keep="first").copy()
    after = len(df)

    if df["Y"].nunique() < 2:
        raise ValueError(f"[{dataset_name}] After cleaning, only one class remains in Y.")

    print(f"[{dataset_name}] Loaded: {before} rows -> dedup pairs: {after} rows")
    print(f"[{dataset_name}] Unique drugs={df['SMILES'].nunique()}, proteins={df['Protein'].nunique()}")
    print(f"[{dataset_name}] Label ratio={df['Y'].value_counts(normalize=True).to_dict()}")
    return df


def label_stats(df: pd.DataFrame) -> str:
    pos = int((df["Y"] == 1).sum())
    neg = int((df["Y"] == 0).sum())
    total = len(df)
    if total == 0:
        return "total=0, pos=0, neg=0"
    return f"total={total}, pos={pos} ({pos/total:.2%}), neg={neg} ({neg/total:.2%})"


def assert_two_classes(df: pd.DataFrame, split_name: str, dataset_name: str):
    if df["Y"].nunique() < 2:
        raise AssertionError(f"[{dataset_name}][{split_name}] Only one class in this split.")


def save_split(out_dir: str, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame):
    os.makedirs(out_dir, exist_ok=True)
    train_df.to_csv(os.path.join(out_dir, "train.csv"), index=False)
    val_df.to_csv(os.path.join(out_dir, "val.csv"), index=False)
    test_df.to_csv(os.path.join(out_dir, "test.csv"), index=False)


def check_cold_overlap(train_df: pd.DataFrame, other_df: pd.DataFrame, col: str, dataset_name: str, split_name: str):
    overlap = set(train_df[col].astype(str).unique()) & set(other_df[col].astype(str).unique())
    if len(overlap) != 0:
        raise AssertionError(
            f"[{dataset_name}][{split_name}] Cold constraint violated on {col}: overlap={len(overlap)}"
        )


def _entity_group_stats(df: pd.DataFrame, key: str) -> pd.DataFrame:
    """
    Return per-entity stats: entity, n_pairs, pos, neg
    """
    g = df.groupby(key)["Y"]
    stats = g.agg(
        n_pairs="count",
        pos=lambda x: int((x == 1).sum()),
        neg=lambda x: int((x == 0).sum()),
    ).reset_index()
    stats[key] = stats[key].astype(str)
    return stats


def _greedy_assign_entities(stats: pd.DataFrame, key: str, seed: int, total_pairs: int):
    """
    Greedy bin packing by #pairs to match target pair counts for train/val/test.
    Entities are assigned to exactly one split => strict cold on (key).
    """
    rng = np.random.RandomState(seed)
    stats = stats.copy()

    # Shuffle first to avoid deterministic tie bias, then sort by size desc (better bin packing)
    perm = rng.permutation(len(stats))
    stats = stats.iloc[perm].reset_index(drop=True)
    stats = stats.sort_values("n_pairs", ascending=False).reset_index(drop=True)

    targets = {
        "train": int(round(TRAIN_RATIO * total_pairs)),
        "val": int(round(VAL_RATIO * total_pairs)),
        "test": total_pairs - int(round(TRAIN_RATIO * total_pairs)) - int(round(VAL_RATIO * total_pairs)),
    }

    buckets = {"train": [], "val": [], "test": []}
    counts = {"train": 0, "val": 0, "test": 0}
    pos_counts = {"train": 0, "val": 0, "test": 0}
    neg_counts = {"train": 0, "val": 0, "test": 0}

    def cost(split: str, add_pairs: int):
        # Penalize deviation from target (relative)
        tgt = max(1, targets[split])
        new = counts[split] + add_pairs
        return abs(new - tgt) / tgt

    for _, row in stats.iterrows():
        ent = row[key]
        n_pairs = int(row["n_pairs"])
        pos = int(row["pos"])
        neg = int(row["neg"])

        # choose split with minimal cost; tie broken by rng
        split_candidates = ["train", "val", "test"]
        costs = [(cost(s, n_pairs), s) for s in split_candidates]
        min_cost = min(c[0] for c in costs)
        best_splits = [s for c, s in costs if abs(c - min_cost) < 1e-12]
        chosen = rng.choice(best_splits)

        buckets[chosen].append(ent)
        counts[chosen] += n_pairs
        pos_counts[chosen] += pos
        neg_counts[chosen] += neg

    return buckets, counts, pos_counts, neg_counts, targets


def _repair_missing_class(df: pd.DataFrame, key: str, buckets: dict, seed: int):
    """
    If some split has only one class, try to move a small entity from other splits to fix it.
    This keeps strict cold (entities still unique per split).
    """
    rng = np.random.RandomState(seed)

    # Build entity -> label availability
    ent_stats = _entity_group_stats(df, key).set_index(key)

    def split_has_two_classes(split_name: str):
        ents = buckets[split_name]
        if len(ents) == 0:
            return False
        sub = df[df[key].astype(str).isin(ents)]
        return sub["Y"].nunique() == 2

    # Try up to limited attempts
    for _ in range(200):
        bad_splits = [s for s in ["train", "val", "test"] if not split_has_two_classes(s)]
        if not bad_splits:
            return buckets  # success

        # pick one bad split
        bad = bad_splits[0]
        sub_bad = df[df[key].astype(str).isin(buckets[bad])]
        present = set(sub_bad["Y"].unique().tolist())
        missing = 0 if 0 not in present else 1  # which label is missing

        # find a donor entity from other splits that contains the missing label
        donors = []
        for s in ["train", "val", "test"]:
            if s == bad:
                continue
            for ent in buckets[s]:
                if ent not in ent_stats.index:
                    continue
                # donor must contain at least one sample of missing label
                # i.e., if missing==1, need pos>0; if missing==0, need neg>0
                pos = int(ent_stats.loc[ent, "pos"])
                neg = int(ent_stats.loc[ent, "neg"])
                if (missing == 1 and pos > 0) or (missing == 0 and neg > 0):
                    donors.append((s, ent, int(ent_stats.loc[ent, "n_pairs"])))

        if not donors:
            break

        # prefer moving smaller entities to reduce ratio distortion
        donors.sort(key=lambda x: x[2])
        top_k = donors[:min(30, len(donors))]
        donor_split, donor_ent, _ = top_k[rng.randint(len(top_k))]

        # move donor_ent from donor_split -> bad
        buckets[donor_split].remove(donor_ent)
        buckets[bad].append(donor_ent)

    return buckets


# =========================
# Split implementations
# =========================
def split_random(df: pd.DataFrame, dataset_name: str, seed: int = None):
    """Pair-level random split: strict 7/1/2 with stratify."""
    if seed is None:
        seed = RANDOM_SEED
    train_val_df, test_df = train_test_split(
        df,
        test_size=TEST_RATIO,
        random_state=seed,
        stratify=df["Y"],
    )
    val_size_within_train_val = VAL_RATIO / (TRAIN_RATIO + VAL_RATIO)  # 0.125
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_size_within_train_val,
        random_state=seed,
        stratify=train_val_df["Y"],
    )

    assert_two_classes(train_df, "random/train", dataset_name)
    assert_two_classes(val_df, "random/val", dataset_name)
    assert_two_classes(test_df, "random/test", dataset_name)
    return train_df, val_df, test_df


def split_cold(df: pd.DataFrame, dataset_name: str, mode: str, seed: int = None):
    """
    Entity-level cold split: 7:1:2 on entity count.

    - train/val/test entities are fully disjoint on key.
    - Pair ratio depends on entity degree distribution (not controlled).
    - Label distribution tends to be balanced due to many entities per split.
    """
    if seed is None:
        seed = RANDOM_SEED
    key = "SMILES" if mode == "cold_drug" else "Protein"
    entities = df[key].astype(str).unique()

    rng = np.random.RandomState(seed)
    rng.shuffle(entities)

    n = len(entities)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)

    # avoid empty splits
    n_train = max(1, min(n_train, n - 2))
    n_val = max(1, min(n_val, n - n_train - 1))
    n_test = n - n_train - n_val
    if n_test <= 0:
        n_test = 1
        if n_val > 1:
            n_val -= 1
        else:
            n_train -= 1

    train_entities = set(entities[:n_train])
    val_entities = set(entities[n_train:n_train + n_val])
    test_entities = set(entities[n_train + n_val:])

    df_str = df.copy()
    df_str[key] = df_str[key].astype(str)
    train_df = df_str[df_str[key].isin(train_entities)].copy()
    val_df = df_str[df_str[key].isin(val_entities)].copy()
    test_df = df_str[df_str[key].isin(test_entities)].copy()

    # disjoint check
    check_cold_overlap(train_df, test_df, key, dataset_name, f"{mode} (train/test)")
    check_cold_overlap(val_df, test_df, key, dataset_name, f"{mode} (val/test)")
    check_cold_overlap(train_df, val_df, key, dataset_name, f"{mode} (train/val)")

    assert_two_classes(train_df, f"{mode}/train", dataset_name)
    assert_two_classes(val_df, f"{mode}/val", dataset_name)
    assert_two_classes(test_df, f"{mode}/test", dataset_name)

    total_pairs = len(df)
    def ratio(x): return x / max(1, total_pairs)
    print(f"[{dataset_name}][{mode}] entity split: {len(train_entities)}/{len(val_entities)}/{len(test_entities)}")
    print(f"[{dataset_name}][{mode}] actual pairs ratio train/val/test = "
          f"{ratio(len(train_df)):.3f}/{ratio(len(val_df)):.3f}/{ratio(len(test_df)):.3f}")

    return train_df, val_df, test_df


def split_cold_hybrid(df: pd.DataFrame, dataset_name: str, mode: str, seed: int = None):
    """
    Hybrid cold split: test-only cold via entity-count + stratified train/val.

    - Test entities are selected by entity-count (20% of entities), cold w.r.t. train/val.
    - Train/val are pair-level stratified split on the remaining data (shared entities OK).
    - Label distribution is balanced across all splits.
    - Val signal is clean for early stopping (no cold noise).
    """
    if seed is None:
        seed = RANDOM_SEED
    key = "SMILES" if mode == "cold_drug" else "Protein"
    df = df.copy()
    df[key] = df[key].astype(str)
    entities = df[key].unique()

    rng = np.random.RandomState(seed)
    rng.shuffle(entities)

    n = len(entities)
    n_test = max(1, int(n * TEST_RATIO))
    test_entities = set(entities[-n_test:])
    non_test_entities = set(entities[:-n_test])

    if len(test_entities) == 0 or len(non_test_entities) == 0:
        raise AssertionError(f"[{dataset_name}][{mode}] Cannot split into non-empty test/non-test.")

    test_df = df[df[key].isin(test_entities)].copy()
    non_test_df = df[df[key].isin(non_test_entities)].copy()

    if len(non_test_df) == 0 or len(test_df) == 0:
        raise AssertionError(f"[{dataset_name}][{mode}] Empty non-test or test split.")

    val_size_within_non_test = VAL_RATIO / (TRAIN_RATIO + VAL_RATIO)  # 0.125
    train_df, val_df = train_test_split(
        non_test_df,
        test_size=val_size_within_non_test,
        random_state=seed,
        stratify=non_test_df["Y"],
    )

    # Cold check: test vs train/val
    check_cold_overlap(train_df, test_df, key, dataset_name, f"{mode} (train/test)")
    check_cold_overlap(val_df, test_df, key, dataset_name, f"{mode} (val/test)")

    assert_two_classes(train_df, f"{mode}/train", dataset_name)
    assert_two_classes(val_df, f"{mode}/val", dataset_name)
    assert_two_classes(test_df, f"{mode}/test", dataset_name)

    total_pairs = len(df)
    def ratio(x): return x / max(1, total_pairs)
    print(f"[{dataset_name}][{mode}] test entities: {len(test_entities)}/{n}")
    print(f"[{dataset_name}][{mode}] actual pairs ratio train/val/test = "
          f"{ratio(len(train_df)):.3f}/{ratio(len(val_df)):.3f}/{ratio(len(test_df)):.3f}")

    return train_df, val_df, test_df


def split_cold_strict(df: pd.DataFrame, dataset_name: str, mode: str, seed: int = None):
    """
    Test-only cold split:

    - key entity (SMILES or Protein) is cold ONLY on test set.
    - Therefore:
        train vs test: no overlap on key
        val   vs test: no overlap on key
        train vs val: overlap is allowed
    - train/val is split pair-level on the non-test subset (with stratify).
    """
    if seed is None:
        seed = RANDOM_SEED
    key = "SMILES" if mode == "cold_drug" else "Protein"
    df = df.copy()
    df[key] = df[key].astype(str)

    total_pairs = len(df)
    stats = _entity_group_stats(df, key)
    if len(stats) < 2:
        raise AssertionError(f"[{dataset_name}][{mode}] Not enough unique entities to split into cold-test and non-test sets.")

    rng = np.random.RandomState(seed)
    stats = stats.copy()
    perm = rng.permutation(len(stats))
    stats = stats.iloc[perm].reset_index(drop=True)
    stats = stats.sort_values("n_pairs", ascending=False).reset_index(drop=True)

    target_test_pairs = int(round(TEST_RATIO * total_pairs))
    test_entities = []
    non_test_entities = []
    test_pairs = 0

    for _, row in stats.iterrows():
        ent = row[key]
        n_pairs = int(row["n_pairs"])

        # Greedy assignment to approach test ratio using entity-level blocks.
        cost_keep = abs(test_pairs - target_test_pairs)
        cost_take = abs((test_pairs + n_pairs) - target_test_pairs)
        if cost_take <= cost_keep:
            test_entities.append(ent)
            test_pairs += n_pairs
        else:
            non_test_entities.append(ent)

    # Safety: ensure both sides are non-empty.
    if len(test_entities) == 0 and len(non_test_entities) > 1:
        moved = non_test_entities.pop()
        test_entities.append(moved)
    if len(non_test_entities) == 0 and len(test_entities) > 1:
        moved = test_entities.pop()
        non_test_entities.append(moved)
    if len(test_entities) == 0 or len(non_test_entities) == 0:
        raise AssertionError(f"[{dataset_name}][{mode}] Failed to build non-empty cold test split.")

    non_test_df = df[df[key].isin(non_test_entities)].copy()
    test_df = df[df[key].isin(test_entities)].copy()

    if len(non_test_df) == 0 or len(test_df) == 0:
        raise AssertionError(f"[{dataset_name}][{mode}] Empty non-test/test split occurred.")

    val_size_within_non_test = VAL_RATIO / (TRAIN_RATIO + VAL_RATIO)  # 0.125
    train_df, val_df = train_test_split(
        non_test_df,
        test_size=val_size_within_non_test,
        random_state=seed,
        stratify=non_test_df["Y"],
    )

    # Cold checks (test-only)
    check_cold_overlap(train_df, test_df, key, dataset_name, f"{mode} (train/test)")
    check_cold_overlap(val_df, test_df, key, dataset_name, f"{mode} (val/test)")

    # Require both classes
    assert_two_classes(train_df, f"{mode}/train", dataset_name)
    assert_two_classes(val_df, f"{mode}/val", dataset_name)
    assert_two_classes(test_df, f"{mode}/test", dataset_name)

    # Basic sanity on empties
    if len(train_df) == 0 or len(val_df) == 0 or len(test_df) == 0:
        raise AssertionError(f"[{dataset_name}][{mode}] Empty split occurred (train/val/test).")

    # Optional: print ratios for transparency.
    def ratio(x): return x / max(1, total_pairs)
    train_r, val_r, test_r = ratio(len(train_df)), ratio(len(val_df)), ratio(len(test_df))
    # test ratio is approximate due to entity-level assignment.
    print(f"[{dataset_name}][{mode}] target test ratio = {TEST_RATIO:.3f}")
    print(f"[{dataset_name}][{mode}] actual pairs ratio train/val/test = "
          f"{train_r:.3f}/{val_r:.3f}/{test_r:.3f}")

    return train_df, val_df, test_df


# =========================
# Main
# =========================
def run_one_dataset(dataset_name: str, csv_path: str):
    df = read_and_validate(csv_path, dataset_name)
    out_base = os.path.join(BASE_DIR, dataset_name)

    # random
    print(f"\n[{dataset_name}] === random split (pair-level 7/1/2, stratify) ===")
    train_df, val_df, test_df = split_random(df, dataset_name)
    out_dir = os.path.join(out_base, "random")
    save_split(out_dir, train_df, val_df, test_df)
    print(f"Train: {label_stats(train_df)}")
    print(f"Val:   {label_stats(val_df)}")
    print(f"Test:  {label_stats(test_df)}")
    print(f"Saved to: {out_dir}")

    # cold_drug (test-only cold, hybrid)
    print(f"\n[{dataset_name}] === cold_drug split (hybrid: entity-count test + stratified train/val) ===")
    for t in range(MAX_RESAMPLE_TRIES):
        try:
            train_df, val_df, test_df = split_cold_hybrid(df, dataset_name, mode="cold_drug", seed=RANDOM_SEED + t)
            break
        except AssertionError as e:
            if t == MAX_RESAMPLE_TRIES - 1:
                raise
            continue

    out_dir = os.path.join(out_base, "cold_drug")
    save_split(out_dir, train_df, val_df, test_df)
    print(f"Train: {label_stats(train_df)}")
    print(f"Val:   {label_stats(val_df)}")
    print(f"Test:  {label_stats(test_df)}")
    print(f"Drug overlap(train/val): {len(set(train_df['SMILES'].astype(str)) & set(val_df['SMILES'].astype(str)))} (allowed)")
    print(f"Drug overlap(train/test): {len(set(train_df['SMILES'].astype(str)) & set(test_df['SMILES'].astype(str)))} (should be 0)")
    print(f"Drug overlap(val/test): {len(set(val_df['SMILES'].astype(str)) & set(test_df['SMILES'].astype(str)))} (should be 0)")
    print(f"Saved to: {out_dir}")

    # cold_protein (test-only cold, hybrid)
    print(f"\n[{dataset_name}] === cold_protein split (hybrid: entity-count test + stratified train/val) ===")
    for t in range(MAX_RESAMPLE_TRIES):
        try:
            train_df, val_df, test_df = split_cold_hybrid(df, dataset_name, mode="cold_protein", seed=RANDOM_SEED + t)
            break
        except AssertionError as e:
            if t == MAX_RESAMPLE_TRIES - 1:
                raise
            continue

    out_dir = os.path.join(out_base, "cold_protein")
    save_split(out_dir, train_df, val_df, test_df)
    print(f"Train: {label_stats(train_df)}")
    print(f"Val:   {label_stats(val_df)}")
    print(f"Test:  {label_stats(test_df)}")
    print(f"Protein overlap(train/val): {len(set(train_df['Protein'].astype(str)) & set(val_df['Protein'].astype(str)))} (allowed)")
    print(f"Protein overlap(train/test): {len(set(train_df['Protein'].astype(str)) & set(test_df['Protein'].astype(str)))} (should be 0)")
    print(f"Protein overlap(val/test): {len(set(val_df['Protein'].astype(str)) & set(test_df['Protein'].astype(str)))} (should be 0)")
    print(f"Saved to: {out_dir}")


def main():
    if not os.path.isdir(DATASETS_DIR):
        print(f"ERROR: datasets folder not found: {DATASETS_DIR}", file=sys.stderr)
        sys.exit(1)

    for dataset_name, path in DATASET_SPECS:
        print("\n" + "=" * 80)
        print(f"Processing {dataset_name}")
        print("=" * 80)
        run_one_dataset(dataset_name, path)

    print("\n✅ Done. Output structure:")
    print("  ./BioSnap/{random|cold_drug|cold_protein}/{train|val|test}.csv")
    print("  ./DrugBank/{random|cold_drug|cold_protein}/{train|val|test}.csv")
    print("\nNOTE:")
    print("  cold_* is TEST-ONLY COLD: test is disjoint from train/val on the key entity.")
    print("  train/val overlap on the key entity is allowed.")


if __name__ == "__main__":
    main()
