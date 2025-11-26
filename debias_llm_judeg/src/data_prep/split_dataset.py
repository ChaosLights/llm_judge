"""
split_dataset.py

Read raw MT-Bench and Chatbot Arena JSONL files from:

  PROJECT_ROOT/data/raw/mt_bench_human_turn1.jsonl
  PROJECT_ROOT/data/raw/chatbot_arena_turn1.jsonl

Then create processed splits under:

  PROJECT_ROOT/data/processed/train.jsonl
  PROJECT_ROOT/data/processed/val.jsonl
  PROJECT_ROOT/data/processed/test.jsonl

Splitting rule (for each split size N):
  - Sample ((len(mt) / (len(mt) + len(arena))) * N) rows from MT-Bench
  - Sample ((len(arena) / (len(mt) + len(arena))) * N) rows from Arena
  - Use seed=1337 for reproducibility.

We keep rows with the following columns:
  question_id, model_a, model_b, winner, judge, conversation_a,
  conversation_b, turn, source
"""

from pathlib import Path

from datasets import load_dataset, Dataset, concatenate_datasets, Value


SEED = 1337

TRAIN_N = 9000
VAL_N = 500
TEST_N = 500


def get_project_root() -> Path:
    """
    Resolve PROJECT_ROOT as the repo root, assuming this file lives at:

        PROJECT_ROOT/src/data_prep/split_dataset.py
    """
    this_file = Path(__file__).resolve()
    return this_file.parent.parent.parent


def load_raw_datasets(raw_dir: Path) -> tuple[Dataset, Dataset]:
    """
    Load raw MT-Bench and Arena datasets from JSONL.
    """
    mt_path = raw_dir / "mt_bench_human_turn1.jsonl"
    arena_path = raw_dir / "chatbot_arena_turn1.jsonl"

    if not mt_path.exists():
        raise FileNotFoundError(f"MT-Bench raw file not found: {mt_path}")
    if not arena_path.exists():
        raise FileNotFoundError(f"Chatbot Arena raw file not found: {arena_path}")

    print(f"[INFO] Loading MT-Bench from {mt_path}")
    mt = load_dataset("json", data_files=str(mt_path), split="train")

    print(f"[INFO] Loading Chatbot Arena from {arena_path}")
    arena = load_dataset("json", data_files=str(arena_path), split="train")

    print(f"[INFO] MT-Bench rows: {len(mt)}")
    print(f"[INFO] Chatbot Arena rows: {len(arena)}")

    return mt, arena


def compute_counts(len_mt: int, len_arena: int, N: int) -> tuple[int, int]:
    """
    Compute how many samples to draw from MT-Bench and Arena for a split of size N.
    """
    total = len_mt + len_arena
    if total == 0:
        raise ValueError("Total number of rows is zero; cannot split.")

    # proportional count for MT-Bench
    n_mt = round(len_mt / total * N)
    # ensure total is exactly N
    n_arena = N - n_mt

    return n_mt, n_arena


def main() -> None:
    project_root = get_project_root()
    print(f"[INFO] PROJECT_ROOT = {project_root}")

    raw_dir = project_root / "data" / "raw"
    processed_dir = project_root / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    mt_raw, arena_raw = load_raw_datasets(raw_dir)

    # 🔹 Keep only the common / needed columns
    keep_cols = [
        "question_id",
        "model_a",
        "model_b",
        "winner",
        "judge",
        "conversation_a",
        "conversation_b",
        "turn",
        "source",
    ]
    mt_raw = mt_raw.select_columns(keep_cols)
    arena_raw = arena_raw.select_columns(keep_cols)

    # 🔹 Make question_id the same type (string) in both
    mt_raw = mt_raw.cast_column("question_id", Value("string"))
    arena_raw = arena_raw.cast_column("question_id", Value("string"))

    len_mt = len(mt_raw)
    len_arena = len(arena_raw)

    # Shuffle once per dataset to ensure reproducible, disjoint splits
    mt_shuffled = mt_raw.shuffle(seed=SEED)
    arena_shuffled = arena_raw.shuffle(seed=SEED)

    # Compute counts for each split
    mt_train_n, arena_train_n = compute_counts(len_mt, len_arena, TRAIN_N)
    mt_val_n, arena_val_n = compute_counts(len_mt, len_arena, VAL_N)
    mt_test_n, arena_test_n = compute_counts(len_mt, len_arena, TEST_N)

    print("[INFO] Split counts:")
    print(f"  Train: MT-Bench={mt_train_n}, Arena={arena_train_n} (total={TRAIN_N})")
    print(f"  Val:   MT-Bench={mt_val_n}, Arena={arena_val_n} (total={VAL_N})")
    print(f"  Test:  MT-Bench={mt_test_n}, Arena={arena_test_n} (total={TEST_N})")

    # Sanity check: ensure we don't ask for more rows than available
    if mt_train_n + mt_val_n + mt_test_n > len_mt:
        raise ValueError(
            f"Requested MT-Bench rows (train+val+test={mt_train_n + mt_val_n + mt_test_n}) "
            f"exceed available MT-Bench rows ({len_mt})."
        )
    if arena_train_n + arena_val_n + arena_test_n > len_arena:
        raise ValueError(
            f"Requested Arena rows (train+val+test={arena_train_n + arena_val_n + arena_test_n}) "
            f"exceed available Arena rows ({len_arena})."
        )

    # Slice MT-Bench splits 
    mt_train = mt_shuffled.select(range(0, mt_train_n))
    mt_val = mt_shuffled.select(range(mt_train_n, mt_train_n + mt_val_n))
    mt_test = mt_shuffled.select(
        range(mt_train_n + mt_val_n, mt_train_n + mt_val_n + mt_test_n)
    )

    # Slice Arena splits
    arena_train = arena_shuffled.select(range(0, arena_train_n))
    arena_val = arena_shuffled.select(range(arena_train_n, arena_train_n + arena_val_n))
    arena_test = arena_shuffled.select(
        range(arena_train_n + arena_val_n, arena_train_n + arena_val_n + arena_test_n)
    )

    # Combine MT-Bench + Arena for each split
    train_ds = concatenate_datasets([mt_train, arena_train])
    val_ds = concatenate_datasets([mt_val, arena_val])
    test_ds = concatenate_datasets([mt_test, arena_test])

    # Shuffle each combined split again (with same seed for determinism)
    train_ds = train_ds.shuffle(seed=SEED)
    val_ds = val_ds.shuffle(seed=SEED)
    test_ds = test_ds.shuffle(seed=SEED)

    # Save to JSONL
    train_path = processed_dir / "train.jsonl"
    val_path = processed_dir / "val.jsonl"
    test_path = processed_dir / "test.jsonl"

    print(f"[INFO] Saving train split to {train_path}")
    train_ds.to_json(train_path.as_posix(), lines=True, orient="records")

    print(f"[INFO] Saving val split to {val_path}")
    val_ds.to_json(val_path.as_posix(), lines=True, orient="records")

    print(f"[INFO] Saving test split to {test_path}")
    test_ds.to_json(test_path.as_posix(), lines=True, orient="records")

    print("[DONE] All processed splits saved under data/processed/")


if __name__ == "__main__":
    main()
