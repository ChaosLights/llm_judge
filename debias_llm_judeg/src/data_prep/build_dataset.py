"""
Prerequisite: export HF_TOKEN="hf_xxx"
source ~/{.zshrc || .bashrc} # depends on your shell

build_dataset.py

Download and save raw MT-Bench (human judgments) and Chatbot Arena
conversations into PROJECT_ROOT/data/raw, keeping conversation_a and
conversation_b (full multi-turn conversations).

We restrict to turn == 1 for now (simpler, single-turn comparisons), but
you can remove that filter if you want all turns.

Outputs:
  - PROJECT_ROOT/data/raw/mt_bench_human_turn1.jsonl
  - PROJECT_ROOT/data/raw/chatbot_arena_turn1.jsonl
"""

import os
from pathlib import Path

from datasets import load_dataset


def get_project_root() -> Path:
    """
    Resolve PROJECT_ROOT as the repo root, assuming this file lives at:

        PROJECT_ROOT/src/data_prep/build_dataset.py
    """
    this_file = Path(__file__).resolve()
    # Go up two levels: src/data_prep -> src -> PROJECT_ROOT
    project_root = this_file.parent.parent.parent
    return project_root


def save_mt_bench_raw(raw_dir: Path) -> None:
    print("[INFO] Loading MT-Bench human judgments from HF...")
    mt = load_dataset("lmsys/mt_bench_human_judgments")

    # Use only the human split
    mt_human = mt["human"]

    # (Optional but recommended) keep only turn == 1
    mt_human = mt_human.filter(lambda ex: ex["turn"] == 1)
    print(f"[INFO] MT-Bench (human, turn==1) rows: {len(mt_human)}")

    # Add a source column so we remember where it came from
    mt_human = mt_human.add_column("source", ["mt_bench"] * len(mt_human))

    out_path = raw_dir / "mt_bench_human_turn1.jsonl"
    print(f"[INFO] Saving MT-Bench raw data to {out_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mt_human.to_json(out_path.as_posix(), lines=True, orient="records")
    print("[DONE] Saved MT-Bench raw data.")


def save_arena_raw(raw_dir: Path) -> None:
    print("[INFO] Loading Chatbot Arena conversations from HF...")
    arena = load_dataset("lmsys/chatbot_arena_conversations")
    arena_train = arena["train"]

    # (Optional but recommended) keep only turn == 1
    arena_train = arena_train.filter(lambda ex: ex["turn"] == 1)
    print(f"[INFO] Chatbot Arena (train, turn==1) rows: {len(arena_train)}")

    # Add a source column
    arena_train = arena_train.add_column("source", ["arena"] * len(arena_train))

    out_path = raw_dir / "chatbot_arena_turn1.jsonl"
    print(f"[INFO] Saving Chatbot Arena raw data to {out_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    arena_train.to_json(out_path.as_posix(), lines=True, orient="records")
    print("[DONE] Saved Chatbot Arena raw data.")


def main() -> None:
    project_root = get_project_root()
    print(f"[INFO] PROJECT_ROOT = {project_root}")

    raw_dir = project_root / "data" / "raw"

    save_mt_bench_raw(raw_dir)
    save_arena_raw(raw_dir)

    print("[INFO] All raw datasets saved under data/raw.")


if __name__ == "__main__":
    main()