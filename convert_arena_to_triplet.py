#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Convert LMSYS Arena-style JSONL data to a simple triplet format:

Input (per line, Arena example):
{
  "question_id": "...",
  "model_a": "...",
  "model_b": "...",
  "winner": "model_a" | "model_b" | "tie" | ...,
  "conversation_a": [{"role": "user", "content": ...}, {"role": "assistant", "content": ...}],
  "conversation_b": [...]
}

Output (per line):
{
  "id": "question_id",
  "question": "...",
  "answer_a": "...",  # final assistant message from conversation_a
  "answer_b": "...",  # final assistant message from conversation_b
  "human_preference": "A" | "B" | "TIE"
}
"""

import json
import argparse
from typing import Dict, Any


def arena_to_triplet(sample: Dict[str, Any]) -> Dict[str, Any]:
    conv_a = sample.get("conversation_a", [])
    conv_b = sample.get("conversation_b", [])

    # Take the first user message as the question (a/b should be identical)
    user_a = [m.get("content", "") for m in conv_a if m.get("role") == "user"]
    user_b = [m.get("content", "") for m in conv_b if m.get("role") == "user"]
    if user_a:
        question = user_a[0]
    elif user_b:
        question = user_b[0]
    else:
        question = ""

    # Take the last assistant message in each conversation as the candidate answer
    ans_a_list = [m.get("content", "") for m in conv_a if m.get("role") == "assistant"]
    ans_b_list = [m.get("content", "") for m in conv_b if m.get("role") == "assistant"]
    answer_a = ans_a_list[-1] if ans_a_list else ""
    answer_b = ans_b_list[-1] if ans_b_list else ""

    # Map Arena winner to content-level preference
    winner_raw = str(sample.get("winner", "")).lower()
    if winner_raw == "model_a":
        human_pref = "A"
    elif winner_raw == "model_b":
        human_pref = "B"
    else:
        human_pref = "TIE"

    out = {
        "id": sample.get("question_id") or sample.get("id"),
        "question": question,
        "answer_a": answer_a,
        "answer_b": answer_b,
        "human_preference": human_pref,
    }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert Arena JSONL to triplet JSONL.")
    parser.add_argument("--input", required=True, help="Input Arena-style JSONL path")
    parser.add_argument("--output", required=True, help="Output triplet JSONL path")
    args = parser.parse_args()

    n_in = 0
    n_out = 0
    with open(args.input, "r", encoding="utf-8") as fin, \
         open(args.output, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            n_in += 1
            sample = json.loads(line)
            triplet = arena_to_triplet(sample)
            fout.write(json.dumps(triplet, ensure_ascii=False) + "\n")
            n_out += 1

    print(f"Converted {n_in} → {n_out} samples to triplet format.")
    print(f"Saved to: {args.output}")


if __name__ == "__main__":
    main()
