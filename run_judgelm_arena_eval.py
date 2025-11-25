import argparse
import json
from typing import Dict, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm


def extract_qa_from_sample(sample: Dict) -> Tuple[str, str, str]:
    """
    从一条 arena 样本里抽出：
    - question 文本
    - answer_a：conversation_a 里的最后一个 assistant 回复
    - answer_b：conversation_b 里的最后一个 assistant 回复
    """
    # 问题：两个对话的第一个 user，一般是相同的
    question = sample["conversation_a"][0]["content"]

    def last_assistant(conv):
        for msg in reversed(conv):
            if msg.get("role") == "assistant":
                return msg["content"]
        # 兜底
        return conv[-1]["content"]

    ans_a = last_assistant(sample["conversation_a"])
    ans_b = last_assistant(sample["conversation_b"])
    return question, ans_a, ans_b


JUDGE_PROMPT_TEMPLATE = """You are an impartial and strict judge.
You will be given a user question and two candidate answers (A and B).
Your task is to decide which answer is better overall.

Criteria include helpfulness, correctness, depth, safety, and following instructions.

Read the question and both answers carefully, then output exactly ONE letter:
- "A" if Answer A is better
- "B" if Answer B is better
- "C" if they are roughly tied

Do NOT explain your choice.

[Question]
{question}

[Answer A]
{answer_a}

[Answer B]
{answer_b}

Your decision (A, B, or C):
"""


def build_prompt(question: str, ans_a: str, ans_b: str) -> str:
    return JUDGE_PROMPT_TEMPLATE.format(
        question=question,
        answer_a=ans_a,
        answer_b=ans_b,
    )


def parse_choice(text: str) -> str:
    """
    从模型输出里解析第一个出现的 A/B/C/1/2/3/T 之类的字母/数字，
    统一成 "A" / "B" / "C"（C 表示 tie）。
    """
    for ch in text.strip():
        up = ch.upper()
        if up in ["A", "B", "C"]:
            return up
        if up in ["1"]:
            return "A"
        if up in ["2"]:
            return "B"
        if up in ["3", "T"]:
            return "C"
    # 实在看不出来，就当 tie
    return "C"


@torch.inference_mode()
def judge_pair(model, tokenizer, device, question: str, ans_a: str, ans_b: str) -> str:
    prompt = build_prompt(question, ans_a, ans_b)
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=2048,
    ).to(device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=4,
        do_sample=False,      # 贪心解码，稳定一点
        temperature=0.0,
    )

    new_tokens = outputs[0, inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return parse_choice(text)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True,
                        help="e.g. BAAI/JudgeLM-7B-v1.0 or local ckpt path")
    parser.add_argument("--data-path", type=str, required=True,
                        help="你的 train/dev/test jsonl 路径")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="调试时可以只跑前 N 条")
    args = parser.parse_args()

    print(f"Loading model from {args.model_path} ...")
    # 1. 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False)

    # 2. 决定用什么设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    # 3. 加载模型（不再用 device_map="auto"，也不 offload）
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )

    # 4. 手动把模型移到对应设备
    model.to(device)

    print("Model loaded on", device)

    # 统计指标
    n_total = 0

    # 顺序：A = model_a, B = model_b
    n_non_tie_seq = 0
    n_correct_seq = 0
    n_left_seq = 0
    n_tie_seq = 0

    # 反转：A = model_b, B = model_a
    n_non_tie_rev = 0
    n_correct_rev = 0
    n_left_rev = 0
    n_tie_rev = 0

    print(f"Evaluating on {args.data_path} ...")
    with open(args.data_path, "r", encoding="utf-8") as f:
        for line in tqdm(f):
            line = line.strip()
            if not line:
                continue
            sample = json.loads(line)

            n_total += 1
            if args.max_samples is not None and n_total > args.max_samples:
                break

            question, ans_a, ans_b = extract_qa_from_sample(sample)
            winner = sample.get("winner", None)  # "model_a" / "model_b" / 也可能有 "tie"

            # 1) 顺序版：A = model_a, B = model_b
            pred_seq = judge_pair(model, tokenizer, device, question, ans_a, ans_b)

            # 2) 反转版：A = model_b, B = model_a
            pred_rev = judge_pair(model, tokenizer, device, question, ans_b, ans_a)

            # 把 human winner 映射到 A/B/C
            if winner == "model_a":
                human_seq = "A"
                human_rev = "B"
            elif winner == "model_b":
                human_seq = "B"
                human_rev = "A"
            else:  # 其他情况当作平局
                human_seq = "C"
                human_rev = "C"

            # 顺序版统计
            if pred_seq == "C":
                n_tie_seq += 1
            else:
                n_non_tie_seq += 1
                if pred_seq == "A":
                    n_left_seq += 1
                if winner in ["model_a", "model_b"] and pred_seq == human_seq:
                    n_correct_seq += 1

            # 反转版统计
            if pred_rev == "C":
                n_tie_rev += 1
            else:
                n_non_tie_rev += 1
                if pred_rev == "A":  # 注意，反转后 A 是原来的 model_b
                    n_left_rev += 1
                if winner in ["model_a", "model_b"] and pred_rev == human_rev:
                    n_correct_rev += 1

    print("==== Results ====")
    print(f"#total samples: {n_total}")

    if n_non_tie_seq > 0:
        pah_seq = n_correct_seq / n_non_tie_seq
        left_rate_seq = n_left_seq / n_non_tie_seq
    else:
        pah_seq = 0.0
        left_rate_seq = 0.0

    if n_non_tie_rev > 0:
        pah_rev = n_correct_rev / n_non_tie_rev
        left_rate_rev = n_left_rev / n_non_tie_rev
    else:
        pah_rev = 0.0
        left_rate_rev = 0.0

    pbi = left_rate_seq - left_rate_rev

    print(f"Sequential (A = model_a, B = model_b):")
    print(f"  Non-tie cases: {n_non_tie_seq}")
    print(f"  PAH_seq (accuracy vs human, no ties): {pah_seq:.4f}")
    print(f"  Left-choice rate: {left_rate_seq:.4f}")
    print(f"  Tie rate: {n_tie_seq / max(n_total,1):.4f}")

    print(f"Reversed (A = model_b, B = model_a):")
    print(f"  Non-tie cases: {n_non_tie_rev}")
    print(f"  PAH_rev (accuracy vs human, no ties): {pah_rev:.4f}")
    print(f"  Left-choice rate: {left_rate_rev:.4f}")
    print(f"  Tie rate: {n_tie_rev / max(n_total,1):.4f}")

    print(f"Position Bias Index (PBI = left_rate_seq - left_rate_rev): {pbi:.4f}")


if __name__ == "__main__":
    main()

