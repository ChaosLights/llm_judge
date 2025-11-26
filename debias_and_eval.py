#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
End-to-end Align-Trace debias pipeline for JudgeLM.

Steps:
 1) Load train/val/test triplet JSONL datasets (converted from LMSYS Arena).
 2) Run Align-Trace on a subset of the train set to find position-bias-related layers.
 3) Learn a low-rank bias subspace (via PCA) for those layers.
 4) On the val set, grid-search steering strength alpha to trade off PAH vs position bias.
 5) On the test set, evaluate baseline and debiased JudgeLM and save metrics.

This script assumes your JudgeLM is a HuggingFace AutoModelForCausalLM-compatible model
that outputs "A"/"B"/"TIE" by emitting one of these tokens as the final token.
"""

import argparse
import json
import math
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import torch
from torch import nn
from sklearn.decomposition import PCA
from transformers import AutoModelForCausalLM, AutoTokenizer


# -----------------------------
# Utils
# -----------------------------


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_triplet_data(path: str) -> List[Dict[str, Any]]:
    data: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data


def build_judge_prompt(question: str, left_answer: str, right_answer: str) -> str:
    """Template for JudgeLM prompt where 'left_answer' is Answer 1 and 'right_answer' is Answer 2."""
    return (
        "You are an impartial judge. You will read a user question and two candidate answers.\n\n"
        "[Question]\n"
        f"{question}\n\n"
        "[Answer 1]\n"
        f"{left_answer}\n\n"
        "[Answer 2]\n"
        f"{right_answer}\n\n"
        'Your job is to decide which answer better follows the instructions, is more helpful, '
        'and is safer.\n\n'
        'Respond with a single token only: "A", "B", or "TIE".'
    )


def decode_ab_tie_from_logits(logits: torch.Tensor, tokenizer) -> str:
    """
    Decode the final token as 'A', 'B', or 'TIE' by argmax over those three token logits.
    logits: (1, seq_len, vocab)
    """
    last = logits[0, -1, :]  # (vocab,)
    id_A = tokenizer("A", add_special_tokens=False).input_ids[0]
    id_B = tokenizer("B", add_special_tokens=False).input_ids[0]
    id_T = tokenizer("TIE", add_special_tokens=False).input_ids[0]

    vals = {
        "A": last[id_A].item(),
        "B": last[id_B].item(),
        "TIE": last[id_T].item(),
    }
    return max(vals, key=vals.get)


def map_lr_to_content(pred_lr: str, order: str) -> str:
    """
    Map 'left/right' prediction to content-level A/B.
    order = "seq" means (A,B) => left=A, right=B
    order = "rev" means (B,A) => left=B, right=A
    """
    if pred_lr == "TIE":
        return "TIE"
    if order == "seq":
        return "A" if pred_lr == "A" else "B"
    elif order == "rev":
        return "B" if pred_lr == "A" else "A"
    else:
        raise ValueError(f"Unknown order: {order}")


def pref_logit(logits: torch.Tensor, tokenizer) -> torch.Tensor:
    """
    Preference logit y = logit_A - logit_B at the final token.
    Returns a scalar tensor.
    """
    last = logits[0, -1, :]
    id_A = tokenizer("A", add_special_tokens=False).input_ids[0]
    id_B = tokenizer("B", add_special_tokens=False).input_ids[0]
    return last[id_A] - last[id_B]


def get_block_list(model: nn.Module) -> List[nn.Module]:
    """
    Try to obtain the list of Transformer blocks to attach forward hooks.
    Supports common architectures; you may need to adjust this for your JudgeLM.
    """
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return list(model.model.layers)
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return list(model.transformer.h)
    raise ValueError(
        "Unsupported model architecture. Please edit get_block_list(model) "
        "to return a list of Transformer blocks for your JudgeLM."
    )


# -----------------------------
# Align-Trace: collect layer deltas and influence scores
# -----------------------------


def collect_layer_deltas(
    model: nn.Module,
    tokenizer,
    data: List[Dict[str, Any]],
    num_traces: int,
    device: str,
) -> Tuple[List[List[torch.Tensor]], List[List[float]]]:
    """
    For a subset of training examples, run JudgeLM on both (A,B) and (B,A) orders,
    identify neutral vs biased traces (using human preference), and collect
    layer-wise activation differences and influence scores.

    Returns:
      layer_deltas: list[num_layers] of list[Tensor(d,)]
      layer_scores: list[num_layers] of list[float]
    """
    model.eval()
    indices = list(range(len(data)))
    random.shuffle(indices)

    collected = 0
    layer_deltas: Optional[List[List[torch.Tensor]]] = None
    layer_scores: Optional[List[List[float]]] = None

    for idx in indices:
        if collected >= num_traces:
            break

        ex = data[idx]
        human = ex.get("human_preference", "TIE")
        if human not in ("A", "B"):
            continue

        q = ex.get("question", "")
        a = ex.get("answer_a", "")
        b = ex.get("answer_b", "")

        prompt_seq = build_judge_prompt(q, a, b)  # (A,B): order="seq"
        prompt_rev = build_judge_prompt(q, b, a)  # (B,A): order="rev"

        # 1) Quick no-grad pass just to get predictions for both orders
        with torch.no_grad():
            inputs_seq = tokenizer(prompt_seq, return_tensors="pt").to(device)
            out_seq = model(**inputs_seq)
            pred_seq_lr = decode_ab_tie_from_logits(out_seq.logits, tokenizer)

            inputs_rev = tokenizer(prompt_rev, return_tensors="pt").to(device)
            out_rev = model(**inputs_rev)
            pred_rev_lr = decode_ab_tie_from_logits(out_rev.logits, tokenizer)

        pred_seq_content = map_lr_to_content(pred_seq_lr, order="seq")
        pred_rev_content = map_lr_to_content(pred_rev_lr, order="rev")

        # Determine neutral vs biased:
        neutral_prompt: Optional[str] = None
        biased_prompt: Optional[str] = None
        if pred_seq_content == human and pred_rev_content != human:
            neutral_prompt = prompt_seq
            biased_prompt = prompt_rev
        elif pred_rev_content == human and pred_seq_content != human:
            neutral_prompt = prompt_rev
            biased_prompt = prompt_seq
        else:
            # Either both correct, both wrong, or ties; skip for Align-Trace
            continue

        # 2) Run neutral order WITH grad, collecting hidden states and gradients
        inputs_neu = tokenizer(neutral_prompt, return_tensors="pt").to(device)
        outputs_neu = model(
            **inputs_neu,
            output_hidden_states=True,
            use_cache=False,
        )
        hs_neu = list(outputs_neu.hidden_states)
        logits_neu = outputs_neu.logits

        # 3) Run biased order WITHOUT grad for hidden states only
        with torch.no_grad():
            inputs_bia = tokenizer(biased_prompt, return_tensors="pt").to(device)
            outputs_bia = model(
                **inputs_bia,
                output_hidden_states=True,
                use_cache=False,
            )
        hs_bia = list(outputs_bia.hidden_states)

        if layer_deltas is None:
            num_layers = len(hs_neu) - 1  # exclude embedding layer 0
            layer_deltas = [[] for _ in range(num_layers)]
            layer_scores = [[] for _ in range(num_layers)]

        # Compute grads w.r.t. the last token at each layer (1..L)
        hs_tokens = [hs_neu[l][:, -1, :] for l in range(1, len(hs_neu))]
        y = pref_logit(logits_neu, tokenizer)
        grads = torch.autograd.grad(
            y,
            hs_tokens,
            retain_graph=False,
            allow_unused=True,
        )

        for l_idx in range(1, len(hs_neu)):
            # Activation difference at this layer, last token
            h_neu_vec = hs_neu[l_idx][0, -1, :].detach()
            h_bia_vec = hs_bia[l_idx][0, -1, :].detach()
            delta = (h_bia_vec - h_neu_vec).cpu()

            g_tensor = grads[l_idx - 1]
            if g_tensor is None:
                influence = 0.0
            else:
                g_vec = g_tensor[0].detach().cpu()  # (hidden_dim,)
                influence = float((g_vec * delta).sum().item())

            layer_deltas[l_idx - 1].append(delta)
            layer_scores[l_idx - 1].append(abs(influence))

        collected += 1
        if collected % 10 == 0:
            print(f"[collect] Collected {collected} traces", flush=True)

    if collected == 0 or layer_deltas is None or layer_scores is None:
        raise RuntimeError("No usable neutral/biased pairs collected; check your data or num_traces.")

    print(f"[collect] Finished with {collected} traces.")
    return layer_deltas, layer_scores


def compute_top_layers_and_bias_dirs(
    layer_deltas: List[List[torch.Tensor]],
    layer_scores: List[List[float]],
    rank_r: int,
) -> Tuple[List[int], Dict[int, torch.Tensor], List[float]]:
    """
    From collected layer deltas and influence scores, compute:
      - mean influence per layer
      - select top-k layers
      - for each selected layer, learn a PCA subspace of dimension <= rank_r

    Returns:
      top_layers: list of layer indices
      bias_dirs: dict[layer_id] -> Tensor(r, d)
      mean_scores: list of mean influence scores for all layers
    """
    num_layers = len(layer_deltas)
    mean_scores: List[float] = []
    for l in range(num_layers):
        scores = layer_scores[l]
        if not scores:
            mean_scores.append(0.0)
        else:
            mean_scores.append(float(np.mean(scores)))

    # Select top-3 layers (or fewer if the model has fewer layers)
    k = min(3, num_layers)
    sorted_idx = np.argsort(mean_scores)
    top_layers = sorted_idx[-k:].tolist()

    bias_dirs: Dict[int, torch.Tensor] = {}
    for l in top_layers:
        deltas_l = layer_deltas[l]
        if not deltas_l:
            continue
        X = torch.stack(deltas_l, dim=0)  # (N, d)
        # Ensure n_components is valid
        n_comp = min(rank_r, X.shape[0], X.shape[1])
        if n_comp < 1:
            continue
        pca = PCA(n_components=n_comp)
        pca.fit(X.numpy())
        comps = torch.tensor(pca.components_, dtype=torch.float32)  # (r, d)
        bias_dirs[int(l)] = comps

    print(f"[pca] Selected top layers: {top_layers}")
    for l in top_layers:
        print(f"[pca] Layer {l}: mean influence = {mean_scores[l]:.4f}, "
              f"subspace rank = {bias_dirs.get(int(l), torch.empty(0)).shape[0]}")
    return top_layers, bias_dirs, mean_scores


# -----------------------------
# Steering hooks & evaluation
# -----------------------------


def make_steering_hook(W: torch.Tensor, alpha: float = 1.0):
    """
    Create a forward hook that subtracts projection onto bias subspace W.
    W: (r, d)
    """
    # Normalize each direction
    W = W / (W.norm(dim=1, keepdim=True) + 1e-8)  # (r, d)

    def hook(module: nn.Module, inp, out):
        if not isinstance(out, torch.Tensor):
            return out
        h = out
        # 确保 W 和 h 在同一个 device & dtype
        W_local = W.to(device=h.device, dtype=h.dtype)

        coeff = torch.einsum("bsd,rd->bsr", h, W_local)
        proj = torch.einsum("bsr,rd->bsd", coeff, W_local)
        return h - alpha * proj

    return hook


def register_steering_hooks(
    model: nn.Module,
    bias_dirs: Dict[int, torch.Tensor],
    top_layers: List[int],
    alpha: float,
    device: str,
) -> List[Any]:
    """
    Register steering hooks for the selected layers and return the hook handles.
    """
    blocks = get_block_list(model)
    handles: List[Any] = []
    for l in top_layers:
        if int(l) >= len(blocks):
            continue
        W = bias_dirs[int(l)].to(device)
        hook = make_steering_hook(W, alpha=alpha)
        handle = blocks[int(l)].register_forward_hook(hook)
        handles.append(handle)
    return handles


def evaluate_judge(
    model: nn.Module,
    tokenizer,
    data: List[Dict[str, Any]],
    device: str,
    bias_dirs: Optional[Dict[int, torch.Tensor]] = None,
    top_layers: Optional[List[int]] = None,
    alpha: Optional[float] = None,
) -> Dict[str, float]:
    """
    Evaluate JudgeLM on a triplet dataset and compute:
      - PAH_seq / PAH_rev (accuracy vs human, ignoring tie cases)
      - left-choice rate (per non-tie case) for seq/rev
      - tie rate for seq/rev
      - PBI = left_rate_seq - left_rate_rev
    If bias_dirs/top_layers/alpha are provided, steering hooks are registered
    during evaluation (and removed afterward).
    """
    model.eval()

    handles: List[Any] = []
    if bias_dirs is not None and top_layers is not None and alpha is not None:
        handles = register_steering_hooks(model, bias_dirs, top_layers, alpha, device)

    total = len(data)
    # Sequential
    non_tie_seq = 0
    left_seq = 0
    tie_seq = 0
    correct_seq = 0
    denom_seq = 0  # for PAH (non-tie human & non-tie pred)
    # Reversed
    non_tie_rev = 0
    left_rev = 0
    tie_rev = 0
    correct_rev = 0
    denom_rev = 0

    try:
        for ex in data:
            q = ex.get("question", "")
            a = ex.get("answer_a", "")
            b = ex.get("answer_b", "")
            human = ex.get("human_preference", "TIE")

            # Sequential (A,B)
            prompt_seq = build_judge_prompt(q, a, b)
            with torch.no_grad():
                inputs_seq = tokenizer(prompt_seq, return_tensors="pt").to(device)
                out_seq = model(**inputs_seq)
                pred_seq_lr = decode_ab_tie_from_logits(out_seq.logits, tokenizer)
            pred_seq_content = map_lr_to_content(pred_seq_lr, "seq")

            if pred_seq_lr == "TIE":
                tie_seq += 1
            else:
                non_tie_seq += 1
                if pred_seq_lr == "A":
                    left_seq += 1

            if human in ("A", "B") and pred_seq_content in ("A", "B"):
                denom_seq += 1
                if pred_seq_content == human:
                    correct_seq += 1

            # Reversed (B,A)
            prompt_rev = build_judge_prompt(q, b, a)
            with torch.no_grad():
                inputs_rev = tokenizer(prompt_rev, return_tensors="pt").to(device)
                out_rev = model(**inputs_rev)
                pred_rev_lr = decode_ab_tie_from_logits(out_rev.logits, tokenizer)
            pred_rev_content = map_lr_to_content(pred_rev_lr, "rev")

            if pred_rev_lr == "TIE":
                tie_rev += 1
            else:
                non_tie_rev += 1
                if pred_rev_lr == "A":
                    left_rev += 1

            if human in ("A", "B") and pred_rev_content in ("A", "B"):
                denom_rev += 1
                if pred_rev_content == human:
                    correct_rev += 1

        # Compute metrics
        pah_seq = (correct_seq / denom_seq) if denom_seq > 0 else 0.0
        pah_rev = (correct_rev / denom_rev) if denom_rev > 0 else 0.0
        left_rate_seq = (left_seq / non_tie_seq) if non_tie_seq > 0 else 0.0
        left_rate_rev = (left_rev / non_tie_rev) if non_tie_rev > 0 else 0.0
        tie_rate_seq = (tie_seq / total) if total > 0 else 0.0
        tie_rate_rev = (tie_rev / total) if total > 0 else 0.0
        pbi = left_rate_seq - left_rate_rev
        avg_pah = 0.5 * (pah_seq + pah_rev)

        metrics = {
            "num_samples": float(total),
            "non_tie_seq": float(non_tie_seq),
            "non_tie_rev": float(non_tie_rev),
            "pah_seq": float(pah_seq),
            "pah_rev": float(pah_rev),
            "left_rate_seq": float(left_rate_seq),
            "left_rate_rev": float(left_rate_rev),
            "tie_rate_seq": float(tie_rate_seq),
            "tie_rate_rev": float(tie_rate_rev),
            "pbi": float(pbi),
            "avg_pah": float(avg_pah),
        }
        return metrics
    finally:
        # Clean up hooks if any
        for h in handles:
            h.remove()


def score_metrics(metrics: Dict[str, float], lambda_bias: float = 0.5) -> float:
    """
    Simple scalar objective to pick alpha on the val set:
      score = avg_pah - lambda_bias * total_left_bias
    where total_left_bias measures deviation of left-choice-rate from 0.5 on both orders.
    """
    avg_pah = metrics.get("avg_pah", 0.0)
    left_seq = metrics.get("left_rate_seq", 0.0)
    left_rev = metrics.get("left_rate_rev", 0.0)
    total_left_bias = abs(left_seq - 0.5) + abs(left_rev - 0.5)
    return float(avg_pah - lambda_bias * total_left_bias)


# -----------------------------
# Main pipeline
# -----------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Align-Trace debias pipeline for JudgeLM.")
    parser.add_argument("--model_name", required=True, help="HuggingFace model name or path for JudgeLM")
    parser.add_argument("--train", required=True, help="Train triplet JSONL path")
    parser.add_argument("--val", required=True, help="Validation triplet JSONL path")
    parser.add_argument("--test", required=True, help="Test triplet JSONL path")
    parser.add_argument("--output_dir", default="align_trace_output", help="Directory to save configs and metrics")
    parser.add_argument("--num_traces", type=int, default=1000, help="Number of Align-Trace samples from train")
    parser.add_argument("--rank_r", type=int, default=2, help="Rank of PCA bias subspace per layer")
    parser.add_argument(
        "--alphas",
        type=float,
        nargs="+",
        default=[0.5, 1.0, 2.0],
        help="Candidate steering strengths to search on val",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None, help="Device (e.g. cuda, cpu). Default: auto-detect.")
    args = parser.parse_args()

    set_seed(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Device
    if args.device is not None:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[device] Using device: {device}")

    # Load data
    print("[data] Loading triplet datasets...")
    train_data = load_triplet_data(args.train)
    val_data = load_triplet_data(args.val)
    test_data = load_triplet_data(args.test)
    print(f"[data] Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

    # Load JudgeLM
    print(f"[model] Loading JudgeLM from: {args.model_name}")
    if device.startswith("cuda"):
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype=torch.float16,
            device_map="auto",
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model_name)
        model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    # Ensure EOS token exists; some judge models may need this
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    # 1) Collect layer deltas and influence scores on a subset of train
    print("[align-trace] Collecting layer deltas and influence scores...")
    layer_deltas, layer_scores = collect_layer_deltas(
        model=model,
        tokenizer=tokenizer,
        data=train_data,
        num_traces=min(args.num_traces, len(train_data)),
        device=device,
    )

    # 2) Compute top layers & bias subspace
    print("[pca] Computing bias subspace via PCA...")
    top_layers, bias_dirs, mean_scores = compute_top_layers_and_bias_dirs(
        layer_deltas=layer_deltas,
        layer_scores=layer_scores,
        rank_r=args.rank_r,
    )

    # 3) Evaluate on val to choose alpha
    print("[val] Evaluating baseline on validation set...")
    val_baseline = evaluate_judge(
        model=model,
        tokenizer=tokenizer,
        data=val_data,
        device=device,
        bias_dirs=None,
        top_layers=None,
        alpha=None,
    )
    print("[val] Baseline metrics:", json.dumps(val_baseline, indent=2))

    best_alpha: Optional[float] = None
    best_score: Optional[float] = None
    val_metrics_all: Dict[str, Dict[str, float]] = {"baseline": val_baseline}

    print("[val] Searching over alphas:", args.alphas)
    for alpha in args.alphas:
        metrics_alpha = evaluate_judge(
            model=model,
            tokenizer=tokenizer,
            data=val_data,
            device=device,
            bias_dirs=bias_dirs,
            top_layers=top_layers,
            alpha=alpha,
        )
        val_metrics_all[f"alpha_{alpha}"] = metrics_alpha
        s = score_metrics(metrics_alpha)
        print(f"[val] alpha={alpha}: score={s:.4f}, metrics={metrics_alpha}")
        if best_score is None or s > best_score:
            best_score = s
            best_alpha = alpha

    print(f"[val] Best alpha = {best_alpha} (score={best_score:.4f})")

    # Save bias config
    bias_config_path = out_dir / "bias_config.pt"
    torch.save(
        {
            "top_layers": top_layers,
            "bias_dirs": {int(k): v.cpu() for k, v in bias_dirs.items()},
            "rank_r": args.rank_r,
            "alphas": args.alphas,
            "best_alpha": best_alpha,
            "mean_scores": mean_scores,
            "model_name_or_path": args.model_name,
        },
        bias_config_path,
    )
    print(f"[save] Saved bias config to: {bias_config_path}")

    # Save val metrics
    metrics_val_path = out_dir / "metrics_val.json"
    with open(metrics_val_path, "w", encoding="utf-8") as f:
        json.dump(val_metrics_all, f, indent=2)
    print(f"[save] Saved validation metrics to: {metrics_val_path}")

    # 4) Evaluate on test set (baseline vs debiased)
    print("[test] Evaluating baseline on test set...")
    test_baseline = evaluate_judge(
        model=model,
        tokenizer=tokenizer,
        data=test_data,
        device=device,
        bias_dirs=None,
        top_layers=None,
        alpha=None,
    )
    print("[test] Baseline metrics:", json.dumps(test_baseline, indent=2))

    print("[test] Evaluating debiased JudgeLM on test set...")
    if best_alpha is None:
        raise RuntimeError("best_alpha is None; something went wrong in validation search.")

    test_debiased = evaluate_judge(
        model=model,
        tokenizer=tokenizer,
        data=test_data,
        device=device,
        bias_dirs=bias_dirs,
        top_layers=top_layers,
        alpha=float(best_alpha),
    )
    print("[test] Debiased metrics:", json.dumps(test_debiased, indent=2))

    metrics_test = {
        "baseline": test_baseline,
        "debiased": test_debiased,
        "best_alpha": best_alpha,
    }
    metrics_test_path = out_dir / "metrics_test.json"
    with open(metrics_test_path, "w", encoding="utf-8") as f:
        json.dump(metrics_test, f, indent=2)
    print(f"[save] Saved test metrics to: {metrics_test_path}")


if __name__ == "__main__":
    main()
