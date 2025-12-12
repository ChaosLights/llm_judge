#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Align-Trace debias pipeline for JudgeLM, with evaluation matching the proposal-style judge:

- For each triplet (question, answer_a, answer_b), we run the judge twice:
    1) Sequential:  Answer 1 = A (answer_a), Answer 2 = B (answer_b)
    2) Reversed:    Answer 1 = B (answer_b), Answer 2 = A (answer_a)

- The judge is prompted to output exactly one token: "A", "B", or "TIE".
- We then compute:
    * PAH_seq / PAH_rev: accuracy vs. human preference under seq/rev order
    * left-choice rate under seq/rev order
    * tie rate under seq/rev order
    * PBI = left_rate_seq - left_rate_rev
    * avg_pah = (PAH_seq + PAH_rev) / 2

Align-Trace itself (finding biased layers and bias subspace) still uses a
logit-based preference score internally, but evaluation is generation-based.

This version additionally:
  - Adds an optional supervised finetuning step (SFT) on comparison data,
    training the judge to predict both discrete winners and scalar preferences.
  - Computes Kendall's tau correlation between baseline and debiased
    preference score lists on the test set.
"""

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
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
        "You are an impartial and strict judge.\n"
        "You will be given a user question and two candidate answers (Answer 1 and Answer 2).\n"
        "Your task is to decide which answer is better overall.\n\n"
        "Criteria include helpfulness, correctness, depth, safety, and following instructions.\n\n"
        "Read the question and both answers carefully, then output exactly ONE token:\n"
        '- "A" if Answer 1 (left) is better\n'
        '- "B" if Answer 2 (right) is better\n'
        '- "TIE" if they are roughly tied\n\n"
        "Do NOT explain your choice.\n\n"
        "[Question]\n"
        f"{question}\n\n"
        "[Answer 1]\n"
        f"{left_answer}\n\n"
        "[Answer 2]\n"
        f"{right_answer}\n\n"
        "Your decision (A, B, or TIE):"
    )


def decode_ab_tie_from_logits(logits: torch.Tensor, tokenizer) -> str:
    """
    Decode the final token as 'A', 'B', or 'TIE' by argmax over those three token logits.
    logits: (1, seq_len, vocab)
    """
    last = logits[0, -1, :]  # (vocab,)
    id_A = tokenizer("A", add_special_tokens=False).input_ids[0]
    id_B = tokenizer("B", add_special_tokens=False).input_ids[0]

    # For "TIE", we take the first token id produced by the tokenizer;
    # if "TIE" is split into multiple tokens, this is still fine because
    # the model is explicitly instructed to output the single token "TIE"
    # and we only compare relative scores among these options.
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

    Here pred_lr is one of {"A", "B", "TIE"}, where "A" means "choose left",
    "B" means "choose right", and "TIE" means tie.
    """
    if pred_lr == "TIE":
        return "TIE"

    if order == "seq":
        # (A,B): left=A, right=B
        return "A" if pred_lr == "A" else "B"
    elif order == "rev":
        # (B,A): left=B, right=A
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


def kendall_tau(xs: List[float], ys: List[float]) -> float:
    """
    Compute Kendall's tau-a correlation between two score lists.
    We ignore pairs that are tied in at least one list.
    """
    n = len(xs)
    if n != len(ys) or n < 2:
        return 0.0

    concordant = 0
    discordant = 0

    for i in range(n):
        for j in range(i + 1, n):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]

            # treat very small differences as ties
            if abs(dx) < 1e-9 or abs(dy) < 1e-9:
                continue

            prod = dx * dy
            if prod > 0:
                concordant += 1
            elif prod < 0:
                discordant += 1

    denom = concordant + discordant
    if denom == 0:
        return 0.0
    return float((concordant - discordant) / denom)


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
# Supervised judge finetuning (SFT on comparison data)
# -----------------------------

LABEL2ID = {"A": 0, "B": 1, "TIE": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


class JudgePreferenceDataset(Dataset):
    """
    Turn triplet comparison data into supervised training examples.

    For each record (question, answer_a, answer_b, human_preference),
    we create two training examples:
      1) Sequential order: Answer 1 = A, Answer 2 = B
      2) Reversed order:   Answer 1 = B, Answer 2 = A

    The label is always defined with respect to the *left* (Answer 1) vs
    *right* (Answer 2) answer:
      - "A": choose left
      - "B": choose right
      - "TIE": tie

    We also attach a scalar preference score for the left answer:
      +1.0  => human prefers left
      -1.0  => human prefers right
       0.0  => tie
    """

    def __init__(self, triplets: List[Dict[str, Any]]):
        self.examples: List[Dict[str, Any]] = []

        for ex in triplets:
            q = ex.get("question", "")
            a = ex.get("answer_a", "")
            b = ex.get("answer_b", "")
            human = ex.get("human_preference", "TIE")

            # Skip unknown labels
            if human not in ("A", "B", "TIE"):
                continue

            # ----- Sequential order: Answer 1 = A (left), Answer 2 = B (right) -----
            prompt_seq = build_judge_prompt(q, a, b)
            if human == "A":
                label_seq = "A"   # choose left
                score_seq = 1.0
            elif human == "B":
                label_seq = "B"   # choose right
                score_seq = -1.0
            else:
                label_seq = "TIE"
                score_seq = 0.0

            self.examples.append(
                {"prompt": prompt_seq, "label": label_seq, "score": score_seq}
            )

            # ----- Reversed order: Answer 1 = B (left), Answer 2 = A (right) -----
            prompt_rev = build_judge_prompt(q, b, a)
            if human == "A":
                # Human prefers A, which is on the RIGHT now -> choose right
                label_rev = "B"
                score_rev = -1.0
            elif human == "B":
                # Human prefers B, which is on the LEFT now -> choose left
                label_rev = "A"
                score_rev = 1.0
            else:
                label_rev = "TIE"
                score_rev = 0.0

            self.examples.append(
                {"prompt": prompt_rev, "label": label_rev, "score": score_rev}
            )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.examples[idx]


def make_collate_fn(tokenizer, max_length: int = 2048):
    """
    Collate a batch of preference examples into tensors suitable for training.
    """

    def collate(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        prompts = [b["prompt"] for b in batch]
        labels_cls = torch.tensor(
            [LABEL2ID[b["label"]] for b in batch],
            dtype=torch.long,
        )
        labels_score = torch.tensor(
            [b["score"] for b in batch],
            dtype=torch.float32,
        )

        enc = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )

        # Attach labels so the training loop can access them
        enc["labels_cls"] = labels_cls
        enc["labels_score"] = labels_score
        return enc

    return collate


def finetune_judge(
    model: nn.Module,
    tokenizer,
    train_triplets: List[Dict[str, Any]],
    val_triplets: List[Dict[str, Any]],
    device: str,
    num_epochs: int = 1,
    batch_size: int = 4,
    lr: float = 5e-6,
    lambda_reg: float = 1.0,
    max_length: int = 2048,
    output_dir: Optional[str] = None,
) -> None:
    """
    Supervised finetuning of JudgeLM on pairwise comparison data.

    The model is trained to:
      1) Predict a discrete winner among {A, B, TIE} (classification)
      2) Predict a scalar preference score for the left answer (regression)

    Implementation details:
      - We do *not* add any extra head; we reuse the base LM head.
      - For each prompt, we take the logits at the last non-padding position.
      - We select the logits corresponding to tokens "A", "B", "TIE" and
        compute a 3-way cross-entropy loss for the discrete label.
      - We define a scalar preference logit as logit("A") - logit("B") and
        regress it (MSE) to the target scalar score in {+1, -1, 0}.
      - Total loss = loss_cls + lambda_reg * loss_reg.

    After training, the same LM can be used in Align-Trace without changing
    its architecture: everything downstream still operates on the LM logits.
    """

    train_dataset = JudgePreferenceDataset(train_triplets)
    val_dataset = JudgePreferenceDataset(val_triplets)
    collate_fn = make_collate_fn(tokenizer, max_length=max_length)

    if len(train_dataset) == 0:
        print("[finetune] WARNING: empty training dataset; skipping SFT.")
        return

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # Pre-compute token ids for A / B / TIE
    id_A = tokenizer("A", add_special_tokens=False).input_ids[0]
    id_B = tokenizer("B", add_special_tokens=False).input_ids[0]
    id_T = tokenizer("TIE", add_special_tokens=False).input_ids[0]

    for epoch in range(num_epochs):
        total_loss = 0.0
        total_cls = 0.0
        total_reg = 0.0
        n_steps = 0

        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels_cls = batch["labels_cls"].to(device)
            labels_score = batch["labels_score"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            logits = outputs.logits  # (B, S, V)

            # last non-padding position per sample
            last_idx = attention_mask.sum(dim=1) - 1  # (B,)
            batch_idx = torch.arange(input_ids.size(0), device=device)
            last_logits = logits[batch_idx, last_idx, :]  # (B, V)

            # 3-way classification over {A, B, TIE}
            choice_logits = torch.stack(
                [
                    last_logits[:, id_A],
                    last_logits[:, id_B],
                    last_logits[:, id_T],
                ],
                dim=1,
            )  # (B, 3)

            loss_cls = F.cross_entropy(choice_logits, labels_cls)

            # Scalar preference: logit difference between A and B
            pref_scores = last_logits[:, id_A] - last_logits[:, id_B]
            loss_reg = F.mse_loss(pref_scores, labels_score)

            loss = loss_cls + lambda_reg * loss_reg
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += float(loss.item())
            total_cls += float(loss_cls.item())
            total_reg += float(loss_reg.item())
            n_steps += 1

        if n_steps == 0:
            print("[finetune] WARNING: no training steps were run.")
        else:
            print(
                f"[finetune] epoch {epoch+1}/{num_epochs} "
                f"loss={total_loss/n_steps:.4f} "
                f"cls={total_cls/n_steps:.4f} "
                f"reg={total_reg/n_steps:.4f}"
            )

        # ---- Quick validation ----
        if len(val_dataset) == 0:
            continue

        model.eval()
        correct = 0
        total = 0
        mse_sum = 0.0

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels_cls = batch["labels_cls"].to(device)
                labels_score = batch["labels_score"].to(device)

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                logits = outputs.logits  # (B, S, V)

                last_idx = attention_mask.sum(dim=1) - 1
                batch_idx = torch.arange(input_ids.size(0), device=device)
                last_logits = logits[batch_idx, last_idx, :]

                choice_logits = torch.stack(
                    [
                        last_logits[:, id_A],
                        last_logits[:, id_B],
                        last_logits[:, id_T],
                    ],
                    dim=1,
                )  # (B, 3)

                pred_cls = choice_logits.argmax(dim=1)
                correct += int((pred_cls == labels_cls).sum().item())
                total += int(labels_cls.size(0))

                pref_scores = last_logits[:, id_A] - last_logits[:, id_B]
                mse_sum += float(
                    F.mse_loss(pref_scores, labels_score, reduction="sum").item()
                )

        if total > 0:
            acc = correct / total
            mse = mse_sum / total
            print(f"[finetune] val acc={acc:.4f}, val MSE={mse:.4f}")

        model.train()

    # Optionally save the finetuned LM
    if output_dir is not None:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        print(f"[finetune] Saved finetuned judge to: {output_dir}")


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

    blocks = get_block_list(model)
    num_layers = len(blocks)

    layer_deltas: Optional[List[List[torch.Tensor]]] = None
    layer_scores: Optional[List[List[float]]] = None

    collected = 0

    for idx in indices:
        if collected >= num_traces:
            break

        ex = data[idx]
        q = ex.get("question", "")
        a = ex.get("answer_a", "")
        b = ex.get("answer_b", "")
        human = ex.get("human_preference", "TIE")

        # We only use examples where human preference is strictly A or B
        if human not in ("A", "B"):
            continue

        # Build sequential and reversed prompts
        prompt_seq = build_judge_prompt(q, a, b)  # (A,B)
        prompt_rev = build_judge_prompt(q, b, a)  # (B,A)

        # 1) Run JudgeLM to determine neutral vs biased ordering
        with torch.no_grad():
            # Sequential
            inputs_seq = tokenizer(prompt_seq, return_tensors="pt").to(device)
            outputs_seq = model(
                **inputs_seq,
                output_hidden_states=True,
                use_cache=False,
            )
            logits_seq = outputs_seq.logits
            hs_seq = list(outputs_seq.hidden_states)
            pred_lr_seq = decode_ab_tie_from_logits(logits_seq, tokenizer)
            pred_seq_content = map_lr_to_content(pred_lr_seq, "seq")

            # Reversed
            inputs_rev = tokenizer(prompt_rev, return_tensors="pt").to(device)
            outputs_rev = model(
                **inputs_rev,
                output_hidden_states=True,
                use_cache=False,
            )
            logits_rev = outputs_rev.logits
            hs_rev = list(outputs_rev.hidden_states)
            pred_lr_rev = decode_ab_tie_from_logits(logits_rev, tokenizer)
            pred_rev_content = map_lr_to_content(pred_lr_rev, "rev")

        # Determine which prompt is "neutral" (matches human) and which is "biased" (mismatch)
        neutral_hs = None
        biased_hs = None

        if pred_seq_content == human and pred_rev_content != human:
            neutral_hs = hs_seq
            biased_hs = hs_rev
        elif pred_rev_content == human and pred_seq_content != human:
            neutral_hs = hs_rev
            biased_hs = hs_seq
        else:
            # Either both correct, both incorrect, or ties; skip
            continue

        # 2) Compute preference logits and gradients for the neutral prompt
        inputs_neu = tokenizer(
            prompt_seq if neutral_hs is hs_seq else prompt_rev,
            return_tensors="pt",
        ).to(device)
        outputs_neu = model(
            **inputs_neu,
            output_hidden_states=True,
            use_cache=False,
        )
        logits_neu = outputs_neu.logits
        hs_neu = list(outputs_neu.hidden_states)

        y = pref_logit(logits_neu, tokenizer)
        grads = torch.autograd.grad(
            y,
            [hs_neu[l][:, -1, :] for l in range(1, len(hs_neu))],
            retain_graph=False,
            create_graph=False,
        )

        # 3) For each layer, compute delta and influence score
        hs_bia = biased_hs
        if hs_bia is None:
            raise RuntimeError("biased_hs is None; logic error.")

        if layer_deltas is None:
            num_layers = len(hs_neu) - 1  # exclude embedding layer 0
            layer_deltas = [[] for _ in range(num_layers)]
            layer_scores = [[] for _ in range(num_layers)]

        for l_idx in range(1, len(hs_neu)):
            # Take last-token hidden states
            h_neu = hs_neu[l_idx][:, -1, :].detach()  # (1, d)
            h_bia = hs_bia[l_idx][:, -1, :].detach()
            delta = (h_bia - h_neu)[0].cpu()  # (d,)

            # Gradient for this layer's last token
            g_tensor = grads[l_idx - 1]  # (1, d)
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


# -----------------------------
# PCA to find bias subspace
# -----------------------------


def compute_top_layers_and_bias_dirs(
    layer_deltas: List[List[torch.Tensor]],
    layer_scores: List[List[float]],
    rank_r: int = 2,
) -> Tuple[List[int], Dict[int, torch.Tensor], List[float]]:
    """
    For each layer ℓ, we have:
      - layer_deltas[ℓ]: list of vectors Δh_ℓ^i
      - layer_scores[ℓ]: list of scalar influences s_ℓ^i

    We compute:
      - mean influence per layer => select top-K layers
      - for each selected layer, run PCA on the weighted deltas to get
        a low-dimensional "bias subspace" of dimension rank_r.

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

    # Select top-k layers (here we use up to 6 layers to steer more strongly)
    k = min(6, num_layers)
    sorted_idx = np.argsort(mean_scores)
    top_layers = sorted_idx[-k:].tolist()

    bias_dirs: Dict[int, torch.Tensor] = {}
    for l in top_layers:
        deltas_l = layer_deltas[l]
        if not deltas_l:
            continue
        X = torch.stack(deltas_l, dim=0)  # (N, d)

        # Center the data
        X_centered = X - X.mean(dim=0, keepdim=True)
        X_np = X_centered.numpy()

        # PCA on centered deltas
        r = min(rank_r, X_np.shape[0], X_np.shape[1])
        pca = PCA(n_components=r)
        pca.fit(X_np)
        comps = torch.from_numpy(pca.components_)  # (r, d)
        bias_dirs[l] = comps

    return top_layers, bias_dirs, mean_scores


# -----------------------------
# Steering function
# -----------------------------


def make_steering_hook(
    W: torch.Tensor,
    alpha: float,
    device: str,
):
    """
    Create a forward hook that projects hidden states at the final token
    onto the bias subspace spanned by rows of W (shape r x d), then subtracts
    alpha * projection.
    """

    W = W.to(device)
    W_norm = W / (W.norm(dim=1, keepdim=True) + 1e-8)  # normalize rows

    def hook(module, inp, out):
        # out: (batch, seq_len, hidden_dim)
        h = out  # (B, S, d)
        B, S, d = h.shape
        h_last = h[:, -1, :]  # (B, d)

        # projection onto row space of W
        # proj = (h_last W^T) W
        proj_coeff = torch.matmul(h_last, W_norm.transpose(0, 1))  # (B, r)
        proj = torch.matmul(proj_coeff, W_norm)  # (B, d)

        h_last_debiased = h_last - alpha * proj
        h[:, -1, :] = h_last_debiased
        return h

    return hook


def register_steering_hooks(
    model: nn.Module,
    top_layers: List[int],
    bias_dirs: Dict[int, torch.Tensor],
    alpha: float,
    device: str,
):
    """
    Register forward hooks on the selected layers to steer hidden states
    along the learned bias subspace.
    """
    blocks = get_block_list(model)
    handles = []
    for l in top_layers:
        if l < 0 or l >= len(blocks):
            continue
        W = bias_dirs.get(l, None)
        if W is None:
            continue
        hook = make_steering_hook(W, alpha, device)
        h = blocks[l].register_forward_hook(hook)
        handles.append(h)
    return handles


# -----------------------------
# Proposal-style evaluation (generation-based)
# -----------------------------


def parse_choice(text: str) -> str:
    """
    Given raw generated text, parse into "A", "B", or "TIE".
    """
    t = text.strip().upper()
    if t.startswith("A"):
        return "A"
    if t.startswith("B"):
        return "B"
    if t.startswith("TIE"):
        return "TIE"
    # fallback: if none matched, treat as tie
    return "TIE"


def judge_pair_generate(
    model: nn.Module,
    tokenizer,
    device: str,
    question: str,
    left_answer: str,
    right_answer: str,
    max_length: int = 2048,
) -> str:
    """
    Build the judge prompt, let the model generate a short answer,
    and parse it into "A"/"B"/"TIE".
    """
    prompt = build_judge_prompt(question, left_answer, right_answer)
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    ).to(device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=4,
            do_sample=False,
        )

    gen_ids = output_ids[0, inputs["input_ids"].shape[1] :]
    text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    return parse_choice(text)


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
    Evaluate JudgeLM on a triplet dataset using proposal-style generation:

      - For each example, run the judge twice:
          * Sequential: Answer 1 = A, Answer 2 = B
          * Reversed:   Answer 1 = B, Answer 2 = A
      - The model generates a short completion, which is parsed into "A"/"B"/"TIE".
      - We compute:
          * pah_seq / pah_rev: accuracy w.r.t. human preference for non-tie human & non-tie model predictions
          * left_rate_seq / left_rate_rev: fraction of non-tie model predictions that choose left
          * tie_rate_seq / tie_rate_rev: fraction of total samples where model outputs tie
          * pbi = left_rate_seq - left_rate_rev
          * avg_pah = (pah_seq + pah_rev) / 2
    """

    # Attach steering hooks if provided
    handles: List[Any] = []
    if bias_dirs is not None and top_layers is not None and alpha is not None:
        handles = register_steering_hooks(model, top_layers, bias_dirs, alpha, device)

    model.eval()

    total = 0
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

            # Sequential (A,B): Answer 1 = A, Answer 2 = B
            pred_seq_lr = judge_pair_generate(model, tokenizer, device, q, a, b)
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

            # Reversed (B,A): Answer 1 = B, Answer 2 = A
            pred_rev_lr = judge_pair_generate(model, tokenizer, device, q, b, a)
            pred_rev_content = map_lr_to_content(pred_rev_lr, "rev")

            if pred_rev_lr == "TIE":
                tie_rev += 1
            else:
                non_tie_rev += 1
                if pred_rev_lr == "A":  # "A" means choose left, i.e., B in content space
                    left_rev += 1

            if human in ("A", "B") and pred_rev_content in ("A", "B"):
                denom_rev += 1
                if pred_rev_content == human:
                    correct_rev += 1

            total += 1

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


# -----------------------------
# Collect score lists & Kendall's tau
# -----------------------------


def collect_pref_scores_for_dataset(
    model: nn.Module,
    tokenizer,
    data: List[Dict[str, Any]],
    device: str,
    bias_dirs: Optional[Dict[int, torch.Tensor]] = None,
    top_layers: Optional[List[int]] = None,
    alpha: Optional[float] = None,
    max_length: int = 2048,
) -> Tuple[List[float], List[float]]:
    """
    For each example in `data`, compute scalar preference scores
    (logit_A - logit_B) under sequential and reversed answer orders.

    If bias_dirs/top_layers/alpha are provided, steering hooks are
    registered (Align-Trace debiasing); otherwise baseline scores
    are collected.

    Returns:
      scores_seq: list of scores for (A,B) order
      scores_rev: list of scores for (B,A) order
    """
    handles: List[Any] = []
    if bias_dirs is not None and top_layers is not None and alpha is not None:
        handles = register_steering_hooks(model, top_layers, bias_dirs, alpha, device)

    model.eval()
    scores_seq: List[float] = []
    scores_rev: List[float] = []

    try:
        with torch.no_grad():
            for ex in data:
                q = ex.get("question", "")
                a = ex.get("answer_a", "")
                b = ex.get("answer_b", "")

                # Sequential (A,B)
                prompt_seq = build_judge_prompt(q, a, b)
                inputs_seq = tokenizer(
                    prompt_seq,
                    return_tensors="pt",
                    truncation=True,
                    max_length=max_length,
                ).to(device)
                out_seq = model(**inputs_seq)
                logits_seq = out_seq.logits
                s_seq = float(pref_logit(logits_seq, tokenizer).item())
                scores_seq.append(s_seq)

                # Reversed (B,A)
                prompt_rev = build_judge_prompt(q, b, a)
                inputs_rev = tokenizer(
                    prompt_rev,
                    return_tensors="pt",
                    truncation=True,
                    max_length=max_length,
                ).to(device)
                out_rev = model(**inputs_rev)
                logits_rev = out_rev.logits
                s_rev = float(pref_logit(logits_rev, tokenizer).item())
                scores_rev.append(s_rev)
    finally:
        for h in handles:
            h.remove()

    return scores_seq, scores_rev


# -----------------------------
# Simple scalar score for validation search
# -----------------------------


def score_metrics(metrics: Dict[str, float]) -> float:
    """
    Convert a metrics dict into a single scalar score for alpha search:

      score = avg_pah - |pbi|

    so we prefer high accuracy and low position bias.
    """
    avg_pah = metrics.get("avg_pah", 0.0)
    pbi = metrics.get("pbi", 0.0)
    return float(avg_pah - abs(pbi))


# -----------------------------
# Main pipeline
# -----------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Align-Trace debias pipeline for JudgeLM (proposal-style eval).")
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
    parser.add_argument(
        "--do_finetune",
        action="store_true",
        help="If set, first supervised-finetune JudgeLM on comparison data "
             "to predict discrete winners + scalar preference scores.",
    )
    parser.add_argument("--ft_epochs", type=int, default=1, help="Number of epochs for judge SFT")
    parser.add_argument("--ft_batch_size", type=int, default=4, help="Batch size for judge SFT")
    parser.add_argument("--ft_lr", type=float, default=5e-6, help="Learning rate for judge SFT")
    parser.add_argument(
        "--ft_lambda_reg",
        type=float,
        default=1.0,
        help="Weight for scalar preference regression loss in SFT",
    )
    parser.add_argument(
        "--ft_max_length",
        type=int,
        default=2048,
        help="Max prompt length for SFT tokenization",
    )
    parser.add_argument(
        "--ft_output_dir",
        type=str,
        default=None,
        help="Optional directory to save the finetuned judge (model+tokenizer).",
    )
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
        if args.do_finetune:
            # For SFT we need the model on a single device (no device_map).
            model = AutoModelForCausalLM.from_pretrained(
                args.model_name,
                torch_dtype=torch.float16,
            )
            model.to(device)
        else:
            model = AutoModelForCausalLM.from_pretrained(
                args.model_name,
                torch_dtype=torch.float16,
                device_map="auto",
            )
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model_name)
        model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    # Ensure pad token exists
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    # Optional: supervised finetuning on comparison data before Align-Trace
    if args.do_finetune:
        print("[finetune] Starting supervised finetuning on train/val comparisons...")
        finetune_judge(
            model=model,
            tokenizer=tokenizer,
            train_triplets=train_data,
            val_triplets=val_data,
            device=device,
            num_epochs=args.ft_epochs,
            batch_size=args.ft_batch_size,
            lr=args.ft_lr,
            lambda_reg=args.ft_lambda_reg,
            max_length=args.ft_max_length,
            output_dir=args.ft_output_dir,
        )
        print("[finetune] Finished finetuning JudgeLM. Proceeding to Align-Trace...")

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

    # 3) Evaluate on val to choose alpha (proposal-style generation-based eval)
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

    # Save validation metrics
    metrics_val_path = out_dir / "metrics_val.json"
    with open(metrics_val_path, "w", encoding="utf-8") as f:
        json.dump(val_metrics_all, f, indent=2)
    print(f"[save] Saved validation metrics to: {metrics_val_path}")

    # 4) Evaluate on test with baseline & best-alpha steering
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

    # Kendall's tau between baseline and debiased score lists
    print("[test] Computing Kendall's tau between baseline and debiased score lists...")
    base_scores_seq, base_scores_rev = collect_pref_scores_for_dataset(
        model=model,
        tokenizer=tokenizer,
        data=test_data,
        device=device,
        bias_dirs=None,
        top_layers=None,
        alpha=None,
    )
    deb_scores_seq, deb_scores_rev = collect_pref_scores_for_dataset(
        model=model,
        tokenizer=tokenizer,
        data=test_data,
        device=device,
        bias_dirs=bias_dirs,
        top_layers=top_layers,
        alpha=float(best_alpha),
    )
    tau_seq = kendall_tau(base_scores_seq, deb_scores_seq)
    tau_rev = kendall_tau(base_scores_rev, deb_scores_rev)
    print(f"[test] Kendall tau (seq) baseline vs debiased: {tau_seq:.4f}")
    print(f"[test] Kendall tau (rev) baseline vs debiased: {tau_rev:.4f}")

    metrics_test = {
        "baseline": test_baseline,
        "debiased": test_debiased,
        "best_alpha": best_alpha,
        "kendall_tau_seq": tau_seq,
        "kendall_tau_rev": tau_rev,
    }
    metrics_test_path = out_dir / "metrics_test.json"
    with open(metrics_test_path, "w", encoding="utf-8") as f:
        json.dump(metrics_test, f, indent=2)
    print(f"[save] Saved test metrics to: {metrics_test_path}")


if __name__ == "__main__":
    main()
