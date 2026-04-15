"""
Loss functions for SubPOP distribution fine-tuning.

Core idea
---------
For each example in the batch:
  1. Find the last non-padding token position (end of the assistant turn
     opener injected by add_generation_prompt=True).
  2. Extract the raw logits at that position for each answer-option letter
     token (A, B, C, ...).
  3. Compute Forward KL divergence: KL(P_human || P_model).

Why forward KL (not reverse)?
  Forward KL forces the model to assign non-negligible probability wherever
  humans assign non-negligible probability — it penalizes mode-dropping.
  Survey distributions are genuinely multimodal (different people choose
  different options), so we must not collapse to a single mode.

Numerical stability
-------------------
  * log_softmax is used instead of log(softmax(x)) to avoid the two-pass
    computation that can overflow.
  * Human probabilities are clamped away from zero before taking log to
    avoid log(0) = -inf.  After clamping we renormalize so the distribution
    still sums to 1.
"""

import torch
import torch.nn.functional as F


def extract_option_logits(
    logits: torch.Tensor,
    attention_mask: torch.Tensor,
    option_token_ids: list,
) -> list:
    """
    Extract raw logits for each answer option at the last prompt token.

    Args:
        logits:           [batch, seq_len, vocab_size]
        attention_mask:   [batch, seq_len]
        option_token_ids: list of LongTensor[num_options], one per example

    Returns:
        List of FloatTensor[num_options], one raw-logit vector per example.
    """
    # Last non-padding position for each example
    last_positions = attention_mask.sum(dim=1) - 1   # [batch]

    result = []
    for i, pos in enumerate(last_positions):
        logit_vec = logits[i, pos.item(), :]          # [vocab_size]
        opt_ids = option_token_ids[i].to(logits.device)
        result.append(logit_vec[opt_ids])             # [num_options]
    return result


def forward_kl_loss(
    batch_option_logits: list,
    target_distributions: list,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    """
    Mean Forward KL: KL(P_human || P_model) averaged over the batch.

    KL(P || Q) = Σ_k P(k) * [log P(k) - log Q(k)]

    Args:
        batch_option_logits:  list of FloatTensor[num_options] (raw logits)
        target_distributions: list of FloatTensor[num_options] (human probs)
        epsilon:              floor for human probs before taking log

    Returns:
        Scalar loss tensor.
    """
    losses = []
    for opt_logits, target in zip(batch_option_logits, target_distributions):
        log_p_model = F.log_softmax(opt_logits.float(), dim=-1)   # [num_options]

        p_human = target.to(opt_logits.device).float()
        p_human = torch.clamp(p_human, min=epsilon)
        p_human = p_human / p_human.sum()                         # renormalize
        log_p_human = torch.log(p_human)

        kl = (p_human * (log_p_human - log_p_model)).sum()
        losses.append(kl)

    return torch.stack(losses).mean()
