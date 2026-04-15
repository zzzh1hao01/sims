"""
SubPOPTrainer — HuggingFace Trainer subclass that overrides compute_loss
to use Forward KL divergence over option-letter logits.

Everything else (gradient accumulation, bf16, LR scheduling, checkpointing,
distributed training, W&B logging) is inherited from the parent Trainer.

IMPORTANT: TrainingArguments must be created with remove_unused_columns=False
so that option_token_ids, target_distributions, and ordinals are not silently
dropped from the batch before reaching compute_loss.
"""

from transformers import Trainer

from training.loss import extract_option_logits, forward_kl_loss


class SubPOPTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Pop non-model-forward keys before passing to the model
        option_token_ids = inputs.pop("option_token_ids")
        target_distributions = inputs.pop("target_distributions")
        inputs.pop("ordinals", None)   # not used in loss, kept for eval hooks

        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
        )

        batch_option_logits = extract_option_logits(
            logits=outputs.logits,
            attention_mask=inputs["attention_mask"],
            option_token_ids=option_token_ids,
        )

        loss = forward_kl_loss(batch_option_logits, target_distributions)

        return (loss, outputs) if return_outputs else loss
