"""
train.py — main entry point for SubPOP fine-tuning.

Typical invocation (single node, 2 GPUs via torchrun):
  torchrun --nproc_per_node=2 train.py --config configs/default.yaml

CLI overrides use dot notation, e.g.:
  --overrides training.learning_rate=1e-4 lora.rank=16
"""

import argparse
import os
import yaml

from transformers import TrainingArguments

from data.dataset import SubPOPDataset
from data.prompt_formatter import PromptFormatter
from data.collator import SubPOPCollator
from models.model_utils import load_model_and_tokenizer, apply_lora
from training.trainer import SubPOPTrainer


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def apply_overrides(cfg: dict, overrides: list):
    """
    Apply dot-notation overrides, e.g. "training.learning_rate=2e-4".
    Values are parsed via yaml.safe_load for automatic type coercion.
    """
    for item in overrides:
        key, val = item.split("=", 1)
        keys = key.split(".")
        d = cfg
        for k in keys[:-1]:
            d = d.setdefault(k, {})
        d[keys[-1]] = yaml.safe_load(val)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Fine-tune Qwen2.5 on SubPOP")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--overrides",
        nargs="*",
        default=[],
        help="Dot-notation overrides, e.g. training.learning_rate=1e-4",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    apply_overrides(cfg, args.overrides)

    # W&B project name (set before Trainer initializes it)
    if cfg["training"].get("report_to") == "wandb":
        os.environ.setdefault("WANDB_PROJECT", cfg["training"].get("wandb_project", "subpop-finetune"))

    # ------------------------------------------------------------------
    # Model + tokenizer
    # ------------------------------------------------------------------
    model, tokenizer = load_model_and_tokenizer(cfg)
    model = apply_lora(model, cfg["lora"])

    # ------------------------------------------------------------------
    # Datasets
    # ------------------------------------------------------------------
    formatter = PromptFormatter(tokenizer)
    data_cfg = cfg["data"]

    train_dataset = SubPOPDataset(
        hf_split="train",
        tokenizer=tokenizer,
        formatter=formatter,
        max_length=data_cfg["max_length"],
        val_fraction=data_cfg["val_fraction"],
        val_seed=data_cfg["val_seed"],
        is_val=False,
    )
    val_dataset = SubPOPDataset(
        hf_split="train",
        tokenizer=tokenizer,
        formatter=formatter,
        max_length=data_cfg["max_length"],
        val_fraction=data_cfg["val_fraction"],
        val_seed=data_cfg["val_seed"],
        is_val=True,
    )

    collator = SubPOPCollator(pad_token_id=tokenizer.pad_token_id)

    # ------------------------------------------------------------------
    # Training arguments
    # ------------------------------------------------------------------
    t = cfg["training"]
    training_args = TrainingArguments(
        output_dir=t["output_dir"],
        num_train_epochs=t["num_epochs"],
        per_device_train_batch_size=t["batch_size"],
        per_device_eval_batch_size=t["eval_batch_size"],
        gradient_accumulation_steps=t["gradient_accumulation_steps"],
        learning_rate=t["learning_rate"],
        lr_scheduler_type="cosine",
        warmup_ratio=t["warmup_ratio"],
        weight_decay=t["weight_decay"],
        bf16=True,
        gradient_checkpointing=True,
        # DDP: don't error on LoRA params that don't get gradients
        ddp_find_unused_parameters=False,
        eval_strategy="steps",
        eval_steps=t["eval_steps"],
        save_strategy="steps",
        save_steps=t["save_steps"],
        save_total_limit=t.get("save_total_limit", 3),
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=10,
        report_to=t.get("report_to", "wandb"),
        dataloader_num_workers=4,
        # CRITICAL: keep option_token_ids / target_distributions in the batch
        remove_unused_columns=False,
    )

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    trainer = SubPOPTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
    )

    trainer.train()

    # Save final adapter + tokenizer
    final_dir = os.path.join(t["output_dir"], "final")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"Model saved to {final_dir}")


if __name__ == "__main__":
    main()
