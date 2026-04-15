"""
Model and tokenizer loading utilities.

Design decisions
----------------
* bf16 throughout — A5000 / A100 both support it natively and it halves
  memory vs fp32 with no meaningful precision loss for fine-tuning.
* device_map="auto" — lets accelerate shard across 2 GPUs if needed,
  though with LoRA the full model fits on one GPU per DDP replica.
* use_cache must be False before enabling gradient checkpointing.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model

from models.lora_config import build_lora_config


def load_model_and_tokenizer(cfg: dict):
    """
    Load Qwen2.5-7B-Instruct in bf16.
    cfg: the top-level config dict (uses cfg['model']).
    """
    model_name = cfg["model"]["name"]
    trust_remote_code = cfg["model"].get("trust_remote_code", True)

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        padding_side="right",
        trust_remote_code=trust_remote_code,
    )
    # Qwen2.5 has a pad token; set it to eos as fallback just in case.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=trust_remote_code,
    )
    # Required before gradient_checkpointing_enable()
    model.config.use_cache = False

    return model, tokenizer


def apply_lora(model, lora_cfg: dict):
    """
    Wrap model with LoRA adapters and print the trainable parameter count.
    lora_cfg: the 'lora' sub-dict from default.yaml.
    """
    config = build_lora_config(lora_cfg)
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    return model
