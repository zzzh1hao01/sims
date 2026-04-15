"""
Build the PEFT LoraConfig from the 'lora' section of default.yaml.

Paper settings (SubPOP, ACL 2025):
  rank=8, alpha=32, dropout=0.05, target_modules=[q_proj, v_proj]
"""

from peft import LoraConfig, TaskType


def build_lora_config(cfg: dict) -> LoraConfig:
    """
    cfg: the 'lora' sub-dict from default.yaml.
    """
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=cfg["rank"],
        lora_alpha=cfg["alpha"],
        lora_dropout=cfg["dropout"],
        target_modules=cfg["target_modules"],
        bias="none",
        inference_mode=False,
    )
