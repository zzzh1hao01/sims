"""
SubPOPPredictor — loads a fine-tuned LoRA checkpoint and runs inference.

Usage example
-------------
predictor = SubPOPPredictor(
    base_model_name="Qwen/Qwen2.5-7B-Instruct",
    lora_checkpoint_dir="outputs/subpop-qwen2.5-7b/checkpoint-1000",
)
probs = predictor.predict(
    attribute="Political Party",
    group="Democrat",
    question="Do you think the federal government should reduce spending on social programs?",
    options=["Strongly Agree", "Agree", "Neutral", "Disagree", "Strongly Disagree"],
)
# probs → {"Strongly Agree": 0.08, "Agree": 0.14, ..., "Strongly Disagree": 0.21}
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from data.prompt_formatter import PromptFormatter


class SubPOPPredictor:
    def __init__(
        self,
        base_model_name: str,
        lora_checkpoint_dir: str,
        device: str = "cuda",
    ):
        self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model_name,
            padding_side="right",
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.bfloat16,
            device_map=device,
            trust_remote_code=True,
        )
        self.model = PeftModel.from_pretrained(base_model, lora_checkpoint_dir)
        self.model.eval()

        self.formatter = PromptFormatter(self.tokenizer)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict(
        self,
        attribute: str,
        group: str,
        question: str,
        options: list,
    ) -> dict:
        """
        Returns a dict mapping option text → probability (floats sum to 1.0).
        """
        row = {
            "attribute": attribute,
            "group": group,
            "question": question,
            "options": options,
        }
        prompt = self.formatter.format(row)
        option_token_ids = self.formatter.get_option_token_ids(options)

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(self.model.device)

        outputs = self.model(**inputs)

        last_pos = inputs["attention_mask"].sum() - 1
        logit_vec = outputs.logits[0, last_pos, :]       # [vocab_size]
        opt_ids = torch.tensor(option_token_ids, device=logit_vec.device)
        opt_logits = logit_vec[opt_ids]
        probs = F.softmax(opt_logits.float(), dim=-1)

        return {opt: probs[i].item() for i, opt in enumerate(options)}

    @torch.no_grad()
    def predict_batch(self, rows: list) -> list:
        """
        Batched inference.  Each row dict must have attribute, group,
        question, options.  Returns list of dicts, same order as input.
        """
        prompts = [self.formatter.format(r) for r in rows]
        all_option_ids = [self.formatter.get_option_token_ids(r["options"]) for r in rows]

        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        ).to(self.model.device)

        outputs = self.model(**inputs)

        results = []
        last_positions = inputs["attention_mask"].sum(dim=1) - 1
        for i, row in enumerate(rows):
            pos = last_positions[i].item()
            logit_vec = outputs.logits[i, pos, :]
            opt_ids = torch.tensor(all_option_ids[i], device=logit_vec.device)
            probs = F.softmax(logit_vec[opt_ids].float(), dim=-1)
            results.append({opt: probs[j].item() for j, opt in enumerate(row["options"])})

        return results
