"""
SubPOPEvaluator — runs the full evaluation loop on SubPOP-Eval (GSS, 133 questions).

Usage
-----
from inference.predictor import SubPOPPredictor
from eval.evaluate import SubPOPEvaluator

predictor = SubPOPPredictor(
    base_model_name="Qwen/Qwen2.5-7B-Instruct",
    lora_checkpoint_dir="outputs/subpop-qwen2.5-7b/final",
)
evaluator = SubPOPEvaluator(predictor)
results = evaluator.run()
print(results)
"""

from collections import defaultdict

import numpy as np
from datasets import load_dataset
from tqdm import tqdm

from eval.metrics import (
    compute_wasserstein,
    majority_wasserstein,
    uniform_wasserstein,
)


class SubPOPEvaluator:
    def __init__(self, predictor):
        self.predictor = predictor

    def run(self, batch_size: int = 16) -> dict:
        """
        Evaluate on SubPOP-Eval (hf split "test").

        Returns a dict with:
          mean_wasserstein       : float
          std_wasserstein        : float
          uniform_baseline       : float  (mean WD for uniform dist)
          majority_baseline      : float  (mean WD for majority-vote dist)
          per_attribute          : dict[attribute, mean WD]
          n_examples             : int
        """
        dataset = load_dataset("jjssuh/subpop", split="test")
        records = list(dataset)

        all_wd, all_wd_uniform, all_wd_majority = [], [], []
        per_attribute = defaultdict(list)

        for start in tqdm(range(0, len(records), batch_size), desc="Evaluating"):
            batch = records[start : start + batch_size]
            pred_list = self.predictor.predict_batch(batch)

            for row, pred_dict in zip(batch, pred_list):
                options = row["options"]
                target = np.array(row["responses"], dtype=np.float64)
                ordinals = np.array(row["ordinal"], dtype=np.float64)
                predicted = np.array([pred_dict[opt] for opt in options], dtype=np.float64)

                wd = compute_wasserstein(predicted, target, ordinals)
                wd_u = uniform_wasserstein(len(options), ordinals, target)
                wd_m = majority_wasserstein(ordinals, target)

                all_wd.append(wd)
                all_wd_uniform.append(wd_u)
                all_wd_majority.append(wd_m)
                per_attribute[row["attribute"]].append(wd)

        return {
            "mean_wasserstein": float(np.mean(all_wd)),
            "std_wasserstein": float(np.std(all_wd)),
            "uniform_baseline": float(np.mean(all_wd_uniform)),
            "majority_baseline": float(np.mean(all_wd_majority)),
            "per_attribute": {
                attr: float(np.mean(vals))
                for attr, vals in per_attribute.items()
            },
            "n_examples": len(all_wd),
        }
