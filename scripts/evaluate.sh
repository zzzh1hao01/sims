#!/usr/bin/env bash
# Run evaluation on SubPOP-Eval (GSS) after training.
# Usage: bash scripts/evaluate.sh outputs/subpop-qwen2.5-7b/final
set -euo pipefail

CHECKPOINT="${1:?Usage: $0 <lora_checkpoint_dir>}"

python - <<EOF
from inference.predictor import SubPOPPredictor
from eval.evaluate import SubPOPEvaluator
import json

predictor = SubPOPPredictor(
    base_model_name="Qwen/Qwen2.5-7B-Instruct",
    lora_checkpoint_dir="$CHECKPOINT",
)
evaluator = SubPOPEvaluator(predictor)
results = evaluator.run()
print(json.dumps(results, indent=2))
EOF
