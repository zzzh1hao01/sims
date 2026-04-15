"""
Prompt formatting and option-token extraction for SubPOP fine-tuning.

Design notes
------------
* We use single capital letters (A, B, C, ...) as the answer proxy tokens.
  These are guaranteed to be single tokens in Qwen2.5's tiktoken-based vocab.
* The prompt ends with add_generation_prompt=True so the last token is the
  assistant turn opener (<|im_start|>assistant\n).  The next-token logits at
  that position are what we index into for KL loss and inference.
"""

SYSTEM_PROMPT = (
    "You are simulating how a specific demographic group responds to survey "
    "questions. Given a person's demographic background and a survey question, "
    "provide a probability distribution over the answer choices."
)


class PromptFormatter:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self._validate_option_tokens()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def format(self, row: dict) -> str:
        """
        Build a Qwen2.5 ChatML prompt string (not yet tokenized).

        row must contain: attribute, group, question, options (list[str])
        """
        options_block = "\n".join(
            f"{chr(65 + i)}. {opt}" for i, opt in enumerate(row["options"])
        )
        user_content = (
            f"Demographic background: {row['attribute']}: {row['group']}\n\n"
            f"Survey question: {row['question']}\n\n"
            f"Answer choices:\n{options_block}\n\n"
            "What is the probability distribution over the answer choices for "
            "this demographic group? Respond with ONLY the answer choice "
            "letter (A, B, C, ...) and its probability."
        )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    def get_option_token_ids(self, options: list) -> list:
        """
        Return a list of int token IDs, one per answer option.
        Each option maps to the single-token ID of its letter (A, B, C, ...).
        """
        return [
            self.tokenizer.encode(chr(65 + i), add_special_tokens=False)[-1]
            for i in range(len(options))
        ]

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _validate_option_tokens(self, max_options: int = 7):
        """Assert that A–G are each exactly one token with no collisions."""
        ids = []
        for i in range(max_options):
            letter = chr(65 + i)
            token_ids = self.tokenizer.encode(letter, add_special_tokens=False)
            if len(token_ids) != 1:
                raise ValueError(
                    f"Letter '{letter}' tokenizes to multiple tokens: {token_ids}. "
                    "Option-letter strategy requires single-token letters."
                )
            ids.append(token_ids[0])
        if len(set(ids)) != len(ids):
            raise ValueError(f"Token ID collision among option letters: {ids}")
