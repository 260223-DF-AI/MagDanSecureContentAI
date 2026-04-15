from __future__ import annotations

import json
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.description_analysis.policy import DEFAULT_POLICY, validate_and_sanitize


class LLMService:
    """
    LLM-backed moderation service for post descriptions.

    Uses the same Hugging Face model pattern as the finished repo's
    description analysis module, but returns a FastAPI-friendly payload.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
        max_new_tokens: int = 300,
    ) -> None:
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens

        # Load tokenizer + model once when the service starts
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype="auto",
            device_map="auto",
        )
        self.model.eval()

    def _build_prompt(self, text: str, vision_result: dict[str, Any]) -> str:
        """
        Build the moderation prompt using the repo's policy plus image context.
        """
        predicted_class = vision_result.get("predicted_class", "unknown")
        image_allowed = vision_result.get("is_post_allowed", False)

        return (
            f"{DEFAULT_POLICY}\n\n"
            f"Image moderation context:\n"
            f"- predicted_class: {predicted_class}\n"
            f"- image_allowed: {image_allowed}\n\n"
            f"Post description:\n{text}\n\n"
            f"Return JSON only in this format:\n"
            f'{{"reasoning": "short explanation", "decision": "safe" or "unsafe"}}'
        )

    def _generate_model_output(self, prompt: str) -> str:
        """
        Tokenize prompt and run generation.
        """
        messages = [{"role": "user", "content": prompt}]

        formatted_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.tokenizer([formatted_text], return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
            )

        prompt_len = inputs["input_ids"].shape[1]
        generated_ids = output_ids[0][prompt_len:]
        response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        return response.strip()

    def _parse_response(self, raw_response: str) -> tuple[str, str]:
        """
        Try to parse JSON first. If the model doesn't return perfect JSON,
        fall back to keyword-based extraction.
        """
        try:
            parsed = json.loads(raw_response)
            decision = str(parsed.get("decision", "unsafe")).strip().lower()
            reasoning = str(parsed.get("reasoning", raw_response)).strip()

            if decision not in {"safe", "unsafe"}:
                decision = "unsafe"

            return decision, reasoning
        except Exception:
            upper = raw_response.upper()

            if "UNSAFE" in upper:
                return "unsafe", raw_response
            if "SAFE" in upper:
                return "safe", raw_response

            return "unsafe", raw_response

    def analyze_text(self, text: str, vision_result: dict[str, Any]) -> dict[str, Any]:
        """
        Main FastAPI-facing method.
        Returns the shape expected by your /analyze route.
        """
        sanitized_text = validate_and_sanitize(text)
        prompt = self._build_prompt(sanitized_text, vision_result)
        raw_response = self._generate_model_output(prompt)
        decision, reasoning = self._parse_response(raw_response)

        # Cross-signal enforcement:
        # if the image was blocked, final text moderation should also block
        if not vision_result.get("is_post_allowed", False):
            decision = "unsafe"
            reasoning = (
                f"{reasoning} The associated image was also blocked by the vision model."
            ).strip()

        is_comment_allowed = decision == "safe"
        moderation_label = "safe" if is_comment_allowed else "flagged"
        suggested_response = (
            "Approved"
            if is_comment_allowed
            else "Policy Violation: Image or text violates moderation policy."
        )

        return {
            "moderation_label": moderation_label,
            "is_comment_allowed": is_comment_allowed,
            "reason": reasoning,
            "suggested_response": suggested_response,
            # optional debug fields if you want them available later
            "raw_model_output": raw_response,
            "sanitized_text": sanitized_text,
        }