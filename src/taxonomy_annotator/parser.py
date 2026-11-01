"""Robust parsing and validation of LLM annotation responses."""

from __future__ import annotations

import ast
import json
import re
from typing import Any, Dict, Optional

from .models import AnnotationResult, ErrorClassification


class AnnotationParser:
    """
    Parse common LLM JSON variants into a validated ``AnnotationResult``.

    Supported forms include raw JSON, fenced JSON, JSON with explanatory
    prefix/suffix text, and Python-dictionary-like single-quoted output.
    """

    REQUIRED_KEYS = {
        "applicable_errors",
        "primary_error",
        "reason",
        "code",
        "category",
    }

    @staticmethod
    def strip_code_fences(text: str) -> str:
        """Remove common Markdown code-fence wrappers."""
        cleaned = text.strip()
        cleaned = re.sub(
            r"^\s*```(?:json|javascript|js)?\s*",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )
        cleaned = re.sub(r"\s*```\s*$", "", cleaned)
        return cleaned.strip()

    @staticmethod
    def first_balanced_object(text: str) -> Optional[str]:
        """
        Extract the first balanced ``{...}`` object while respecting strings.

        This avoids the brittle "first opening brace to last closing brace"
        approach when the model adds extra prose.
        """
        start = text.find("{")
        if start < 0:
            return None

        depth = 0
        in_string = False
        escaped = False

        for index in range(start, len(text)):
            char = text[index]

            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue

            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return text[start : index + 1]

        return None

    @staticmethod
    def _parse_python_literal(text: str) -> Optional[Dict[str, Any]]:
        """Safely parse Python-literal dictionaries without executing code."""
        try:
            value = ast.literal_eval(text)
        except (ValueError, SyntaxError):
            return None
        return value if isinstance(value, dict) else None

    def parse(self, raw_text: str) -> AnnotationResult:
        """Parse and validate a raw model response."""
        if not isinstance(raw_text, str):
            raise ValueError("Model response must be a string.")

        cleaned = self.strip_code_fences(raw_text)
        candidates = [cleaned]

        balanced = self.first_balanced_object(cleaned)
        if balanced and balanced != cleaned:
            candidates.append(balanced)

        parsed: Optional[Dict[str, Any]] = None

        for candidate in candidates:
            try:
                value = json.loads(candidate)
                if isinstance(value, dict):
                    parsed = value
                    break
            except json.JSONDecodeError:
                pass

            parsed = self._parse_python_literal(candidate)
            if parsed is not None:
                break

        if parsed is None:
            raise ValueError(
                "Could not parse a JSON object from the model response. "
                f"Raw response: {raw_text[:1000]}"
            )

        missing = self.REQUIRED_KEYS.difference(parsed.keys())
        if missing:
            raise ValueError(
                f"Annotation output missing required keys: {sorted(missing)}"
            )

        primary = parsed["primary_error"]
        if not isinstance(primary, dict):
            raise ValueError("Annotation output primary_error must be an object.")

        primary_code = str(primary.get("code", "")).strip()
        primary_category = str(primary.get("category", "")).strip()
        if not all([primary_code, primary_category]):
            raise ValueError("Annotation output primary_error is incomplete.")

        applicable = parsed["applicable_errors"]
        if not isinstance(applicable, list) or not applicable:
            raise ValueError(
                "Annotation output applicable_errors must be a non-empty list."
            )

        applicable_errors = []
        for index, error in enumerate(applicable):
            if not isinstance(error, dict):
                raise ValueError(
                    f"Annotation output applicable_errors[{index}] must be an object."
                )
            error_code = str(error.get("code", "")).strip()
            error_category = str(error.get("category", "")).strip()
            if not all([error_code, error_category]):
                raise ValueError(
                    f"Annotation output applicable_errors[{index}] is incomplete."
                )
            applicable_errors.append(
                ErrorClassification(code=error_code, category=error_category)
            )

        if not any(
            error.code == primary_code and error.category == primary_category
            for error in applicable_errors
        ):
            raise ValueError(
                "Annotation output primary_error must be included in "
                "applicable_errors."
            )

        code = str(parsed["code"]).strip()
        category = str(parsed["category"]).strip()
        reason = str(parsed["reason"]).strip()

        if not all([code, category, reason]):
            raise ValueError("Annotation output contains an empty required value.")

        if code != primary_code or category != primary_category:
            raise ValueError(
                "Top-level code/category must match primary_error.code/category."
            )

        primary_error = ErrorClassification(
            code=primary_code,
            category=primary_category,
        )
        return AnnotationResult(
            code=code,
            category=category,
            reason=reason,
            applicable_errors=applicable_errors,
            primary_error=primary_error,
        )
