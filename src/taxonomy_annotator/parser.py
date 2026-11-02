"""Parsing and strict validation of structured LLM annotation responses."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Set

from .models import AnnotationResult, ErrorAnnotation


class AnnotationParser:
    """Validate the minimal ``errors`` JSON schema and taxonomy codes."""

    def __init__(self, allowed_codes: Set[str]) -> None:
        self.allowed_codes = allowed_codes

    def parse(self, raw_text: str) -> AnnotationResult:
        if not isinstance(raw_text, str):
            raise ValueError("Model response must be a string.")
        try:
            parsed = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            parsed = self._parse_fenced_json(raw_text, exc)
        if not isinstance(parsed, dict) or set(parsed) != {"errors"}:
            raise ValueError('Annotation must be a JSON object with only an "errors" key.')
        if not isinstance(parsed["errors"], list):
            raise ValueError('Annotation "errors" must be an array.')

        errors: list[ErrorAnnotation] = []
        seen: set[tuple[str, str]] = set()
        for index, value in enumerate(parsed["errors"]):
            if not isinstance(value, dict) or set(value) != {
                "code", "location", "explanation"
            }:
                raise ValueError(
                    f'Annotation errors[{index}] must contain only "code", '
                    '"location", and "explanation".'
                )
            code, location, explanation = (
                value["code"], value["location"], value["explanation"]
            )
            if not all(
                isinstance(item, str) and item.strip()
                for item in (code, location, explanation)
            ):
                raise ValueError(f"Annotation errors[{index}] contains an empty value.")
            if code not in self.allowed_codes:
                raise ValueError(
                    f"Unknown taxonomy code {code!r} in annotation errors[{index}]."
                )
            normalized = (code.strip(), location.strip())
            if normalized in seen:
                raise ValueError(f"Duplicate error annotation at errors[{index}].")
            seen.add(normalized)
            errors.append(
                ErrorAnnotation(
                    code=code.strip(),
                    location=location.strip(),
                    explanation=explanation.strip(),
                )
            )
        return AnnotationResult(errors=errors)

    @staticmethod
    def _parse_fenced_json(
        raw_text: str,
        original_error: json.JSONDecodeError,
    ) -> Any:
        """Parse a JSON code fence only when the complete response is not JSON."""
        for match in re.finditer(
            r"```(?:\s*json\b)?\s*(.*?)```",
            raw_text,
            flags=re.IGNORECASE | re.DOTALL,
        ):
            try:
                return json.loads(match.group(1).strip())
            except json.JSONDecodeError:
                continue

        raise ValueError(
            f"Model response is not valid JSON and contains no valid JSON code "
            f"fence: {original_error}"
        ) from original_error
