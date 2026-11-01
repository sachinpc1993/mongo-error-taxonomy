"""Application service coordinating prompts, retries, model calls, and output."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List

from .io_utils import JsonlRepository
from .models import RuntimeMetadata
from .parser import AnnotationParser
from .prompt import PromptBuilder
from .providers import LLMProvider


class AnnotationService:
    """Coordinates one complete dataset annotation run."""

    def __init__(
        self,
        provider: LLMProvider,
        prompt_builder: PromptBuilder,
        parser: AnnotationParser,
        output_path: str | Path,
        max_retries: int = 3,
        resume: bool = True,
        provider_name: str = "",
    ) -> None:
        """Initialize the service with injected strategies and runtime settings."""
        self.provider = provider
        self.prompt_builder = prompt_builder
        self.parser = parser
        self.output_path = Path(output_path)
        self.max_retries = max(1, int(max_retries))
        self.resume = resume
        self.provider_name = provider_name

    @staticmethod
    def item_id(entry: Dict[str, Any], index: int) -> str:
        """Resolve a stable item ID from ``id``, ``question_id``, or row number."""
        if entry.get("id") is not None:
            return str(entry["id"])
        if entry.get("question_id") is not None:
            return str(entry["question_id"])
        return str(index + 1)

    def _annotate_once(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        """Run a single annotation attempt and return the validated result."""
        prompt = self.prompt_builder.build(entry)
        raw_response = self.provider.generate(prompt)
        result = self.parser.parse(raw_response)

        metadata = RuntimeMetadata(
            provider=self.provider_name,
            model=self.provider.model_name(),
            include_db_schema=self.prompt_builder.include_db_schema,
            include_gold_query=self.prompt_builder.include_gold_query,
        )

        return {
            "status": "success",
            **result.to_dict(),
            **metadata.to_dict(),
        }

    def annotate_entry(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        """
        Annotate one input entry with retries.

        A final failed result includes the last parser/provider error and the raw
        response when available, allowing problematic cases to be inspected.
        """
        last_error = ""
        last_raw_response = ""

        for attempt in range(1, self.max_retries + 1):
            try:
                return self._annotate_once(entry)
            except Exception as exc:
                last_error = str(exc)
                print(
                    f"    attempt {attempt}/{self.max_retries} failed: "
                    f"{last_error}"
                )
                if attempt < self.max_retries:
                    time.sleep(min(2 ** attempt, 8))

        return {
            "status": "failed",
            "code": "",
            "category": "",
            "reason": "",
            "applicable_errors": [],
            "primary_error": None,
            "annotator_provider": self.provider_name,
            "annotator_model": self.provider.model_name(),
            "include_db_schema": self.prompt_builder.include_db_schema,
            "include_gold_query": self.prompt_builder.include_gold_query,
            "error": last_error,
            "raw_response": last_raw_response,
        }

    def run(self, entries: List[Dict[str, Any]]) -> None:
        """Annotate all entries and append each completed record to output JSONL."""
        completed = (
            JsonlRepository.completed_ids(self.output_path)
            if self.resume
            else set()
        )

        for index, entry in enumerate(entries):
            item_id = self.item_id(entry, index)

            if item_id in completed:
                print(f"[{index + 1}/{len(entries)}] SKIP item={item_id}")
                continue

            print(f"[{index + 1}/{len(entries)}] Annotating item={item_id}")

            annotation = self.annotate_entry(entry)

            output_record = dict(entry)
            output_record["_annotation_item_id"] = item_id
            output_record["taxonomy_annotation"] = annotation

            JsonlRepository.append(self.output_path, output_record)
