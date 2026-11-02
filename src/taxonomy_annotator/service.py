"""Application service coordinating prompts, retries, model calls, and output."""

from __future__ import annotations

import json
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List
from uuid import uuid4

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
        llm_log_path: str | Path | None = None,
    ) -> None:
        """Initialize the service with injected strategies and runtime settings."""
        self.provider = provider
        self.prompt_builder = prompt_builder
        self.parser = parser
        self.output_path = Path(output_path)
        self.max_retries = max(1, int(max_retries))
        self.resume = resume
        self.provider_name = provider_name
        self.llm_log_path = Path(llm_log_path) if llm_log_path is not None else None
        self._llm_log_lock = threading.Lock()

    @staticmethod
    def item_id(entry: Dict[str, Any], index: int) -> str:
        """Resolve an ID from benchmark identifiers or the stable input row."""
        for key in (
            "_annotation_item_id", "query_id", "question_id", "query_idx", "id"
        ):
            if entry.get(key) is not None:
                parts = [
                    str(entry[name])
                    for name in (
                        "dataset",
                        "dataset_name",
                        "model",
                        "model_name",
                        "approach",
                        "approach_id",
                        key,
                    )
                    if entry.get(name) is not None
                ]
                return ":".join(parts)
        return str(index + 1)

    def _metadata(self) -> Dict[str, Any]:
        metadata = RuntimeMetadata(
            provider=self.provider_name,
            model=self.provider.model_name(),
            generation_configuration=self.provider.generation_configuration(),
            taxonomy_version=self.prompt_builder.taxonomy_version,
            prompt_version=self.prompt_builder.PROMPT_VERSION,
            include_db_schema=self.prompt_builder.include_db_schema,
            include_gold_query=self.prompt_builder.include_gold_query,
        )
        return metadata.to_dict()

    def _append_llm_log(self, record: Dict[str, Any]) -> None:
        if self.llm_log_path is None:
            return

        log_record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **record,
        }
        self.llm_log_path.parent.mkdir(parents=True, exist_ok=True)
        serialized = json.dumps(log_record, ensure_ascii=False) + "\n"
        with self._llm_log_lock:
            with self.llm_log_path.open("a", encoding="utf-8") as log_file:
                log_file.write(serialized)

    def annotate_entry(
        self,
        entry: Dict[str, Any],
        item_id: str | None = None,
    ) -> Dict[str, Any]:
        """
        Annotate one input entry with retries.

        A final failed result includes the last parser/provider error and the raw
        response when available, allowing problematic cases to be inspected.
        """
        last_error = ""
        last_raw_response = ""
        system_prompt = self.prompt_builder.system_prompt
        user_prompt = self.prompt_builder.build_user_prompt(entry)
        request_details = {
            "item_id": item_id or self.item_id(entry, 0),
            "provider": self.provider_name,
            "model": self.provider.model_name(),
            "generation_configuration": self.provider.generation_configuration(),
            "input": entry,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
        }

        for attempt in range(1, self.max_retries + 1):
            raw_response = ""
            request_id = uuid4().hex
            self._append_llm_log({
                "event": "llm_request",
                "request_id": request_id,
                **request_details,
                "attempt": attempt,
            })
            started_at = time.monotonic()
            try:
                raw_response = self.provider.generate(
                    system_prompt,
                    user_prompt,
                )
                last_raw_response = raw_response
                result = self.parser.parse(raw_response)
                self._append_llm_log({
                    "event": "llm_response",
                    "request_id": request_id,
                    "item_id": request_details["item_id"],
                    "provider": request_details["provider"],
                    "model": request_details["model"],
                    "attempt": attempt,
                    "status": "success",
                    "duration_seconds": time.monotonic() - started_at,
                    "raw_response": raw_response,
                })
                return {
                    "status": "success",
                    **result.to_dict(),
                    **self._metadata(),
                    "raw_response": raw_response,
                }
            except Exception as exc:
                last_error = str(exc)
                if raw_response:
                    last_raw_response = raw_response
                self._append_llm_log({
                    "event": "llm_response",
                    "request_id": request_id,
                    "item_id": request_details["item_id"],
                    "provider": request_details["provider"],
                    "model": request_details["model"],
                    "attempt": attempt,
                    "status": "failed",
                    "duration_seconds": time.monotonic() - started_at,
                    "raw_response": raw_response,
                    "error": last_error,
                })
                print(
                    f"    attempt {attempt}/{self.max_retries} failed: "
                    f"{last_error}"
                )
                if attempt < self.max_retries:
                    time.sleep(min(2 ** attempt, 8))

        return {
            "status": "failed",
            **self._metadata(),
            "error": last_error,
            "raw_response": last_raw_response,
        }

    def run(self, entries: List[Dict[str, Any]]) -> None:
        """Annotate all entries and append each completed record to output JSONL."""
        completed = (
            JsonlRepository.completed_ids(self.output_path, self._metadata())
            if self.resume
            else set()
        )

        for index, entry in enumerate(entries):
            item_id = self.item_id(entry, index)

            if item_id in completed:
                print(f"[{index + 1}/{len(entries)}] SKIP item={item_id}")
                continue

            print(f"[{index + 1}/{len(entries)}] Annotating item={item_id}")

            annotation = self.annotate_entry(entry, item_id=item_id)

            output_record = dict(entry)
            output_record["_annotation_item_id"] = item_id
            output_record["taxonomy_annotation"] = annotation

            JsonlRepository.append(self.output_path, output_record)
