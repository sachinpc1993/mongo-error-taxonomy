"""FastAPI application for individual and batch taxonomy annotation."""

from __future__ import annotations

import json
import os
from functools import lru_cache
from typing import Any, Dict, List

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import ValidationError

from .api_models import (
    AnnotationRequest,
    AnnotationResponse,
    BatchResponse,
)
from .config import ConfigLoader
from .parser import AnnotationParser
from .prompt import PromptBuilder
from .providers import LLMProviderFactory
from .service import AnnotationService


app = FastAPI(
    title="Text-to-NoSQL Taxonomy Annotator",
    description="Classify generated NoSQL query errors using the configured taxonomy.",
    version="0.1.0",
)


def _build_service(config_path: str) -> AnnotationService:
    """Create the configured annotation service used by API requests."""
    config = ConfigLoader.load(config_path)
    provider_name = str(config["provider"]).strip().lower()
    provider = LLMProviderFactory.create(provider_name, config)
    taxonomy, taxonomy_version = PromptBuilder.read_taxonomy(
        config.get("taxonomy_file", "taxonomy.yaml")
    )
    prompt_builder = PromptBuilder(
        taxonomy=taxonomy,
        include_db_schema=bool(config.get("include_db_schema", True)),
        include_gold_query=bool(config.get("include_gold_query", True)),
        taxonomy_version=taxonomy_version,
    )
    return AnnotationService(
        provider=provider,
        prompt_builder=prompt_builder,
        parser=AnnotationParser(prompt_builder.taxonomy_codes),
        output_path=config.get("output_dir", "output")
        + "/"
        + config.get("output_filename", "annotated_output.jsonl"),
        max_retries=int(config.get("max_retries", 3)),
        resume=False,
        provider_name=provider_name,
        llm_log_path=config.get(
            "llm_log_path", "output/llm_interactions.jsonl"
        ),
    )


@lru_cache(maxsize=1)
def get_annotation_service() -> AnnotationService:
    """Return one lazily initialized service for the application process."""
    return _build_service(os.getenv("TAXONOMY_CONFIG_PATH", "config.yaml"))


def _validate_entries(records: Any) -> List[Dict[str, Any]]:
    """Validate a decoded JSON array and convert it to service input records."""
    if not isinstance(records, list):
        raise ValueError("The batch JSON must contain an array of objects.")

    entries: List[Dict[str, Any]] = []
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"Batch item {index} must be a JSON object.")
        try:
            entries.append(AnnotationRequest(**record).to_entry())
        except ValidationError as exc:
            raise ValueError(f"Batch item {index} is invalid: {exc}") from exc
    return entries


def _parse_batch_content(content: bytes) -> List[Dict[str, Any]]:
    """Parse either a JSON array or newline-delimited JSON batch file."""
    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError("The uploaded file must be UTF-8 encoded.") from exc

    try:
        decoded = json.loads(text)
    except json.JSONDecodeError:
        entries = []
        for line_number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at line {line_number}: {exc}") from exc
        return _validate_entries(entries)
    return _validate_entries(decoded)


def _annotate(entry: Dict[str, Any]) -> AnnotationResponse:
    """Annotate one validated entry and shape the public response."""
    return AnnotationResponse(
        input=entry,
        taxonomy_annotation=get_annotation_service().annotate_entry(entry),
    )


@app.post(
    "/v1/annotate/individual",
    response_model=AnnotationResponse,
    summary="Annotate one query error",
)
def annotate_individual(request: AnnotationRequest) -> AnnotationResponse:
    """Classify one query using only the required input and optional evidence."""
    return _annotate(request.to_entry())


@app.post(
    "/v1/annotate/batch",
    response_model=BatchResponse,
    summary="Annotate an uploaded JSON batch",
)
def annotate_batch(file: UploadFile = File(...)) -> BatchResponse:
    """Classify records from an uploaded JSON array or JSONL file."""
    try:
        entries = _parse_batch_content(file.file.read())
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    results = [_annotate(entry) for entry in entries]
    return BatchResponse(results=results, count=len(results))
