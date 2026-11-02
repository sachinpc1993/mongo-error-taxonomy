"""Domain models used by the taxonomy annotation runner."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass(frozen=True)
class ErrorAnnotation:
    """One validated, evidence-grounded query defect."""

    code: str
    location: str
    explanation: str

    def to_dict(self) -> Dict[str, str]:
        return {
            "code": self.code,
            "location": self.location,
            "explanation": self.explanation,
        }


@dataclass(frozen=True)
class AnnotationResult:
    """Validated list of zero or more query defects."""

    errors: list[ErrorAnnotation]

    def to_dict(self) -> Dict[str, Any]:
        return {"errors": [error.to_dict() for error in self.errors]}


@dataclass(frozen=True)
class RuntimeMetadata:
    """Reproducibility metadata describing a single annotation."""

    provider: str
    model: str
    generation_configuration: Dict[str, Any]
    taxonomy_version: str
    prompt_version: str
    include_db_schema: bool
    include_gold_query: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "annotator_provider": self.provider,
            "annotator_model": self.model,
            "generation_configuration": self.generation_configuration,
            "taxonomy_version": self.taxonomy_version,
            "prompt_version": self.prompt_version,
            "include_db_schema": self.include_db_schema,
            "include_gold_query": self.include_gold_query,
        }
