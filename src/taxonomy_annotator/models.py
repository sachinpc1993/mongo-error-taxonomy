"""Domain models used by the taxonomy annotation runner."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass(frozen=True)
class ErrorClassification:
    """One applicable leaf-level taxonomy classification."""

    code: str
    category: str

    def to_dict(self) -> Dict[str, str]:
        """Serialize the classification into a JSON-compatible dictionary."""
        return {"code": self.code, "category": self.category}


@dataclass(frozen=True)
class AnnotationResult:
    """Validated taxonomy annotation produced by an LLM."""

    code: str
    category: str
    reason: str
    applicable_errors: list[ErrorClassification]
    primary_error: ErrorClassification

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the annotation into a JSON-compatible dictionary."""
        return {
            "code": self.code,
            "category": self.category,
            "reason": self.reason,
            "applicable_errors": [
                error.to_dict() for error in self.applicable_errors
            ],
            "primary_error": self.primary_error.to_dict(),
        }


@dataclass(frozen=True)
class RuntimeMetadata:
    """Metadata describing the model and evidence configuration for a run."""

    provider: str
    model: str
    include_db_schema: bool
    include_gold_query: bool

    def to_dict(self) -> Dict[str, Any]:
        """Serialize runtime metadata into a JSON-compatible dictionary."""
        return {
            "annotator_provider": self.provider,
            "annotator_model": self.model,
            "include_db_schema": self.include_db_schema,
            "include_gold_query": self.include_gold_query,
        }
