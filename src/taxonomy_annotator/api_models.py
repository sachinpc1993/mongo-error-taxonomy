"""Request and response models for the HTTP API."""

from __future__ import annotations

from typing import Any, Dict

from pydantic import BaseModel, Field


class AnnotationRequest(BaseModel):
    """Input fields required to classify one generated NoSQL query."""

    natural_language_query: str = Field(..., description="The user's natural-language query.")
    generated_nosql_query: str = Field(..., description="The generated NoSQL query.")
    execution_error_message: str = Field(..., description="The database execution error.")

    class Config:
        """Allow optional evidence and identifier fields used by the prompt builder."""

        extra = "allow"

    def to_entry(self) -> Dict[str, Any]:
        """Return the request as the service's dictionary-based domain input."""
        if hasattr(self, "model_dump"):
            return self.model_dump()
        return self.dict()


class AnnotationResponse(BaseModel):
    """Annotation result returned for an individual request."""

    input: Dict[str, Any]
    taxonomy_annotation: Dict[str, Any]


class BatchResponse(BaseModel):
    """Annotations returned for all records in a batch."""

    results: list[AnnotationResponse]
    count: int
