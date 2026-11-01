"""Dynamic system-prompt construction."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


class PromptBuilder:
    """Builds a complete annotation system prompt from one dataset entry."""

    OUTPUT_FORMAT = {
        "applicable_errors": [
            {
                "code": "<applicable leaf taxonomy code>",
                "category": "<applicable leaf taxonomy category>",
            }
        ],
        "primary_error": {
            "code": "<primary leaf taxonomy code>",
            "category": "<primary leaf taxonomy category>",
        },
        "reason": (
            "<concise explanation of why the primary error is the most "
            "directly responsible>"
        ),
        "code": "<same code as primary_error.code>",
        "category": "<same category as primary_error.category>",
    }

    def __init__(
        self,
        taxonomy_text: str,
        include_db_schema: bool,
        include_gold_query: bool,
    ) -> None:
        """Initialize the prompt builder with fixed experiment settings."""
        self.taxonomy_text = taxonomy_text.strip()
        self.include_db_schema = include_db_schema
        self.include_gold_query = include_gold_query

    @staticmethod
    def read_taxonomy(path: str | Path) -> str:
        """Read the taxonomy text verbatim from a text file."""
        taxonomy_path = Path(path)
        if not taxonomy_path.exists():
            raise FileNotFoundError(f"Taxonomy file not found: {taxonomy_path}")
        return taxonomy_path.read_text(encoding="utf-8").strip()

    @staticmethod
    def _render(value: Any) -> str:
        """Render strings or structured JSON values cleanly inside a prompt."""
        if value is None:
            return ""
        if isinstance(value, (dict, list)):
            return json.dumps(value, indent=2, ensure_ascii=False)
        return str(value).strip()

    def build(self, entry: Dict[str, Any]) -> str:
        """Construct the full system prompt for a single annotation item."""
        role = self._render(
            entry.get(
                "task_instruction",
                "You are an expert annotator for a Text-to-NoSQL error taxonomy. "
                "Identify all applicable errors and the single primary error "
                "in the generated query.",
            )
        )

        sections: List[str] = [
            f"[TASK INSTRUCTION]\n{role}",
            (
                "[Natural-language query]\n"
                + self._render(entry.get("natural_language_query"))
            ),
        ]

        if self.include_db_schema:
            sections.append(
                "[DB SCHEMA along with Collections]\n"
                + self._render(entry.get("db_schema"))
            )

        if self.include_gold_query:
            sections.append(
                "[Gold NoSQL query]\n"
                + self._render(entry.get("gold_nosql_query"))
            )

        sections.extend(
            [
                (
                    "[Generated NoSQL query]\n"
                    + self._render(entry.get("generated_nosql_query"))
                ),
                (
                    "[Execution error message]\n"
                    + self._render(entry.get("execution_error_message"))
                ),
                (
                    "[Taxonomy Instructions and class-wise descriptions]\n"
                    + self.taxonomy_text
                ),
                (
                    "[Classification instructions]\n"
                    "Select every applicable leaf-level taxonomy class and "
                    "subclass; do not stop after selecting the primary error. "
                    "Select exactly one primary error from the applicable "
                    "errors. Use only codes and category names defined in the "
                    "taxonomy. The primary error must be included in "
                    "applicable_errors. Explain the primary error in reason. "
                    "The taxonomy text may describe single-label annotation, "
                    "but for this task multi-label classification is required."
                ),
                (
                    "[Output format]\n"
                    "Return exactly one JSON object and nothing else:\n"
                    + json.dumps(self.OUTPUT_FORMAT, indent=2)
                ),
            ]
        )

        return "\n\n".join(sections).strip()
