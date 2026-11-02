"""System and user prompt construction for taxonomy-guided annotation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import yaml


class PromptBuilder:
    """Build the consistent system prompt and per-query user prompt."""

    PROMPT_VERSION = "taxonomy-guided-v1"
    
    SYSTEM_TEMPLATE = """You are an expert annotator of MongoDB aggregation-pipeline errors.

                    Identify observable defects using the provided taxonomy.

                    Rules:
                    1. Assign the most specific leaf-node category to each distinct error.
                    2. Do not double-count errors or infer unsupported defects.
                    3. The reference query is not the only valid solution; differences alone are not errors.
                    4. If no defect can be established, return an empty errors array.
                    5. Return only valid JSON, without Markdown or additional text.

                    Output format:
                    {{
                    "errors": [
                        {{
                        "code": "EC...",
                        "location": "...",
                        "explanation": "..."
                        }}
                    ]
                    }}

                    Taxonomy:
                    {taxonomy}"""

    
    USER_TEMPLATE = """Natural-language question:
                    {question}

                    Database schema:
                    {schema}

                    Reference query:
                    {ground_truth_query}

                    Predicted query:
                    {predicted_query}

                    Execution error (if available):
                    {execution_error}"""


    def __init__(
        self,
        taxonomy: List[Dict[str, str]],
        include_db_schema: bool = True,
        include_gold_query: bool = True,
        taxonomy_version: str = "taxonomy-guided-v1",
    ) -> None:
        self.taxonomy = taxonomy
        self.include_db_schema = include_db_schema
        self.include_gold_query = include_gold_query
        self.taxonomy_version = taxonomy_version
        self.taxonomy_codes = {item["code"] for item in taxonomy}
        rendered_taxonomy = "\n".join(
            f'{item["code"]} {item["name"]}: {item["definition"]} '
            f'Disambiguation rule: {item["disambiguation_rule"]}'
            for item in taxonomy
        )
        self.system_prompt = self.SYSTEM_TEMPLATE.format(
            taxonomy=rendered_taxonomy
        )

    @staticmethod
    def read_taxonomy(path: str | Path) -> tuple[List[Dict[str, str]], str]:
        """Load and validate taxonomy categories and version from YAML."""
        taxonomy_path = Path(path)
        if not taxonomy_path.exists():
            raise FileNotFoundError(f"Taxonomy file not found: {taxonomy_path}")
        data = yaml.safe_load(taxonomy_path.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or not isinstance(data.get("categories"), list):
            raise ValueError(f"Invalid taxonomy configuration: {taxonomy_path}")

        categories = data["categories"]
        required = {"code", "name", "definition", "disambiguation_rule"}
        codes: set[str] = set()
        for index, item in enumerate(categories):
            if not isinstance(item, dict) or not required.issubset(item):
                raise ValueError(f"Taxonomy category {index} is incomplete.")
            if not all(
                isinstance(item[key], str) and item[key].strip()
                for key in required
            ):
                raise ValueError(f"Taxonomy category {index} contains an empty field.")
            if item["code"] in codes:
                raise ValueError(f"Duplicate taxonomy code: {item['code']}")
            codes.add(item["code"])
        if len(categories) != 15:
            raise ValueError(
                f"Expected 15 taxonomy categories, found {len(categories)}."
            )
        version = data.get("version")
        if not isinstance(version, str) or not version.strip():
            raise ValueError("Taxonomy configuration must specify a version.")
        return categories, version.strip()

    @staticmethod
    def _render(value: Any) -> str:
        if value is None:
            return "null"
        if isinstance(value, (dict, list)):
            return json.dumps(value, indent=2, ensure_ascii=False)
        return str(value).strip()

    @classmethod
    def _first_present(cls, entry: Dict[str, Any], *keys: str) -> Any:
        for key in keys:
            if key in entry:
                return entry[key]
        return None

    def build_user_prompt(self, entry: Dict[str, Any]) -> str:
        """Construct evidence for one query without duplicating the taxonomy."""
        question = self._first_present(
            entry, "natural_language_query", "question", "query"
        )
        schema = self._first_present(
            entry, "db_schema", "schema", "database_schema"
        )
        reference = self._first_present(
            entry, "ground_truth_query", "gold_nosql_query", "reference_query"
        )
        prediction = self._first_present(
            entry, "predicted_query", "generated_nosql_query", "prediction"
        )
        execution_error = self._first_present(
            entry, "execution_error", "execution_error_message"
        )

        if not self.include_db_schema:
            schema = None
        if not self.include_gold_query:
            reference = None
        return self.USER_TEMPLATE.format(
            question=self._render(question),
            schema=self._render(schema),
            ground_truth_query=self._render(reference),
            predicted_query=self._render(prediction),
            execution_error=self._render(execution_error),
        )
