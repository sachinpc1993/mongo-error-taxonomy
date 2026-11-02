"""JSONL input/output utilities."""

from __future__ import annotations

import json
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Dict, List, Set


class JsonlRepository:
    """Repository abstraction for reading and appending JSONL records."""

    REQUIRED_FIELDS = {
        "natural_language_query",
        "generated_nosql_query",
    }

    @staticmethod
    def execution_success_population(
        entries: List[Dict[str, Any]],
    ) -> tuple[List[Dict[str, Any]], int, int]:
        """Return EX=0 records plus counts of records excluded for EX or missing EX."""
        eligible: List[Dict[str, Any]] = []
        skipped_nonzero = 0
        skipped_missing = 0
        for entry in entries:
            if "EX" not in entry or entry["EX"] is None or isinstance(entry["EX"], bool):
                skipped_missing += 1
                continue
            try:
                execution_result = Decimal(str(entry["EX"]).strip())
            except (InvalidOperation, ValueError):
                skipped_missing += 1
                continue
            if not execution_result.is_finite():
                skipped_missing += 1
                continue
            if execution_result == 0:
                eligible.append(entry)
            else:
                skipped_nonzero += 1
        return eligible, skipped_nonzero, skipped_missing

    @classmethod
    def read_directory(cls, directory: str | Path) -> List[Dict[str, Any]]:
        """
        Read every ``*.jsonl`` file in a directory in lexical filename order.

        Blank lines are ignored. Each non-empty line must contain one JSON object.
        """
        input_dir = Path(directory)

        if not input_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")

        files = sorted(input_dir.glob("*.jsonl"))
        if not files:
            raise FileNotFoundError(
                f"No .jsonl files found in input directory: {input_dir}"
            )

        records: List[Dict[str, Any]] = []
        for path in files:
            records.extend(cls.read_file(path))
        return records

    @classmethod
    def read_file(cls, path: str | Path) -> List[Dict[str, Any]]:
        """Read and validate one JSONL file."""
        source = Path(path)
        records: List[Dict[str, Any]] = []

        with source.open("r", encoding="utf-8-sig") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                line = raw_line.strip()
                if not line:
                    continue

                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSON in {source} at line {line_number}: {exc}"
                    ) from exc

                if not isinstance(record, dict):
                    raise ValueError(
                        f"{source} line {line_number} must contain a JSON object."
                    )

                missing = cls.REQUIRED_FIELDS.difference(record.keys())
                if "natural_language_query" not in record and any(
                    key in record for key in ("question", "query")
                ):
                    missing.discard("natural_language_query")
                if "generated_nosql_query" not in record and any(
                    key in record for key in ("predicted_query", "prediction")
                ):
                    missing.discard("generated_nosql_query")
                if missing:
                    raise ValueError(
                        f"{source} line {line_number} is missing fields: "
                        f"{sorted(missing)}"
                    )

                record.setdefault("_source_file", source.name)
                records.append(record)

        return records

    @staticmethod
    def append(path: str | Path, record: Dict[str, Any]) -> None:
        """Append one JSON object as one line to a JSONL file."""
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)

        with destination.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    @staticmethod
    def completed_ids(
        path: str | Path,
        expected_metadata: Dict[str, Any] | None = None,
    ) -> Set[str]:
        """
        Return IDs with successful results matching the current run configuration.

        Invalid/truncated output lines are ignored so interrupted runs can resume.
        """
        output = Path(path)
        completed: Set[str] = set()

        if not output.exists():
            return completed

        with output.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue

                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue

                annotation = record.get("taxonomy_annotation", {})
                if (
                    isinstance(annotation, dict)
                    and annotation.get("status") == "success"
                    and isinstance(annotation.get("errors"), list)
                    and record.get("_annotation_item_id") is not None
                    and all(
                        annotation.get(key) == value
                        for key, value in (expected_metadata or {}).items()
                    )
                ):
                    completed.add(str(record["_annotation_item_id"]))

        return completed
