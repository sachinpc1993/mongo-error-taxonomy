#!/usr/bin/env python3
"""Run individual or batch annotations through the HTTP API."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import requests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Submit annotation requests to the taxonomy HTTP API."
    )
    parser.add_argument(
        "mode",
        choices=("individual", "batch"),
        help="API operation to call.",
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Individual JSON file or batch JSONL/JSON file.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Path for the response. Batch output is JSONL when this ends in .jsonl.",
    )
    parser.add_argument(
        "--base-url",
        default="http://localhost:8000",
        help="API base URL (default: http://localhost:8000).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=300,
        help="Request timeout in seconds (default: 300).",
    )
    return parser.parse_args()


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise RuntimeError(f"Unable to read input file {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid JSON in {path}: {exc}") from exc


def call_individual(base_url: str, input_path: Path, timeout: float) -> dict[str, Any]:
    payload = read_json(input_path)
    if not isinstance(payload, dict):
        raise RuntimeError("Individual input must contain one JSON object.")

    response = requests.post(
        f"{base_url.rstrip('/')}/v1/annotate/individual",
        json=payload,
        timeout=timeout,
    )
    response.raise_for_status()
    result = response.json()
    if not isinstance(result, dict):
        raise RuntimeError("The individual API returned an unexpected response.")
    return result


def call_batch(base_url: str, input_path: Path, timeout: float) -> dict[str, Any]:
    try:
        with input_path.open("rb") as batch_file:
            response = requests.post(
                f"{base_url.rstrip('/')}/v1/annotate/batch",
                files={"file": (input_path.name, batch_file, "application/jsonl")},
                timeout=timeout,
            )
    except OSError as exc:
        raise RuntimeError(f"Unable to read input file {input_path}: {exc}") from exc

    response.raise_for_status()
    result = response.json()
    if not isinstance(result, dict):
        raise RuntimeError("The batch API returned an unexpected response.")
    return result


def write_result(result: dict[str, Any], output_path: Path | None, mode: str) -> None:
    if output_path is None:
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if mode == "batch" and output_path.suffix.lower() == ".jsonl":
        results = result.get("results")
        if not isinstance(results, list):
            raise RuntimeError("The batch response does not contain a results list.")
        output_path.write_text(
            "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in results),
            encoding="utf-8",
        )
        return

    output_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    try:
        if args.mode == "individual":
            result = call_individual(args.base_url, args.input, args.timeout)
        else:
            result = call_batch(args.base_url, args.input, args.timeout)
        write_result(result, args.output, args.mode)
    except (requests.RequestException, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
