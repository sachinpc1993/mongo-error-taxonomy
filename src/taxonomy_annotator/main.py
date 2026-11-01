"""Command-line entry point for the taxonomy annotation runner."""

from __future__ import annotations

from pathlib import Path

from .config import ConfigLoader
from .io_utils import JsonlRepository
from .parser import AnnotationParser
from .prompt import PromptBuilder
from .providers import LLMProviderFactory
from .service import AnnotationService


def main() -> None:
    """Load configuration and execute a complete annotation run."""
    config = ConfigLoader.load("config.yaml")

    provider_name = str(config["provider"]).strip().lower()
    provider = LLMProviderFactory.create(provider_name, config)

    taxonomy_text = PromptBuilder.read_taxonomy(
        config.get("taxonomy_file", "taxonomy.txt")
    )

    prompt_builder = PromptBuilder(
        taxonomy_text=taxonomy_text,
        include_db_schema=bool(config.get("include_db_schema", True)),
        include_gold_query=bool(config.get("include_gold_query", True)),
    )

    input_dir = Path(config.get("input_dir", "input"))
    output_dir = Path(config.get("output_dir", "output"))
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / config.get(
        "output_filename", "annotated_output.jsonl"
    )

    entries = JsonlRepository.read_directory(input_dir)

    print(
        f"Loaded {len(entries)} input entries.\n"
        f"Provider: {provider_name}\n"
        f"Model: {provider.model_name()}\n"
        f"Include DB schema: {prompt_builder.include_db_schema}\n"
        f"Include gold query: {prompt_builder.include_gold_query}\n"
        f"Output: {output_path}\n"
    )

    service = AnnotationService(
        provider=provider,
        prompt_builder=prompt_builder,
        parser=AnnotationParser(),
        output_path=output_path,
        max_retries=int(config.get("max_retries", 3)),
        resume=bool(config.get("resume", True)),
        provider_name=provider_name,
    )

    service.run(entries)
    print(f"\nFinished. Output written to: {output_path}")


if __name__ == "__main__":
    main()
