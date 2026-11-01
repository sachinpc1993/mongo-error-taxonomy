# Text-to-NoSQL Error Taxonomy Annotator

A structured annotation runner for classifying Text-to-NoSQL errors using
either local Ollama models or AWS Bedrock models.

## Project structure

```text
text_to_nosql_taxonomy_annotator/
├── config.yaml
├── taxonomy.txt
├── pyproject.toml
├── README.md
├── input/
│   └── .gitkeep
├── output/
│   └── .gitkeep
└── src/
    └── taxonomy_annotator/
        ├── __init__.py
        ├── config.py
        ├── io_utils.py
        ├── main.py
        ├── models.py
        ├── parser.py
        ├── prompt.py
        ├── providers.py
        └── service.py
```

## Design

The implementation uses several lightweight design patterns:

- **Strategy pattern**: `LLMProvider` with `OllamaProvider` and `BedrockProvider`.
- **Factory pattern**: `LLMProviderFactory` selects the configured provider.
- **Repository pattern**: `JsonlRepository` encapsulates JSONL file access.
- **Builder pattern**: `PromptBuilder` dynamically constructs the system prompt.
- **Service layer**: `AnnotationService` coordinates retries, inference, parsing,
  resume behaviour, and output persistence.
- **Domain models**: dataclasses represent validated annotation output and
  runtime metadata.

## Input folder

Place one or more `.jsonl` files in:

```text
input/
```

The folder is intentionally empty in the distributed project.

Each line must contain one JSON object.

Required fields:

```json
{
  "natural_language_query": "...",
  "generated_nosql_query": "...",
  "execution_error_message": "..."
}
```

Optional evidence fields:

```json
{
  "id": "151",
  "task_instruction": "You are an expert annotator...",
  "db_schema": {
    "collections": {}
  },
  "gold_nosql_query": "db.collection.aggregate(...)"
}
```

If several `.jsonl` files exist in `input/`, all are processed in lexical
filename order.

## Evidence configuration

Edit `config.yaml`:

```yaml
include_db_schema: true
include_gold_query: true
```

This supports four experiment conditions:

| Schema | Gold |
|---|---|
| Yes | Yes |
| Yes | No |
| No | Yes |
| No | No |

## Ollama

```yaml
provider: ollama

ollama:
  model: llama3.1:8b
  base_url: http://localhost:11434
  temperature: 0.0
```

Ensure the model exists locally:

```bash
ollama pull llama3.1:8b
```

## AWS Bedrock

```yaml
provider: bedrock

bedrock:
  model_id: anthropic.claude-3-5-sonnet-20240620-v1:0
  region_name: ap-southeast-2
  temperature: 0.0
  max_tokens: 1200
```

AWS credentials must already be configured.

The application also loads a root-level `.env` file before creating the
Bedrock client. Copy `.env.example` to `.env` and provide valid credentials:

```bash
cp .env.example .env
```

For production, prefer an IAM role, AWS profile, or environment variables
managed by the deployment platform. Never commit `.env`.

## Install

From the project root:

```bash
pip install -e .
```

## Run

```bash
taxonomy-annotator
```

or:

```bash
python -m taxonomy_annotator.main
```

## HTTP API

Install the API dependencies with:

```bash
pip install -e .
```

Start the development server from the project root:

```bash
uvicorn taxonomy_annotator.api:app --reload
```

The API uses the same `config.yaml`, taxonomy file, and configured Ollama or
Bedrock provider as the command-line runner. Set `TAXONOMY_CONFIG_PATH` to use
a different configuration file.

### Individual annotation

`POST /v1/annotate/individual` accepts JSON containing the three required
fields. Optional fields such as `id`, `task_instruction`, `db_schema`, and
`gold_nosql_query` are passed through to the prompt builder.

```bash
curl -X POST http://localhost:8000/v1/annotate/individual \
  -H "Content-Type: application/json" \
  -d '{
    "natural_language_query": "In which years were cars produced weighing at least 3000?",
    "generated_nosql_query": "db.cars_data.find({\"Weight\": {\"$greaterThanOrEqual\": 3000}})",
    "execution_error_message": "unknown operator: $greaterThanOrEqual"
  }'
```

### Batch annotation

`POST /v1/annotate/batch` accepts a multipart upload named `file`. The file can
contain either a JSON array of objects or one JSON object per line (JSONL).
Each object must include the same three required fields as the individual
endpoint.

```bash
curl -X POST http://localhost:8000/v1/annotate/batch \
  -F "file=@input/test.jsonl"
```

Both endpoints return the original input and a `taxonomy_annotation` object
with all applicable leaf-level taxonomy errors, the primary error, its reason,
and model metadata. The legacy top-level `code` and `category` fields mirror
the primary error. Interactive OpenAPI documentation is available at
`http://localhost:8000/docs`.

### Python API runner

Use `scripts/api_runner.py` to call either endpoint and optionally save the
response:

```bash
python scripts/api_runner.py individual input/request.json \
  -o output/individual.json

python scripts/api_runner.py batch input/sample.jsonl \
  -o output/annotated_output.jsonl
```

Without `-o`, the response is printed to standard output. Batch output is
written as JSONL when the output filename ends in `.jsonl`; otherwise the full
batch response is written as JSON.

## Output

The default output is:

```text
output/annotated_output.jsonl
```

For 100 input JSON entries, the output contains 100 corresponding entries
assuming all entries are processed.

Each original JSON object is preserved and augmented with:

```json
{
  "_annotation_item_id": "151",
  "taxonomy_annotation": {
    "status": "success",
    "code": "EC2.2.1",
    "category": "Operator Selection",
    "reason": "The generated query uses an incorrect comparison operator.",
    "applicable_errors": [
      {
        "code": "EC2.2.1",
        "category": "Operator Selection"
      }
    ],
    "primary_error": {
      "code": "EC2.2.1",
      "category": "Operator Selection"
    },
    "annotator_provider": "ollama",
    "annotator_model": "llama3.1:8b",
    "include_db_schema": true,
    "include_gold_query": true
  }
}
```

## Robust LLM-output handling

`AnnotationParser` supports:

- raw JSON,
- fenced JSON such as ```` ```json ... ``` ````,
- JSON surrounded by explanatory text,
- extraction of the first balanced JSON object,
- Python-dictionary-style output using single quotes.

All parsed outputs are validated for:

- `applicable_errors`
- `primary_error`
- `reason`

Malformed responses are retried according to:

```yaml
max_retries: 3
```

## Resume support

With:

```yaml
resume: true
```

successful entries already written to the output JSONL are skipped if the
process is restarted after interruption.
