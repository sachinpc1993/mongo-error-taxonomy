# Text-to-NoSQL Taxonomy Annotator

A JSONL runner and HTTP API for zero-shot, taxonomy-guided annotation of
observable defects in predicted MongoDB aggregation pipelines. The runner uses
the configured Ollama or AWS Bedrock backend and does not execute or repair
queries.

## Configure and run

Install the project and choose a provider in `config.yaml`:

```bash
pip install -e .
taxonomy-annotator
```

The CLI reads `.jsonl` files from `input/` in lexical file order and appends
records to `output/annotated_output.jsonl`. By default, only entries with `EX`
equal to numeric or string zero are annotated. Records missing `EX`, and records
with nonzero `EX`, are skipped and counted in the run log. Set
`execution_success_only: false` to disable this filter.

Ollama and Bedrock settings, including temperature (zero by default), model
identifier, and token limits, are configured under their respective sections
in `config.yaml`. For Bedrock, use the standard AWS credential chain; do not
store credentials in the configuration.

## Input fields

Each JSONL line is a JSON object. The runner accepts these primary fields and
the indicated alternatives:

| Information | Supported field names |
|---|---|
| Question | `natural_language_query`, `question`, or `query` |
| Prediction | `generated_nosql_query`, `predicted_query`, or `prediction` |
| Schema | `db_schema`, `schema`, or `database_schema` (optional) |
| Reference query | `ground_truth_query`, `gold_nosql_query`, or `reference_query` (optional) |
| Execution exception | `execution_error` or `execution_error_message` (optional; missing values are sent as `null`) |
| Execution result | `EX` (required for the default EX=0 population filter) |

Schema objects are passed through as supplied, including collection, field,
type, and relationship metadata. The runner does not fabricate missing schema
or execution evidence. Original input fields—including dataset, question/query
IDs, model names, and approach identifiers—are preserved in the output.

This repository currently provides a generic JSONL input path; it does not
contain standalone DocSpider- or SM3-specific dataset adapters. Convert those
datasets to the supported fields above while preserving their identifiers and
benchmark outputs before running the annotator.

## Taxonomy and prompts

The taxonomy is loaded from `taxonomy.yaml`, separately from the system
prompt. It contains the 15 permitted leaf codes, definitions, disambiguation
rules, and taxonomy version. The same taxonomy and system prompt are used for
every item; the per-item evidence appears only in the user prompt. No labelled
examples are included.

The system prompt template is:

```text
You are an expert annotator of MongoDB aggregation pipeline errors.

Identify observable defects in the predicted query using the provided error taxonomy.

Instructions:
1. Assign the most specific applicable leaf-node category.
2. Annotate all distinct errors without double-counting the same defect.
3. Do not classify semantically equivalent query formulations as errors.
4. Use only the provided evidence; do not infer causes or unsupported errors.
5. Return only valid JSON.

ERROR TAXONOMY:
{taxonomy}
```

The user prompt template is:

```text
Annotate the predicted MongoDB query.

Natural-language question:
{question}

Database schema:
{schema}

Reference query:
{ground_truth_query}

Predicted query:
{predicted_query}

Execution error (if available):
{execution_error}

Return JSON with the following structure:
{
  "errors": [
    {
      "code": "EC...",
      "location": "...",
      "explanation": "..."
    }
  ]
}
```

## Output, validation, and resume

Successful output records preserve the original input and add
`_annotation_item_id` and `taxonomy_annotation`. The annotation value contains
`status: "success"`, an `errors` array, the raw response, provider/model,
generation configuration, taxonomy and prompt versions, and schema/reference
inclusion settings. An empty `errors` array is a validated finding of no
established defect—not a missing-evidence marker or API failure.

Every model attempt is also written as structured JSONL events to
`output/llm_interactions.jsonl` by default. Request events include a UTC
timestamp, item ID, provider/model settings, original input, and the exact
system and user prompts. Response events include the timestamp, attempt status,
duration, raw model response, and any provider or validation error. Configure
`llm_log_path` in `config.yaml` to change the log path. Logs can contain
sensitive input data and full prompts, so protect them accordingly.

Every error object must contain exactly:

```json
{
  "code": "EC2.1.2",
  "location": "pipeline[0:2]",
  "explanation": "The pipeline places valid stages in an order that changes the result."
}
```

The parser first tries to parse the complete model response as JSON. If that
fails, it tries the contents of fenced code blocks, including language-labelled
JSON fences; it does not extract unfenced JSON from surrounding commentary.
It rejects unknown taxonomy codes and invalid or duplicate error objects. After
configured retries are exhausted, the output is marked
`status: "failed"` with the error and last available raw response; this is
never converted to an empty annotation. Failed entries can be retried on a
subsequent run. With `resume: true`, previously successful entries—including
validated empty annotations—are skipped when their provider/model,
generation settings, taxonomy version, prompt version, and evidence settings
match the current run. Legacy-format results are not treated as completed
annotations.

## HTTP API

Start the API with:

```bash
uvicorn taxonomy_annotator.api:app --reload
```

Set `TAXONOMY_CONFIG_PATH` to select another configuration. The API provides
`POST /v1/annotate/individual` and `POST /v1/annotate/batch`; interactive docs
are available at `/docs`. The HTTP API's required field names are
`natural_language_query` and `generated_nosql_query`; schema and reference
fields are optional, and `execution_error_message` is optional.
