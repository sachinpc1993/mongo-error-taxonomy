from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import Mock, patch

from taxonomy_annotator.io_utils import JsonlRepository
from taxonomy_annotator.parser import AnnotationParser
from taxonomy_annotator.prompt import PromptBuilder
from taxonomy_annotator.providers import BedrockProvider, OllamaProvider
from taxonomy_annotator.service import AnnotationService


ROOT = Path(__file__).resolve().parents[1]


class FakeProvider:
    def __init__(self, response: str) -> None:
        self.response = response
        self.calls = 0

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        self.calls += 1
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        return self.response

    def model_name(self) -> str:
        return "mock-model-v1"

    def generation_configuration(self) -> dict[str, Any]:
        return {"temperature": 0.0}


class AnnotationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.taxonomy, cls.taxonomy_version = PromptBuilder.read_taxonomy(
            ROOT / "taxonomy.yaml"
        )
        cls.builder = PromptBuilder(cls.taxonomy, taxonomy_version=cls.taxonomy_version)
        cls.parser = AnnotationParser(cls.builder.taxonomy_codes)

    def test_loads_all_15_categories(self) -> None:
        self.assertEqual(len(self.taxonomy), 15)
        self.assertEqual(len(self.builder.taxonomy_codes), 15)
        self.assertIn("EC3", self.builder.taxonomy_codes)

    def test_prompts_include_taxonomy_once_and_separate_evidence(self) -> None:
        self.assertEqual(self.builder.system_prompt.count("ERROR TAXONOMY:"), 1)
        self.assertIn("Disambiguation rule:", self.builder.system_prompt)
        self.assertNotIn("ERROR TAXONOMY:", self.builder.build_user_prompt({
            "question": "q",
            "prediction": "[]",
        }))
        self.assertIn("Execution error (if available):\nnull", self.builder.build_user_prompt({}))
        self.assertIn("Incorrect stage ordering belongs to EC2.1.2", self.builder.system_prompt)
        self.assertIn("otherwise valid expressions are incorrectly composed", self.builder.system_prompt)
        prompt = self.builder.system_prompt.lower()
        self.assertIn("valid but incorrect values belong to ec2.6", prompt)
        self.assertIn("incorrect implementation of the correct relationship belongs to ec2.5", prompt)
        self.assertIn("invalid argument type, arity, or structure", prompt)

    def test_parses_multiple_annotations_and_empty_result(self) -> None:
        result = self.parser.parse(json.dumps({
            "errors": [
                {"code": "EC2.1.2", "location": "pipeline[0:2]", "explanation": "Stages are reordered."},
                {"code": "EC2.6", "location": "pipeline[0].$limit", "explanation": "The limit conflicts with the request."},
            ]
        }))
        self.assertEqual([item.code for item in result.errors], ["EC2.1.2", "EC2.6"])
        self.assertEqual(self.parser.parse('{"errors": []}').errors, [])

    def test_parses_json_fences_only_after_full_response_parse_fails(self) -> None:
        fenced = '```json\n{"errors": []}\n```'
        with_commentary = (
            "Here is the result:\n```JSON\n"
            '{"errors": []}\n```\nDone.'
        )
        self.assertEqual(self.parser.parse(fenced).errors, [])
        self.assertEqual(self.parser.parse(with_commentary).errors, [])

    def test_rejects_invalid_json_fences_and_unfenced_json_in_commentary(self) -> None:
        with self.assertRaisesRegex(ValueError, "no valid JSON code fence"):
            self.parser.parse("```json\nnot json\n```")
        with self.assertRaisesRegex(ValueError, "no valid JSON code fence"):
            self.parser.parse('Result: {"errors": []}')

    def test_rejects_unknown_codes_and_duplicates(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown taxonomy code"):
            self.parser.parse('{"errors":[{"code":"EC9","location":"x","explanation":"y"}]}')
        duplicate = {
            "errors": [
                {"code": "EC2.6", "location": "x", "explanation": "y"},
                {"code": "EC2.6", "location": "x", "explanation": "y"},
            ]
        }
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            self.parser.parse(json.dumps(duplicate))

    def test_execution_success_filter(self) -> None:
        entries = [
            {"EX": 0},
            {"EX": "0.0"},
            {"EX": 1},
            {"EX": None},
            {"EX": "NaN"},
            {},
        ]
        selected, nonzero, missing = JsonlRepository.execution_success_population(entries)
        self.assertEqual(len(selected), 2)
        self.assertEqual((nonzero, missing), (1, 3))

    def test_jsonl_aliases_and_optional_execution_diagnostics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.jsonl"
            path.write_text(
                json.dumps({"question": "q", "prediction": "[]"}) + "\n",
                encoding="utf-8",
            )
            entries = JsonlRepository.read_file(path)
        self.assertEqual(entries[0]["question"], "q")
        self.assertNotIn("execution_error_message", entries[0])

    def test_existing_providers_receive_separate_system_and_user_prompts(self) -> None:
        ollama_response = Mock()
        ollama_response.json.return_value = {"message": {"content": '{"errors": []}'}}
        with patch("taxonomy_annotator.providers.requests.post", return_value=ollama_response) as post:
            ollama = OllamaProvider({"model": "local-model"})
            self.assertEqual(ollama.generate("system", "user"), '{"errors": []}')
        self.assertEqual(
            post.call_args.kwargs["json"]["messages"],
            [
                {"role": "system", "content": "system"},
                {"role": "user", "content": "user"},
            ],
        )

        bedrock_client = Mock()
        bedrock_client.converse.return_value = {
            "output": {"message": {"content": [{"text": '{"errors": []}'}]}}
        }
        with patch("taxonomy_annotator.providers.boto3.client", return_value=bedrock_client):
            bedrock = BedrockProvider({"model_id": "versioned-model"})
            self.assertEqual(bedrock.generate("system", "user"), '{"errors": []}')
        self.assertEqual(bedrock_client.converse.call_args.kwargs["system"], [{"text": "system"}])
        self.assertEqual(
            bedrock_client.converse.call_args.kwargs["messages"][0]["content"],
            [{"text": "user"}],
        )

    def test_successful_empty_annotation_resumes_and_preserves_input(self) -> None:
        provider = FakeProvider('{"errors": []}')
        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "annotations.jsonl"
            service = AnnotationService(
                provider=provider,
                prompt_builder=self.builder,
                parser=self.parser,
                output_path=output_path,
                resume=True,
                provider_name="mock",
            )
            input_record = {
                "dataset": "DocSpider",
                "question_id": "q-1",
                "approach": "baseline",
                "natural_language_query": "Question",
                "generated_nosql_query": "[]",
                "EX": 0,
            }
            service.run([input_record])
            service.run([input_record])
            self.assertEqual(provider.calls, 1)
            saved = json.loads(output_path.read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(saved["question_id"], "q-1")
            self.assertEqual(saved["dataset"], "DocSpider")
            self.assertEqual(saved["taxonomy_annotation"]["errors"], [])
            self.assertEqual(saved["taxonomy_annotation"]["taxonomy_version"], self.taxonomy_version)
            self.assertEqual(saved["taxonomy_annotation"]["raw_response"], '{"errors": []}')

    def test_logs_timestamped_model_request_and_response(self) -> None:
        provider = FakeProvider('{"errors": []}')
        with tempfile.TemporaryDirectory() as directory:
            log_path = Path(directory) / "logs" / "llm.jsonl"
            service = AnnotationService(
                provider=provider,
                prompt_builder=self.builder,
                parser=self.parser,
                output_path=Path(directory) / "annotations.jsonl",
                resume=False,
                provider_name="mock",
                llm_log_path=log_path,
            )
            entry = {
                "id": "request-1",
                "natural_language_query": "Question",
                "generated_nosql_query": "[]",
                "EX": 0,
            }

            service.annotate_entry(entry, item_id="request-1")

            events = [
                json.loads(line)
                for line in log_path.read_text(encoding="utf-8").splitlines()
            ]

        self.assertEqual([event["event"] for event in events], [
            "llm_request",
            "llm_response",
        ])
        request, response = events
        self.assertTrue(request["timestamp"].endswith("+00:00"))
        self.assertEqual(request["item_id"], "request-1")
        self.assertEqual(request["input"], entry)
        self.assertEqual(request["system_prompt"], self.builder.system_prompt)
        self.assertEqual(
            request["user_prompt"],
            self.builder.build_user_prompt(entry),
        )
        self.assertEqual(request["model"], "mock-model-v1")
        self.assertEqual(response["status"], "success")
        self.assertEqual(response["raw_response"], provider.response)
        self.assertEqual(request["request_id"], response["request_id"])

    def test_resume_does_not_skip_legacy_results(self) -> None:
        provider = FakeProvider('{"errors": []}')
        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "annotations.jsonl"
            legacy = {
                "_annotation_item_id": "old-1",
                "taxonomy_annotation": {"status": "success"},
            }
            output_path.write_text(json.dumps(legacy) + "\n", encoding="utf-8")
            service = AnnotationService(
                provider=provider,
                prompt_builder=self.builder,
                parser=self.parser,
                output_path=output_path,
                resume=True,
                provider_name="mock",
            )
            service.run([{"id": "old-1", "question": "q", "prediction": "[]"}])
        self.assertEqual(provider.calls, 1)

    def test_failure_is_not_an_empty_annotation_and_keeps_raw_response(self) -> None:
        provider = FakeProvider('{"errors":[{"code":"EC0","location":"x","explanation":"bad"}]}')
        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "annotations.jsonl"
            service = AnnotationService(
                provider=provider,
                prompt_builder=self.builder,
                parser=self.parser,
                output_path=output_path,
                max_retries=1,
                resume=True,
                provider_name="mock",
            )
            record = {"id": "bad-1", "question": "q", "prediction": "[]"}
            service.run([record])
            service.run([record])
            output = [
                json.loads(line)
                for line in output_path.read_text(encoding="utf-8").splitlines()
            ]
        self.assertEqual(provider.calls, 2)
        self.assertEqual(len(output), 2)
        failed = output[0]["taxonomy_annotation"]
        self.assertEqual(failed["status"], "failed")
        self.assertNotIn("errors", failed)
        self.assertEqual(failed["raw_response"], provider.response)


if __name__ == "__main__":
    unittest.main()
