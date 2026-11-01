"""LLM provider strategy implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict

import boto3
import requests


class LLMProvider(ABC):
    """Strategy interface for invoking an annotation LLM."""

    @abstractmethod
    def generate(self, system_prompt: str) -> str:
        """Return the model's raw textual response."""
        raise NotImplementedError

    @abstractmethod
    def model_name(self) -> str:
        """Return the configured model identifier."""
        raise NotImplementedError


class OllamaProvider(LLMProvider):
    """Ollama-backed LLM provider."""

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize the provider from the ``ollama`` configuration block."""
        self.config = config

    def generate(self, system_prompt: str) -> str:
        """Invoke Ollama's chat endpoint and return its response content."""
        base_url = str(
            self.config.get("base_url", "http://localhost:11434")
        ).rstrip("/")

        payload = {
            "model": self.config["model"],
            "stream": False,
            "format": "json",
            "options": {
                "temperature": float(self.config.get("temperature", 0.0)),
            },
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": (
                        "Perform the annotation now. "
                        "Return only the requested JSON object."
                    ),
                },
            ],
        }

        response = requests.post(
            f"{base_url}/api/chat",
            json=payload,
            timeout=int(self.config.get("timeout_seconds", 180)),
        )
        response.raise_for_status()

        data = response.json()
        return data["message"]["content"]

    def model_name(self) -> str:
        """Return the configured Ollama model name."""
        return str(self.config["model"])


class BedrockProvider(LLMProvider):
    """AWS Bedrock Converse API provider."""

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize the provider and Bedrock Runtime client."""
        self.config = config
        self.client = boto3.client(
            "bedrock-runtime",
            region_name=self.config.get("region_name", "ap-southeast-2"),
        )

    def generate(self, system_prompt: str) -> str:
        """Invoke the Bedrock Converse API and combine text content blocks."""
        response = self.client.converse(
            modelId=self.config["model_id"],
            system=[{"text": system_prompt}],
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "text": (
                                "Perform the annotation now. "
                                "Return only the requested JSON object."
                            )
                        }
                    ],
                }
            ],
            inferenceConfig={
                "temperature": float(self.config.get("temperature", 0.0)),
                "maxTokens": int(self.config.get("max_tokens", 1200)),
            },
        )

        return "\n".join(
            block["text"]
            for block in response["output"]["message"]["content"]
            if "text" in block
        )

    def model_name(self) -> str:
        """Return the configured Bedrock model identifier."""
        return str(self.config["model_id"])


class LLMProviderFactory:
    """Factory for selecting a concrete LLM provider strategy."""

    @staticmethod
    def create(provider_name: str, config: Dict[str, Any]) -> LLMProvider:
        """Instantiate the provider requested by configuration."""
        normalized = provider_name.strip().lower()

        if normalized == "ollama":
            return OllamaProvider(config["ollama"])

        if normalized == "bedrock":
            return BedrockProvider(config["bedrock"])

        raise ValueError(f"Unsupported provider: {provider_name}")
