"""LLM client interface, configurable HTTP integration, and deterministic mock client."""

from __future__ import annotations

import json
import logging
import os
import re
import time
import warnings
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar
from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    """Standardized response from an LLM call."""
    content: str
    parsed: Optional[Any] = None
    raw: Optional[Dict[str, Any]] = None
    model: str = "mock"
    latency_sec: float = 0.0
    error: Optional[str] = None


class LLMClient:
    """Abstract interface for LLM generation with validated structured outputs."""

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        raise NotImplementedError

    def generate_structured(
        self,
        prompt: str,
        schema: Type[T],
        system_prompt: Optional[str] = None,
    ) -> T:
        raise NotImplementedError


class MockLLMClient(LLMClient):
    """Deterministic mock LLM client for tests and offline execution.
    
    Can be configured with a sequence of canned responses, a dynamic generator callback,
    or simulated network/schema failures.
    """

    def __init__(
        self,
        canned_responses: Optional[List[Any]] = None,
        generator_fn: Optional[Callable[[str, Optional[Type[BaseModel]]], Any]] = None,
        simulate_timeout: bool = False,
        simulate_error: Optional[str] = None,
        simulate_malformed_json: bool = False,
    ) -> None:
        self.canned_responses: List[Any] = list(canned_responses or [])
        self.generator_fn = generator_fn
        self.simulate_timeout = simulate_timeout
        self.simulate_error = simulate_error
        self.simulate_malformed_json = simulate_malformed_json
        self.call_history: List[Dict[str, Any]] = []

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        self.call_history.append({"prompt": prompt, "system_prompt": system_prompt})
        if self.simulate_timeout:
            raise TimeoutError("Simulated LLM call timed out after 30s.")
        if self.simulate_error:
            raise RuntimeError(f"Simulated LLM error: {self.simulate_error}")
        if self.simulate_malformed_json:
            return "{ unclosed invalid json string ... "

        if self.canned_responses:
            item = self.canned_responses.pop(0)
            if isinstance(item, BaseModel):
                return item.model_dump_json()
            if isinstance(item, dict):
                return json.dumps(item)
            return str(item)

        if self.generator_fn:
            res = self.generator_fn(prompt, None)
            if isinstance(res, BaseModel):
                return res.model_dump_json()
            if isinstance(res, dict):
                return json.dumps(res)
            return str(res)

        return '{"result": "mock default response"}'

    def generate_structured(
        self,
        prompt: str,
        schema: Type[T],
        system_prompt: Optional[str] = None,
    ) -> T:
        self.call_history.append({"prompt": prompt, "system_prompt": system_prompt, "schema": schema.__name__})
        if self.simulate_timeout:
            raise TimeoutError("Simulated LLM call timed out after 30s.")
        if self.simulate_error:
            raise RuntimeError(f"Simulated LLM error: {self.simulate_error}")
        if self.simulate_malformed_json:
            raise json.JSONDecodeError("Simulated malformed JSON", "{ invalid", 0)

        if self.canned_responses:
            item = self.canned_responses.pop(0)
            if isinstance(item, schema):
                return item
            if isinstance(item, dict):
                return schema.model_validate(item)
            if isinstance(item, str):
                return schema.model_validate_json(item)
            raise ValueError(f"Cannot cast canned response of type {type(item)} to {schema.__name__}")

        if self.generator_fn:
            item = self.generator_fn(prompt, schema)
            if isinstance(item, schema):
                return item
            if isinstance(item, dict):
                return schema.model_validate(item)
            if isinstance(item, str):
                return schema.model_validate_json(item)

        raise RuntimeError(f"MockLLMClient ran out of canned responses for schema {schema.__name__}")


class GenericHTTPClient(LLMClient):
    """Provider-agnostic HTTP REST client supporting OpenAI-compatible endpoints.
    
    Uses standard library urllib.request to avoid external dependency lock-in.
    Credentials must be provided via environment variables.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        model: Optional[str] = None,
        timeout_seconds: float = 30.0,
    ) -> None:
        self.api_key = api_key or os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY") or os.getenv("GEMINI_API_KEY")
        self.api_base = api_base or os.getenv("LLM_API_BASE", "https://api.openai.com/v1")
        self.model = model or os.getenv("LLM_MODEL", "gpt-4o-mini")
        self.timeout = timeout_seconds

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        if not self.api_key:
            raise RuntimeError(
                "No LLM API key configured. Set LLM_API_KEY (or OPENAI_API_KEY / GEMINI_API_KEY) "
                "environment variable or use MockLLMClient."
            )

        endpoint = f"{self.api_base.rstrip('/')}/chat/completions"
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.2,
        }
        data_bytes = json.dumps(payload).encode("utf-8")

        req = urllib.request.Request(
            endpoint,
            data=data_bytes,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                body = response.read().decode("utf-8")
                resp_json = json.loads(body)
                return resp_json["choices"][0]["message"]["content"]
        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"LLM API returned HTTP {e.code}: {error_body}") from e
        except urllib.error.URLError as e:
            raise ConnectionError(f"Failed to connect to LLM endpoint {endpoint}: {e.reason}") from e

    def generate_structured(
        self,
        prompt: str,
        schema: Type[T],
        system_prompt: Optional[str] = None,
    ) -> T:
        schema_json = json.dumps(schema.model_json_schema(), indent=2)
        augmented_system = (
            (system_prompt or "") + "\n\n"
            "You MUST return ONLY valid JSON matching the following JSON Schema. "
            "Do not include markdown codeblocks or explanatory text outside the JSON.\n"
            f"Schema:\n{schema_json}"
        ).strip()

        raw_output = self.generate(prompt=prompt, system_prompt=augmented_system)

        # Clean codeblock wrappers if present
        clean_text = raw_output.strip()
        if clean_text.startswith("```"):
            clean_text = re.sub(r"^```(?:json)?\s*", "", clean_text)
            clean_text = re.sub(r"\s*```$", "", clean_text)

        try:
            return schema.model_validate_json(clean_text)
        except (json.JSONDecodeError, ValidationError) as e:
            raise ValueError(f"Failed to validate LLM output against schema {schema.__name__}: {str(e)}\nRaw: {raw_output}") from e


_MOCK_FALLBACK_MESSAGE = (
    "!!! NO LLM API KEY CONFIGURED: falling back to MockLLMClient. Results are NOT from a real model. "
    "Set LLM_API_KEY (or OPENAI_API_KEY / GEMINI_API_KEY), or request provider='mock' explicitly to silence this."
)


def create_llm_client(provider: Optional[str] = None) -> LLMClient:
    """Factory creating an LLMClient instance based on environment and requested provider.
    
    If no API key is available or provider is explicitly 'mock', returns MockLLMClient.
    """
    prov = (provider or os.getenv("LLM_PROVIDER", "")).strip().lower()
    has_key = bool(os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY") or os.getenv("GEMINI_API_KEY"))

    if prov == "mock":
        return MockLLMClient()
    if not has_key:
        warnings.warn(_MOCK_FALLBACK_MESSAGE, RuntimeWarning, stacklevel=2)
        logger.warning(_MOCK_FALLBACK_MESSAGE)
        return MockLLMClient()
    return GenericHTTPClient()
