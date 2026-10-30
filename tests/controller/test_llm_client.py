"""Tests for LLMClient implementations, MockLLMClient, and failure modes."""

import json
import pytest
from pydantic import BaseModel

from controller.llm_client import GenericHTTPClient, MockLLMClient, create_llm_client


class SimpleTestSchema(BaseModel):
    message: str
    count: int


def test_mock_llm_client_canned_responses():
    client = MockLLMClient(
        canned_responses=[
            SimpleTestSchema(message="hello", count=42),
            {"message": "world", "count": 99},
            '{"message": "json_str", "count": 100}',
        ]
    )

    r1 = client.generate_structured("prompt 1", schema=SimpleTestSchema)
    assert r1.message == "hello"
    assert r1.count == 42

    r2 = client.generate_structured("prompt 2", schema=SimpleTestSchema)
    assert r2.message == "world"
    assert r2.count == 99

    r3 = client.generate_structured("prompt 3", schema=SimpleTestSchema)
    assert r3.message == "json_str"
    assert r3.count == 100

    assert len(client.call_history) == 3


def test_mock_llm_client_timeout():
    client = MockLLMClient(simulate_timeout=True)
    with pytest.raises(TimeoutError, match="timed out"):
        client.generate("hello")

    with pytest.raises(TimeoutError, match="timed out"):
        client.generate_structured("hello", schema=SimpleTestSchema)


def test_mock_llm_client_malformed_json():
    client = MockLLMClient(simulate_malformed_json=True)
    with pytest.raises(json.JSONDecodeError):
        client.generate_structured("hello", schema=SimpleTestSchema)


def test_generic_http_client_missing_key():
    client = GenericHTTPClient(api_key=None)
    # With no key, generate should raise RuntimeError
    client.api_key = None
    with pytest.raises(RuntimeError, match="No LLM API key configured"):
        client.generate("hello")


def test_create_llm_client_defaults_to_mock_when_no_key(monkeypatch):
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    client = create_llm_client()
    assert isinstance(client, MockLLMClient)
