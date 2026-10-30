"""LLM client resolution for evaluation: a REAL evaluation can never silently run on the mock."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any, Dict, Optional

from controller.llm_client import GenericHTTPClient, LLMClient, MockLLMClient


class EvalConfigError(RuntimeError):
    """Raised when an evaluation is configured in a way that could mislead (e.g. mock posing as real)."""


class AnthropicClient(GenericHTTPClient):
    """Minimal Anthropic Messages API client (standard library only). Records token usage per call."""

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None, timeout_seconds: float = 120.0,
                 max_tokens: int = 4096) -> None:
        super().__init__(api_key=api_key or os.getenv("ANTHROPIC_API_KEY"), api_base="https://api.anthropic.com/v1",
                         model=model or os.getenv("LLM_MODEL"), timeout_seconds=timeout_seconds)
        self.max_tokens = max_tokens
        self.last_usage: Dict[str, int] = {}

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        if not self.api_key:
            raise EvalConfigError("ANTHROPIC_API_KEY is not set.")
        payload: Dict[str, Any] = {"model": self.model, "max_tokens": self.max_tokens, "temperature": 0.2,
                                   "messages": [{"role": "user", "content": prompt}]}
        if system_prompt:
            payload["system"] = system_prompt
        req = urllib.request.Request(
            f"{self.api_base}/messages", data=json.dumps(payload).encode("utf-8"), method="POST",
            headers={"Content-Type": "application/json", "x-api-key": self.api_key, "anthropic-version": "2023-06-01"},
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"Anthropic API returned HTTP {e.code}: {e.read().decode('utf-8', errors='replace')}") from e
        except urllib.error.URLError as e:
            raise ConnectionError(f"Failed to reach Anthropic API: {e.reason}") from e
        usage = body.get("usage", {})
        self.last_usage = {"input_tokens": int(usage.get("input_tokens", 0)), "output_tokens": int(usage.get("output_tokens", 0))}
        return "".join(block.get("text", "") for block in body.get("content", []) if block.get("type") == "text")


def resolve_client(provider: str, model: Optional[str] = None, allow_mock: bool = False) -> LLMClient:
    """Return the requested client, or raise. Real providers need a key; the mock needs an explicit opt-in."""
    prov = (provider or "").strip().lower()
    if prov == "mock":
        if not allow_mock:
            raise EvalConfigError("provider 'mock' requires --allow-mock: mock results are NOT a model evaluation.")
        return MockLLMClient()
    if prov == "anthropic":
        if not (os.getenv("ANTHROPIC_API_KEY")):
            raise EvalConfigError("provider 'anthropic' requires ANTHROPIC_API_KEY; refusing to fall back to the mock.")
        if not (model or os.getenv("LLM_MODEL")):
            raise EvalConfigError("provider 'anthropic' requires --model (or LLM_MODEL).")
        return AnthropicClient(model=model)
    if prov in ("openai", "openai-compatible"):
        key = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not key:
            raise EvalConfigError("provider 'openai' requires LLM_API_KEY or OPENAI_API_KEY; refusing to fall back to the mock.")
        if not (model or os.getenv("LLM_MODEL")):
            raise EvalConfigError("provider 'openai' requires --model (or LLM_MODEL).")
        return GenericHTTPClient(api_key=key, model=model)
    raise EvalConfigError(f"Unknown provider '{provider}'. Use anthropic, openai or mock (with --allow-mock).")


def assert_real(client: LLMClient, allow_mock: bool) -> None:
    """Last line of defence: a client that is (or wraps) the mock may only run when explicitly allowed."""
    inner = getattr(client, "inner", client)
    if isinstance(inner, MockLLMClient) and not allow_mock:
        raise EvalConfigError("The resolved client is a mock but mock use was not allowed; refusing to run.")
