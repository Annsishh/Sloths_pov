"""Provider clients for the experiment: OpenAI, Anthropic, Gemini (+ offline mocks), and a recording wrapper.

Reuses controller.llm_client.GenericHTTPClient (prompt-embedded JSON schema + parsing + fence stripping) and
evaluation.clients.AnthropicClient unchanged; only adds what is missing: token usage for OpenAI, a Gemini client, and
per-call capture of raw responses. Nothing in src/ is modified.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Type, TypeVar

from pydantic import BaseModel

from controller.llm_client import GenericHTTPClient, LLMClient, MockLLMClient
from evaluation.clients import AnthropicClient, EvalConfigError
from evaluation.mock_policy import make_mock_client

T = TypeVar("T", bound=BaseModel)
PLACEHOLDER_MODEL = "SET_MODEL_ID"
MOCK_PROVIDERS = ("mock", "mock-faulty")


def _post_json(url: str, headers: Dict[str, str], payload: Dict[str, Any], timeout: float, name: str) -> Dict[str, Any]:
    req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), method="POST",
                                 headers={"Content-Type": "application/json", **headers})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"{name} API returned HTTP {e.code}: {e.read().decode('utf-8', errors='replace')[:500]}") from e
    except urllib.error.URLError as e:
        raise ConnectionError(f"Failed to reach {name} API: {e.reason}") from e


class OpenAIClient(GenericHTTPClient):
    """GenericHTTPClient plus token usage capture."""

    def __init__(self, api_key: str, model: str, temperature: Optional[float] = 0.2, timeout_seconds: float = 120.0) -> None:
        super().__init__(api_key=api_key, api_base="https://api.openai.com/v1", model=model, timeout_seconds=timeout_seconds)
        self.temperature = temperature
        self.last_usage: Dict[str, int] = {}

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        messages = ([{"role": "system", "content": system_prompt}] if system_prompt else []) + [{"role": "user", "content": prompt}]
        payload: Dict[str, Any] = {"model": self.model, "messages": messages}
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        body = _post_json(f"{self.api_base}/chat/completions", {"Authorization": f"Bearer {self.api_key}"}, payload, self.timeout, "OpenAI")
        u = body.get("usage", {})
        self.last_usage = {"input_tokens": int(u.get("prompt_tokens", 0)), "output_tokens": int(u.get("completion_tokens", 0))}
        return body["choices"][0]["message"]["content"] or ""


class GeminiClient(GenericHTTPClient):
    """Google Gemini generateContent REST client. Thinking tokens are counted as output tokens."""

    def __init__(self, api_key: str, model: str, temperature: Optional[float] = 0.2, timeout_seconds: float = 120.0) -> None:
        super().__init__(api_key=api_key, api_base="https://generativelanguage.googleapis.com/v1beta", model=model, timeout_seconds=timeout_seconds)
        self.temperature = temperature
        self.last_usage: Dict[str, int] = {}

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        payload: Dict[str, Any] = {"contents": [{"role": "user", "parts": [{"text": prompt}]}]}
        if system_prompt:
            payload["systemInstruction"] = {"parts": [{"text": system_prompt}]}
        if self.temperature is not None:
            payload["generationConfig"] = {"temperature": self.temperature}
        body = _post_json(f"{self.api_base}/models/{self.model}:generateContent", {"x-goog-api-key": self.api_key}, payload, self.timeout, "Gemini")
        u = body.get("usageMetadata", {})
        self.last_usage = {"input_tokens": int(u.get("promptTokenCount", 0)),
                           "output_tokens": int(u.get("candidatesTokenCount", 0)) + int(u.get("thoughtsTokenCount", 0))}
        cands = body.get("candidates") or []
        if not cands:
            raise RuntimeError(f"Gemini returned no candidates: {json.dumps(body.get('promptFeedback', body))[:300]}")
        return "".join(p.get("text", "") for p in cands[0].get("content", {}).get("parts", []))


def build_client(cfg: Dict[str, Any], allow_mock: bool = False) -> LLMClient:
    """Resolve one configured model to a client, or raise. A real provider can never silently become a mock."""
    prov = str(cfg.get("provider", "")).strip().lower()
    if prov in MOCK_PROVIDERS:
        if not allow_mock:
            raise EvalConfigError(f"provider '{prov}' requires --allow-mock: mock results are NOT a model evaluation.")
        return make_mock_client()[0] if prov == "mock" else MockLLMClient(simulate_malformed_json=True)
    model = cfg.get("model")
    if not model or model == PLACEHOLDER_MODEL:
        raise EvalConfigError(f"model '{cfg.get('id')}': set a real model id (got {model!r}).")
    env = cfg.get("api_key_env") or {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY", "gemini": "GEMINI_API_KEY"}.get(prov, "")
    key = os.getenv(env) if env else None
    if prov not in ("openai", "anthropic", "gemini"):
        raise EvalConfigError(f"Unknown provider '{prov}'. Use openai, anthropic or gemini.")
    if not key:
        raise EvalConfigError(f"model '{cfg.get('id')}': environment variable {env} is not set; refusing to fall back to a mock.")
    temp = cfg.get("temperature", 0.2)
    if prov == "openai":
        return OpenAIClient(key, model, temp)
    if prov == "gemini":
        return GeminiClient(key, model, temp)
    return AnthropicClient(api_key=key, model=model)


class RecordingClient(LLMClient):
    """Wraps a client and records every call: raw response, validity, error kind, usage, latency.

    Raw text is captured by tapping the inner client's `generate` (which `generate_structured` calls), so a response that
    FAILS schema validation is still kept. error_kind is 'schema' for parse/validation failures, 'api' for everything else.
    """

    def __init__(self, inner: LLMClient) -> None:
        self.inner = inner
        self.calls: List[Dict[str, Any]] = []
        self.last_usage: Dict[str, int] = {}
        self._open: Optional[Dict[str, Any]] = None
        orig = inner.generate

        def tap(prompt: str, system_prompt: Optional[str] = None) -> str:
            out = orig(prompt, system_prompt)
            if self._open is not None:
                self._open["raw_response"] = out
                self._open["usage"] = dict(getattr(inner, "last_usage", {}) or {})
            return out
        inner.generate = tap  # type: ignore[method-assign]

    def _call(self, schema_name: str, fn: Any, *args: Any) -> Any:
        rec: Dict[str, Any] = {"seq": len(self.calls), "schema": schema_name, "raw_response": None, "usage": {},
                               "valid": False, "error": None, "error_kind": None, "latency_sec": 0.0}
        self._open = rec
        t0 = time.perf_counter()
        try:
            out = fn(*args)
            rec["valid"] = True
            if rec["raw_response"] is None:        # mocks do not go through generate(); keep the parsed output instead
                rec["raw_response"] = out.model_dump_json() if isinstance(out, BaseModel) else str(out)
            return out
        except Exception as e:
            rec["error"] = f"{type(e).__name__}: {str(e)[:500]}"
            rec["error_kind"] = "schema" if isinstance(e, ValueError) else "api"
            raise
        finally:
            rec["latency_sec"] = round(time.perf_counter() - t0, 4)
            self.last_usage = dict(rec["usage"])
            self.calls.append(rec)
            self._open = None

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        return self._call("text", self.inner.generate, prompt, system_prompt)

    def generate_structured(self, prompt: str, schema: Type[T], system_prompt: Optional[str] = None) -> T:
        return self._call(schema.__name__, self.inner.generate_structured, prompt, schema, system_prompt)
