"""Unified OpenAI-compatible LLM client.

All scripts that need to query a hosted LLM (EToM 7-factor generation, PFD
labeling, ORM outdated-memory judgment, etc.) go through this module so that
the endpoint, key, and model name live in a single config file.

Config file: config/llm_api.yaml (resolved relative to the project root).
The project root is the directory that contains the `config/` folder; it is
discovered by walking up from this file's location.

Usage:
    from src.llm_client import LLMClient
    client = LLMClient()
    text = client.chat("system prompt", "user prompt", role="gen")
"""

from __future__ import annotations

import os
import time
import logging
from pathlib import Path
from typing import Optional

import yaml
from openai import OpenAI, APIError, APIConnectionError, RateLimitError

log = logging.getLogger(__name__)


def _project_root() -> Path:
    here = Path(__file__).resolve().parent
    for p in [here, *here.parents]:
        if (p / "config" / "llm_api.yaml").is_file():
            return p
    raise FileNotFoundError(
        "Could not locate config/llm_api.yaml — run from inside code_new/."
    )


def load_config(config_path: Optional[str] = None) -> dict:
    """Load the YAML config. `config_path` overrides discovery when given."""
    if config_path is None:
        config_path = _project_root() / "config" / "llm_api.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class LLMClient:
    """Thin wrapper around an OpenAI-compatible chat endpoint.

    role="gen"   → uses cfg["model_gen"]   (default: Qwen2.5-72B-Instruct).
    role="judge" → uses cfg["model_judge"] (default: Qwen2.5-32B-Instruct).
    role="eval"  → uses cfg["model_eval"]  (fallback: model_judge).
    role="predict" → uses cfg["model_predict"] (fallback: model_gen).
    """

    def __init__(self, config_path: Optional[str] = None):
        self.cfg = load_config(config_path)
        self.client = OpenAI(
            base_url=self.cfg["base_url"],
            api_key=self.cfg.get("api_key", "EMPTY"),
            timeout=self.cfg.get("timeout", 120),
        )
        self.eval_client = None
        if self.cfg.get("eval_base_url") or self.cfg.get("eval_api_key"):
            self.eval_client = OpenAI(
                base_url=self.cfg.get("eval_base_url") or self.cfg["base_url"],
                api_key=self.cfg.get("eval_api_key") or self.cfg.get("api_key", "EMPTY"),
                timeout=self.cfg.get("timeout", 120),
            )
        self.predict_client = None
        if self.cfg.get("predict_base_url") or self.cfg.get("predict_api_key"):
            self.predict_client = OpenAI(
                base_url=self.cfg.get("predict_base_url") or self.cfg["base_url"],
                api_key=self.cfg.get("predict_api_key") or self.cfg.get("api_key", "EMPTY"),
                timeout=self.cfg.get("timeout", 120),
            )

    def _model_for(self, role: str) -> str:
        if role == "gen":
            return self.cfg["model_gen"]
        if role == "judge":
            return self.cfg["model_judge"]
        if role == "eval":
            return self.cfg.get("model_eval") or self.cfg["model_judge"]
        if role == "predict":
            return self.cfg.get("model_predict") or self.cfg["model_gen"]
        raise ValueError(f"unknown role: {role!r} (expected 'gen', 'judge', 'eval', or 'predict')")

    def chat(
        self,
        system: str,
        user: str,
        *,
        role: str = "gen",
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> str:
        model = self._model_for(role)
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        return self._complete(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            client_role=role,
        )

    def chat_messages(
        self,
        messages: list,
        *,
        role: str = "gen",
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> str:
        """Full message-list form for callers that need multi-turn context."""
        return self._complete(
            model=self._model_for(role),
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            client_role=role,
        )

    def _complete(
        self,
        *,
        model,
        messages,
        temperature,
        max_tokens,
        top_p,
        client_role: str = "gen",
    ) -> str:
        if client_role == "eval" and self.eval_client:
            client = self.eval_client
        elif client_role == "predict" and self.predict_client:
            client = self.predict_client
        else:
            client = self.client
        max_retries = int(self.cfg.get("max_retries", 3))
        backoff = float(self.cfg.get("retry_backoff", 2.0))
        last_err: Optional[Exception] = None
        for attempt in range(1, max_retries + 1):
            try:
                kwargs = {
                    "model": model,
                    "messages": messages,
                    "temperature": temperature
                    if temperature is not None
                    else self.cfg.get("temperature", 0.7),
                    "top_p": top_p if top_p is not None else self.cfg.get("top_p", 0.9),
                }
                if max_tokens is not None:
                    kwargs["max_tokens"] = max_tokens
                if client_role == "eval":
                    disable_thinking = self.cfg.get("eval_disable_thinking_extra_body", False)
                elif client_role == "predict":
                    disable_thinking = self.cfg.get(
                        "predict_disable_thinking_extra_body",
                        self.cfg.get("disable_thinking_extra_body", True),
                    )
                else:
                    disable_thinking = self.cfg.get("disable_thinking_extra_body", True)
                if disable_thinking:
                    kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
                resp = client.chat.completions.create(**kwargs)
                return resp.choices[0].message.content or ""
            except (APIConnectionError, RateLimitError, APIError) as e:
                last_err = e
                wait = backoff ** attempt
                log.warning(
                    "LLM call failed (attempt %d/%d): %s — retrying in %.1fs",
                    attempt, max_retries, e, wait,
                )
                time.sleep(wait)
        assert last_err is not None
        raise last_err


# Convenience singleton for scripts that just want a default client.
_default_client: Optional[LLMClient] = None


def default_client() -> LLMClient:
    global _default_client
    if _default_client is None:
        _default_client = LLMClient()
    return _default_client
