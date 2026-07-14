"""A thin OpenAI-compatible client so the rest of the pipeline is endpoint-agnostic.

Today this points at a local ``llama-server`` (llama.cpp). Tomorrow you can point
the same interface at ``llamafile`` or LM Studio by changing only host/port — the
task runner and evaluators never need to know.

Nothing here is Ollama. Local inference is llama.cpp's OpenAI-compatible server.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from openai import OpenAI


@dataclass
class ModelSpec:
    name: str
    display: str
    params_b: float
    port: int
    gguf: str
    host: str = "127.0.0.1"

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}/v1"


def load_registry(models_yaml: str | Path, host: str = "127.0.0.1") -> list[ModelSpec]:
    """Load the candidate models (not the judge) from models.yaml."""
    data = yaml.safe_load(Path(models_yaml).read_text())
    return [
        ModelSpec(
            name=m["name"],
            display=m["display"],
            params_b=float(m["params_b"]),
            port=int(m["port"]),
            gguf=m["gguf"],
            host=host,
        )
        for m in data.get("models", [])
    ]


class LocalLLM:
    """Minimal chat wrapper over an OpenAI-compatible endpoint.

    ``model`` is ignored by llama-server (it serves one model) but is passed
    through so the same class works against multi-model backends.
    """

    def __init__(self, base_url: str, api_key: str = "not-needed-for-local", model: str = "local"):
        self.base_url = base_url
        self.model = model
        self._client = OpenAI(base_url=base_url, api_key=api_key)

    def chat(self, prompt: str, *, system: str | None = None, **kwargs: Any) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=kwargs.pop("temperature", 0.0),
            **kwargs,
        )
        return (resp.choices[0].message.content or "").strip()

    def is_up(self) -> bool:
        """True if /v1/models responds — used by the startup probe."""
        try:
            self._client.models.list()
            return True
        except Exception:
            return False
