"""Inference client.

Defaults to TensorMux, the hackathon's inference partner, over their
OpenAI-compatible endpoint. Override by environment so the provider can change
without touching call sites:

    SLUICE_LLM_API_KEY     required (TensorMux keys start with tmx_)
    SLUICE_LLM_BASE_URL    default https://api.tensormux.com/v1
    SLUICE_LLM_MODEL       default glm-4-7-flash

GLM-4.7-Flash has a 32k context window, which is small enough to shape the
design. Never pass the raw projection -- 84 forecast rows, 30 intercompany
agreements and the full covenant set crowd the window and measurably degrade
the reasoning. Pass `positions.summarise()` and only the constraints that bind.
"""

from __future__ import annotations

import os
from functools import lru_cache

from openai import OpenAI

DEFAULT_BASE_URL = "https://api.tensormux.com/v1"
DEFAULT_MODEL = "glm-4-7-flash"


@lru_cache(maxsize=1)
def client() -> OpenAI:
    api_key = os.environ.get("SLUICE_LLM_API_KEY")
    if not api_key:
        raise RuntimeError(
            "SLUICE_LLM_API_KEY is unset. Get a key from app.tensormux.com."
        )
    return OpenAI(
        api_key=api_key,
        base_url=os.environ.get("SLUICE_LLM_BASE_URL", DEFAULT_BASE_URL),
    )


def complete(system: str, user: str, *, max_tokens: int = 2048) -> str:
    """Single-turn completion.

    Temperature is pinned at zero and not exposed. Treasury output that differs
    between identical runs cannot be audited, and an auditor asking why the
    plan changed is not a conversation worth having.
    """
    response = client().chat.completions.create(
        model=os.environ.get("SLUICE_LLM_MODEL", DEFAULT_MODEL),
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.0,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content or ""
