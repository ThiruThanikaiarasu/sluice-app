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
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from openai import OpenAI

DEFAULT_BASE_URL = "https://api.tensormux.com/v1"
DEFAULT_MODEL = "glm-4-7-flash"


@lru_cache(maxsize=1)
def client() -> "OpenAI":
    # Imported here, not at module top level: neatlogs instruments OpenAI by
    # patching it at import time, so anything that imports the client before
    # neatlogs.init() runs gets an unpatched client and silently produces no
    # trace. Deferring the import into this lru_cache'd function means the
    # import happens on first real use rather than at `import src.llm` time,
    # so init() only has to run before the first call, not before every other
    # module that might transitively import llm.py.
    from openai import OpenAI

    api_key = os.environ.get("SLUICE_LLM_API_KEY")
    if not api_key:
        raise RuntimeError(
            "SLUICE_LLM_API_KEY is unset. Get a key from app.tensormux.com."
        )
    return OpenAI(
        api_key=api_key,
        base_url=os.environ.get("SLUICE_LLM_BASE_URL", DEFAULT_BASE_URL),
    )


def complete(system: str, user: str, *, max_tokens: int = 4096) -> str:
    """Single-turn completion.

    Temperature is pinned at zero and not exposed. Treasury output that differs
    between identical runs cannot be audited, and an auditor asking why the
    plan changed is not a conversation worth having.

    GLM-4.7-Flash is a reasoning model: it fills a `reasoning` field before
    `message.content`, burning ~450-500 completion tokens before any content
    appears. A low max_tokens truncates the reasoning and leaves content
    empty -- indistinguishable from a real empty answer unless we check
    finish_reason explicitly.
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
    choice = response.choices[0]
    content = choice.message.content or ""
    if choice.finish_reason == "length" or not content:
        raise RuntimeError(
            f"LLM response truncated or empty (finish_reason={choice.finish_reason!r}, "
            f"max_tokens={max_tokens}). GLM-4.7-Flash burns ~450-500 tokens on "
            "reasoning before content -- raise max_tokens if this recurs."
        )
    return content
