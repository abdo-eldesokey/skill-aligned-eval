"""Shared OpenAI client helper for the project.

Provides an `instructor`-wrapped OpenAI client. Reads `OPENAI_API_KEY` from
the environment (also picked up from `.env` via `python-dotenv`).

Use `get_client(model_name)` for lazy/cached construction — this is the
right call site for app endpoints, where the key is only required when an
LLM-backed feature is actually invoked. Use `create_client(model_name)` for
scripts that always need the client and want to fail fast.
"""

from __future__ import annotations

import os
from functools import lru_cache

import instructor
from openai import OpenAI


def _build_client():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Copy .env.example to .env and fill it in."
        )
    return instructor.from_openai(OpenAI(api_key=api_key))


def create_client(model_name: str):
    """Eagerly construct the client. Raises if OPENAI_API_KEY is missing."""
    client = _build_client()
    print(f"Using OpenAI API with model: {model_name}")
    return client, model_name


@lru_cache(maxsize=1)
def get_client(model_name: str = ""):
    """Lazily construct the client on first call and cache it.

    Designed for app endpoints: import-time module loading does not require
    an API key; the key is only checked when an LLM call is actually made.
    """
    return _build_client()
