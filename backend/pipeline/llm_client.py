"""
Single abstraction over LLM providers.
Set LLM_PROVIDER=cloudflare (default, free via course credits)
     LLM_PROVIDER=anthropic  (claude-sonnet-4-6, ~$10-20 total if needed)
"""
from __future__ import annotations
import os
import json
import asyncio
import httpx

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "cloudflare")

# Cloudflare Workers AI via AI Gateway
CF_ACCOUNT_ID = os.getenv("CF_ACCOUNT_ID", "")
CF_GATEWAY_ID = os.getenv("CF_GATEWAY_ID", "")
CF_API_TOKEN = os.getenv("CF_API_TOKEN", "")
CF_MODEL = os.getenv("CF_MODEL", "@cf/meta/llama-3.3-70b-instruct")
CF_URL = (
    f"https://gateway.ai.cloudflare.com/v1/{CF_ACCOUNT_ID}/{CF_GATEWAY_ID}"
    f"/workers-ai/{CF_MODEL}"
)

# Anthropic (fallback)
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")


async def chat(
    messages: list[dict],
    system: str = "",
    json_mode: bool = True,
    temperature: float = 0.0,
    max_retries: int = 3,
) -> str:
    for attempt in range(max_retries):
        try:
            if LLM_PROVIDER == "anthropic":
                return await _chat_anthropic(messages, system, json_mode, temperature)
            else:
                return await _chat_cloudflare(messages, system, json_mode, temperature)
        except (ValueError, json.JSONDecodeError) as e:
            if attempt == max_retries - 1:
                raise
            await asyncio.sleep(2 ** attempt)
    raise RuntimeError("LLM call failed after retries")


async def _chat_cloudflare(
    messages: list[dict],
    system: str,
    json_mode: bool,
    temperature: float,
) -> str:
    payload: dict = {
        "messages": [{"role": "system", "content": system}, *messages] if system else messages,
        "temperature": temperature,
        "max_tokens": 4096,
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}

    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(
            CF_URL,
            headers={
                "Authorization": f"Bearer {CF_API_TOKEN}",
                "Content-Type": "application/json",
            },
            json=payload,
        )
        resp.raise_for_status()
        data = resp.json()

    raw = data.get("result", {}).get("response", "")
    if json_mode:
        _validate_json(raw)
    return raw


async def _chat_anthropic(
    messages: list[dict],
    system: str,
    json_mode: bool,
    temperature: float,
) -> str:
    import anthropic  # lazy import — only needed when provider=anthropic

    client = anthropic.AsyncAnthropic(api_key=ANTHROPIC_API_KEY)
    kwargs: dict = {
        "model": ANTHROPIC_MODEL,
        "max_tokens": 4096,
        "temperature": temperature,
        "messages": messages,
    }
    if system:
        kwargs["system"] = system

    response = await client.messages.create(**kwargs)
    raw = response.content[0].text
    if json_mode:
        _validate_json(raw)
    return raw


def _validate_json(text: str) -> None:
    """Raises ValueError if text is not valid JSON. Tries to extract a JSON block first."""
    try:
        json.loads(text)
        return
    except json.JSONDecodeError:
        pass
    # try to extract last {...} block (some models wrap JSON in prose)
    start = text.rfind("{")
    end = text.rfind("}") + 1
    if start != -1 and end > start:
        json.loads(text[start:end])
        return
    raise ValueError(f"Response is not valid JSON: {text[:200]}")
