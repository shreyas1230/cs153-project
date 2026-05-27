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
from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "cloudflare")

# Cloudflare Workers AI via AI Gateway (OpenAI-compat endpoint)
CF_ACCOUNT_ID = os.getenv("CF_ACCOUNT_ID", "")
CF_GATEWAY_ID = os.getenv("CF_GATEWAY_ID", "default")
CF_API_TOKEN = os.getenv("CF_API_TOKEN", "")
CF_GATEWAY_TOKEN = os.getenv("CF_GATEWAY_TOKEN", "")
CF_MODEL = os.getenv("CF_MODEL", "@cf/meta/llama-3.3-70b-instruct-fp8-fast")
CF_URL = (
    f"https://gateway.ai.cloudflare.com/v1/{CF_ACCOUNT_ID}/{CF_GATEWAY_ID}"
    f"/workers-ai/v1/chat/completions"
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
    # OpenAI-compat format
    payload: dict = {
        "model": CF_MODEL,
        "messages": [{"role": "system", "content": system}, *messages] if system else messages,
        "temperature": temperature,
        "max_tokens": 4096,
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}

    headers = {
        "Authorization": f"Bearer {CF_API_TOKEN}",
        "Content-Type": "application/json",
    }
    if CF_GATEWAY_TOKEN:
        headers["cf-aig-authorization"] = f"Bearer {CF_GATEWAY_TOKEN}"

    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(CF_URL, headers=headers, json=payload)
        resp.raise_for_status()
        data = resp.json()

    # OpenAI-compat response shape: choices[0].message.content
    raw = data.get("choices", [{}])[0].get("message", {}).get("content", "")
    if json_mode:
        return _extract_json(raw)
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


def _extract_json(text: str) -> str:
    """Extract and return the JSON object from a response that may have prose or code fences."""
    # Strip markdown code fences (```json ... ``` or ``` ... ```)
    import re
    stripped = re.sub(r"```(?:json)?\s*", "", text).strip()

    # Try parsing cleaned text directly
    try:
        json.loads(stripped)
        return stripped
    except json.JSONDecodeError:
        pass

    # Find first { and its matching closing }
    start = stripped.find("{")
    if start == -1:
        raise ValueError(f"No JSON object found in response: {text[:200]}")

    depth = 0
    in_string = False
    escape = False
    for i, ch in enumerate(stripped[start:], start):
        if escape:
            escape = False
            continue
        if ch == "\\" and in_string:
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if not in_string:
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = stripped[start: i + 1]
                    json.loads(candidate)  # raises if still invalid
                    return candidate

    raise ValueError(f"Could not extract valid JSON from response: {text[:200]}")


def _validate_json(text: str) -> None:
    """Raises ValueError if a JSON object cannot be extracted from text."""
    _extract_json(text)
