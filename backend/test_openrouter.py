"""Quick smoke test for OpenRouter integration. Sends a tiny prompt."""
import asyncio
import os
from dotenv import load_dotenv
load_dotenv()

import httpx

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.3-70b-instruct")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


async def test():
    payload = {
        "model": OPENROUTER_MODEL,
        "messages": [
            {"role": "user", "content": 'Reply with exactly this JSON and nothing else: {"ok": true}'}
        ],
        "temperature": 0.0,
        "max_tokens": 20,
    }
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "X-Title": "CS153 Claim Conflict Detector",
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(OPENROUTER_URL, headers=headers, json=payload)
        resp.raise_for_status()
        data = resp.json()

    content = data.get("choices", [{}])[0].get("message", {}).get("content")
    cost = data.get("usage", {}).get("cost", "unknown")
    print(f"Response : {content}")
    print(f"Cost     : ${cost}")
    print(f"Tokens   : {data.get('usage', {}).get('total_tokens')} total")
    print(f"Provider : {data.get('provider')}")
    print("PASS" if content else "FAIL — content is None")


asyncio.run(test())
