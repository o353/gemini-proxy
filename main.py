import asyncio
import hmac
import json
import os
import time
import uuid
from typing import Any

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import StreamingResponse
from google import genai
from google.genai import types
from pydantic import BaseModel, ConfigDict, Field

APP_NAME = "Gemini Janitor Proxy"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
PROXY_API_KEY = os.getenv("PROXY_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3-flash-preview")

app = FastAPI(title=APP_NAME, version="1.0.0")
client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)

    model: str = "gemini-3-flash-preview"
    messages: list[dict[str, Any]] = Field(default_factory=list)
    temperature: float | None = None
    max_tokens: int | None = None
    max_completion_tokens: int | None = None
    top_p: float | None = None
    stream: bool = False


def content_to_text(content: Any) -> str:
    """Extract text from common OpenAI message content formats."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                if item.get("type") == "text" and isinstance(item.get("text"), str):
                    parts.append(item["text"])
                elif isinstance(item.get("text"), str):
                    parts.append(item["text"])
        return "\n".join(parts)
    return "" if content is None else str(content)


def convert_messages(messages: list[dict[str, Any]]):
    system_parts: list[str] = []
    contents: list[types.Content] = []

    for message in messages:
        role = str(message.get("role", "user")).lower()
        text = content_to_text(message.get("content", ""))

        if role in ("system", "developer"):
            if text.strip():
                system_parts.append(text)
            continue

        if role == "assistant":
            gemini_role = "model"
        elif role in ("user", "tool", "function"):
            gemini_role = "user"
        else:
            continue

        if not text.strip():
            continue

        # Gemini expects turns to alternate; merge adjacent turns with the same role.
        if contents and contents[-1].role == gemini_role:
            contents[-1].parts.append(types.Part.from_text(text=text))
        else:
            contents.append(
                types.Content(
                    role=gemini_role,
                    parts=[types.Part.from_text(text=text)],
                )
            )

    if not contents or not any(item.role == "user" for item in contents):
        raise HTTPException(status_code=400, detail="Provide at least one non-empty user message.")

    return "\n\n".join(system_parts) or None, contents


def make_config(request: ChatRequest, system_instruction: str | None):
    # These settings are enforced by the proxy and cannot be disabled by the caller.
    options: dict[str, Any] = {
        "thinking_config": types.ThinkingConfig(thinking_level="high"),
        "tools": [types.Tool(google_search=types.GoogleSearch())],
    }

    if system_instruction:
        options["system_instruction"] = system_instruction
    if request.temperature is not None:
        options["temperature"] = request.temperature
    token_limit = request.max_completion_tokens or request.max_tokens
    if token_limit is not None:
        options["max_output_tokens"] = token_limit
    if request.top_p is not None:
        options["top_p"] = request.top_p

    return types.GenerateContentConfig(**options)


def authenticate(authorization: str | None) -> None:
    if not PROXY_API_KEY:
        raise HTTPException(status_code=503, detail="Proxy API key is not configured.")
    expected = f"Bearer {PROXY_API_KEY}"
    if not authorization or not hmac.compare_digest(authorization, expected):
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if client is None:
        raise HTTPException(status_code=503, detail="Gemini API is not configured.")


def usage_from_response(response) -> dict[str, int]:
    usage = getattr(response, "usage_metadata", None)
    prompt = int(getattr(usage, "prompt_token_count", 0) or 0)
    completion = int(getattr(usage, "candidates_token_count", 0) or 0)
    total = int(getattr(usage, "total_token_count", 0) or (prompt + completion))
    return {"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": total}


@app.get("/")
async def health():
    return {"status": "ok", "service": APP_NAME, "model": GEMINI_MODEL}


@app.get("/v1/models")
async def list_models(authorization: str | None = Header(default=None)):
    authenticate(authorization)
    return {
        "object": "list",
        "data": [{"id": GEMINI_MODEL, "object": "model", "created": int(time.time()), "owned_by": "google"}],
    }


@app.post("/v1/chat/completions")
async def chat_completions(
    request: ChatRequest,
    authorization: str | None = Header(default=None),
):
    authenticate(authorization)
    system_instruction, contents = convert_messages(request.messages)
    config = make_config(request, system_instruction)

    try:
        response = await asyncio.to_thread(
            client.models.generate_content,
            model=GEMINI_MODEL,
            contents=contents,
            config=config,
        )
        answer = response.text or ""
    except Exception as exc:
        # Do not log API keys or the full request body.
        print(f"Gemini request failed ({type(exc).__name__}).")
        raise HTTPException(
            status_code=502,
            detail="Gemini request failed. Check service logs and model availability.",
        ) from exc

    completion_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())
    usage = usage_from_response(response)

    if not request.stream:
        return {
            "id": completion_id,
            "object": "chat.completion",
            "created": created,
            "model": GEMINI_MODEL,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": answer},
                "finish_reason": "stop",
            }],
            "usage": usage,
        }

    # Compatibility SSE: generation finishes before chunks are sent, so this is not
    # true token-by-token streaming. It supports clients that require SSE format.
    def sse(payload: dict[str, Any]) -> str:
        return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

    async def stream_response():
        yield sse({
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": GEMINI_MODEL,
            "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
        })
        for start in range(0, len(answer), 80):
            yield sse({
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": GEMINI_MODEL,
                "choices": [{"index": 0, "delta": {"content": answer[start:start + 80]}, "finish_reason": None}],
            })
        yield sse({
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": GEMINI_MODEL,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        })
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        stream_response(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
