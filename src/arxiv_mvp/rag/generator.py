"""OpenAI-compatible chat-completions client for the /ask RAG path.

Endpoint contract:
- POST ``LLM_URL`` (default http://localhost:8000/v1/chat/completions)
- Request: {model, messages, max_tokens, temperature}
- Response: OpenAI-compatible {choices: [{message: {content}, finish_reason}]}

The reference deployment uses Qwen3.5-4B behind vLLM. Any instruct model with
an OpenAI-compatible endpoint works; quality and latency will differ.

Env overrides:
  LLM_URL            full /v1/chat/completions URL
  LLM_MODEL          model id as served
  LLM_API_KEY        optional bearer token
  LLM_TIMEOUT_S      request timeout (default 120)
  LLM_MAX_TOKENS     default 512
  LLM_TEMPERATURE    default 0.3
"""
import os
import time
from dataclasses import dataclass

import httpx

from ..utils.logging import get_logger

log = get_logger("rag.generator")

DEFAULT_LLM_URL = os.environ.get(
    "LLM_URL", "http://localhost:8000/v1/chat/completions")
DEFAULT_LLM_MODEL = os.environ.get("LLM_MODEL", "Qwen/Qwen3.5-4B")
DEFAULT_MAX_TOKENS = int(os.environ.get("LLM_MAX_TOKENS", "512"))
DEFAULT_TEMPERATURE = float(os.environ.get("LLM_TEMPERATURE", "0.3"))
DEFAULT_TIMEOUT_S = float(os.environ.get("LLM_TIMEOUT_S", "120.0"))
DEFAULT_CONTEXT_CHARS_PER_CHUNK = 2000
DEFAULT_CONTEXT_MAX_CHUNKS = 10

# System prompt: answer ONLY from provided passages; inline [N] citations.
ASK_SYSTEM_PROMPT = (
    "You are a research assistant answering questions about academic papers. "
    "Use ONLY the information in the provided passages. If the passages do not "
    "contain enough information to answer, say so explicitly — do not "
    "hallucinate. Be concise: 2-3 sentences. Cite every claim inline with the "
    "bracketed passage number(s) it came from, placed IMMEDIATELY after the "
    "claim, e.g. \"RAG cuts hallucinations by 30-50% [1]\" or \"[2, 4]\". "
    "Do not collect citations at the end."
)


def _strip_think(text: str) -> str:
    """Remove <think>...</think> CoT blocks emitted by some Qwen builds.

    Handles an unclosed block running to EOF. No-op for models that do not
    emit thinking blocks.
    """
    import re

    text = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL)
    text = re.sub(r"<think>.*\Z", "", text, flags=re.DOTALL)
    return text


@dataclass
class GeneratedAnswer:
    """Result of one /ask generation call."""
    answer: str
    elapsed_ms: float
    model: str
    finish_reason: str | None
    usage: dict | None  # prompt_tokens, completion_tokens, total_tokens


class Generator:
    """Async client for an OpenAI-compatible /v1/chat/completions endpoint.

    Lifecycle:
        async with Generator() as gen:
            ans = await gen.generate(query, context_chunks)
        # or ensure_client()/aclose() manually in a lifespan hook.
    """

    def __init__(
        self,
        url: str | None = None,
        model: str | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        context_chars_per_chunk: int = DEFAULT_CONTEXT_CHARS_PER_CHUNK,
        context_max_chunks: int = DEFAULT_CONTEXT_MAX_CHUNKS,
        api_key: str | None = None,
        chat_template_kwargs: dict | None = None,
    ):
        self.url = url or os.environ.get("LLM_URL") or DEFAULT_LLM_URL
        self.model = model or os.environ.get("LLM_MODEL") or DEFAULT_LLM_MODEL
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout_s = timeout_s
        self.context_chars_per_chunk = context_chars_per_chunk
        self.context_max_chunks = context_max_chunks
        self.api_key = api_key if api_key is not None else os.environ.get("LLM_API_KEY")
        # vLLM-specific flag; leave None for generic OpenAI-compatible servers.
        self.chat_template_kwargs = chat_template_kwargs
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> "Generator":
        await self.ensure_client()
        return self

    async def __aexit__(self, *a):
        await self.aclose()

    async def ensure_client(self) -> None:
        if self._client is None:
            headers = {}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            self._client = httpx.AsyncClient(
                timeout=self.timeout_s, headers=headers)

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def warmup(self) -> float:
        """Send one tiny request to warm the model. Best-effort: logs failure."""
        await self.ensure_client()
        t0 = time.monotonic()
        try:
            body = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": ASK_SYSTEM_PROMPT},
                    {"role": "user", "content": "warmup: reply with 'ok'"},
                ],
                "max_tokens": 8,
                "temperature": 0.0,
            }
            if self.chat_template_kwargs:
                body["chat_template_kwargs"] = self.chat_template_kwargs
            r = await self._client.post(self.url, json=body)
            r.raise_for_status()
            elapsed_ms = (time.monotonic() - t0) * 1000.0
            log.info("llm_warmup_complete", url=self.url, model=self.model,
                     elapsed_ms=round(elapsed_ms, 1))
            return elapsed_ms
        except Exception as e:
            elapsed_ms = (time.monotonic() - t0) * 1000.0
            log.warning("llm_warmup_failed", url=self.url, model=self.model,
                        error=f"{type(e).__name__}: {str(e)[:200]}",
                        elapsed_ms=round(elapsed_ms, 1))
            return elapsed_ms

    def build_prompt(self, query: str, context_chunks: list[str]) -> list[dict]:
        """Assemble chat messages; context passages carry [N] markers."""
        truncated = context_chunks[: self.context_max_chunks]
        context = "\n\n".join(
            f"[{i + 1}] {c[: self.context_chars_per_chunk]}"
            for i, c in enumerate(truncated)
        )
        user_msg = f"Question: {query}\n\nPassages:\n{context}"
        return [
            {"role": "system", "content": ASK_SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ]

    async def generate(self, query: str, context_chunks: list[str]) -> GeneratedAnswer:
        """Generate an answer from retrieved passages."""
        await self.ensure_client()
        body = {
            "model": self.model,
            "messages": self.build_prompt(query, context_chunks),
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        if self.chat_template_kwargs:
            body["chat_template_kwargs"] = self.chat_template_kwargs
        t0 = time.monotonic()
        r = await self._client.post(self.url, json=body)
        r.raise_for_status()
        data = r.json()
        elapsed_ms = (time.monotonic() - t0) * 1000.0
        choice = data["choices"][0]
        answer = _strip_think(choice["message"]["content"].strip())
        return GeneratedAnswer(
            answer=answer,
            elapsed_ms=elapsed_ms,
            model=data.get("model", self.model),
            finish_reason=choice.get("finish_reason"),
            usage=data.get("usage"),
        )
