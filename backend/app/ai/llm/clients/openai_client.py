from typing import AsyncGenerator, Any

from openai import AsyncOpenAI, OpenAI

from app.ai.llm.clients.base import LLMClient
from app.ai.llm.types import LLMResponse, LLMUsage


class OpenAi(LLMClient):
    def __init__(self, api_key: str, base_url: str = "https://api.openai.com/v1"):
        super().__init__()
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.async_client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    def inference(self, model_id: str, prompt: str) -> LLMResponse:
        temperature = 1 if model_id == "gpt-5" else 0.3
        response = self.client.responses.create(
            model=model_id,
            input={
                "type": "text",
                "content": prompt.strip(),
                "content_type": "text",
            },
            temperature=temperature,
        )
        usage = self._extract_usage(getattr(response, "usage", None))
        self._set_last_usage(usage)
        content = response.output_text or ""
        return LLMResponse(text=content, usage=usage)

    async def inference_stream(self, model_id: str, prompt: str) -> AsyncGenerator[str, None]:
        temperature = 1 if model_id == "gpt-5" else 0.3
        stream = await self.async_client.responses.create(
            model=model_id,
            input={
                "type": "text",
                "content": prompt.strip(),
                "content_type": "text",
            },
            stream=True,
            temperature=temperature,
        )

        prompt_tokens = 0
        completion_tokens = 0
        async for chunk in stream:
            usage = self._extract_usage(getattr(chunk, "usage", None))
            if usage.prompt_tokens or usage.completion_tokens:
                prompt_tokens = usage.prompt_tokens or prompt_tokens
                completion_tokens = usage.completion_tokens or completion_tokens

            content = chunk.output_text
            if content is not None:
                yield content

        self._set_last_usage(
            LLMUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )
        )

    @staticmethod
    def _extract_usage(raw: Any) -> LLMUsage:
        if raw is None:
            return LLMUsage()
        if isinstance(raw, dict):
            prompt = raw.get("prompt_tokens") or 0
            completion = raw.get("completion_tokens") or 0
            return LLMUsage(prompt_tokens=int(prompt or 0), completion_tokens=int(completion or 0))
        prompt = getattr(raw, "prompt_tokens", 0) or getattr(raw, "prompt_tokens_cost", 0) or 0
        completion = getattr(raw, "completion_tokens", 0) or getattr(raw, "completion_tokens_cost", 0) or 0
        return LLMUsage(prompt_tokens=int(prompt or 0), completion_tokens=int(completion or 0))
