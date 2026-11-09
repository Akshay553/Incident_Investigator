"""One provider adapter and explicit scripted test doubles; neither can execute tools."""

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel


def structured_schema(schema: type[BaseModel]) -> dict:
    """OpenAI strict outputs require every property, including nullable defaults."""
    def strict(value):
        if isinstance(value, list):
            return [strict(item) for item in value]
        if not isinstance(value, dict):
            return value
        result = {key: strict(item) for key, item in value.items() if key != "default"}
        if result.get("type") == "object":
            result["additionalProperties"] = False
            result["required"] = list(result.get("properties", {}))
        return result

    return strict(schema.model_json_schema())


@dataclass(frozen=True)
class ModelResponse:
    response_json: str
    input_tokens: int
    output_tokens: int
    failure: "LLMFailure | None" = None


class LLMFailure(RuntimeError):
    def __init__(self, code: str, message: str, *, retryable: bool = False):
        super().__init__(message)
        self.code = code
        self.retryable = retryable


class StructuredLLM(Protocol):
    mode: str
    model: str

    async def respond(
        self, task_id: str, system: str, user: str,
        schema: type[BaseModel], max_output_tokens: int,
    ) -> ModelResponse: ...


class OpenAILLM:
    mode = "live"

    def __init__(self, model: str = "gpt-4.1-mini", *, temperature: float = 0.0):
        from openai import AsyncOpenAI

        if not os.environ.get("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY is not set; configure it locally, never in chat")
        self.model = model
        if not 0 <= temperature <= 2:
            raise ValueError("temperature must be between 0 and 2")
        self.temperature = temperature
        # Runtime owns retries and deadlines, including calls that fail.
        self.client = AsyncOpenAI(max_retries=0)

    async def close(self):
        await self.client.close()

    async def respond(self, task_id, system, user, schema, max_output_tokens):
        from openai import APIConnectionError, APIResponseValidationError, APIStatusError, APITimeoutError

        try:
            response = await self.client.responses.create(
                model=self.model,
                instructions=system,
                input=user,
                max_output_tokens=max_output_tokens,
                temperature=self.temperature,
                store=False,
                text={"format": {
                    "type": "json_schema",
                    "name": schema.__name__,
                    "schema": structured_schema(schema),
                    "strict": True,
                }},
            )
        except (APIConnectionError, APITimeoutError) as exc:
            raise LLMFailure("connection", "OpenAI connection or timeout failure", retryable=True) from exc
        except APIStatusError as exc:
            raise LLMFailure(
                f"http_{exc.status_code}", f"OpenAI returned HTTP {exc.status_code}",
                retryable=exc.status_code in {408, 429} or exc.status_code >= 500,
            ) from exc
        except APIResponseValidationError as exc:
            raise LLMFailure("invalid_provider_response", "OpenAI response envelope was invalid") from exc
        if response.usage is None:
            raise LLMFailure("missing_usage", "OpenAI did not return token usage")
        # Parsing occurs in the coordinator AFTER retaining usage, even for invalid output.
        return ModelResponse(
            response_json=response.output_text,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            failure=(
                LLMFailure("incomplete_response", f"OpenAI response status: {response.status}")
                if response.status != "completed" else (
                    LLMFailure("refusal", "OpenAI returned no structured response")
                    if not response.output_text else None
                )
            ),
        )


class ScriptedLLM:
    """A supplied script, not model reasoning or a diagnostic algorithm."""
    mode = "mocked"
    model = "scripted-test-double"

    def __init__(self, responses: dict[str, list[dict | str | LLMFailure]]):
        self.responses = {key: list(values) for key, values in responses.items()}
        self.calls: list[str] = []

    @classmethod
    def from_file(cls, path: Path):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or any(
            not isinstance(key, str) or not isinstance(values, list)
            for key, values in payload.items()
        ):
            raise ValueError("mock script must map task IDs to lists of structured replies")
        return cls(payload)

    async def respond(self, task_id, system, user, schema, max_output_tokens):
        self.calls.append(task_id)
        responses = self.responses.get(task_id)
        if not responses:
            raise LLMFailure("script_exhausted", f"No mocked response for task {task_id}")
        response = responses.pop(0)
        if isinstance(response, LLMFailure):
            raise response
        return ModelResponse(
            response_json=response if isinstance(response, str) else json.dumps(response),
            input_tokens=0, output_tokens=0,
        )
