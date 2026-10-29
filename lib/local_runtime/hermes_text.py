"""Text and structured output through a fresh, tool-free local Hermes process."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from uuid import uuid4

import jsonschema
from pydantic import BaseModel, ValidationError

from lib.backends.text_backends.base import (
    StructuredOutputExhaustedError,
    TextCapability,
    TextGenerationRequest,
    TextGenerationResult,
    check_truncation,
    resolve_schema,
    strip_json_code_fence,
    strip_leading_think_block,
)
from lib.infra.async_thread import run_sync_transaction
from lib.infra.json_io import atomic_write_json
from lib.local_runtime.config import HERMES_PROVIDER, LocalRuntimeConfig, get_local_runtime_config
from lib.local_runtime.hermes_process import (
    LocalProcessError,
    ProcessSpawner,
    build_hermes_environment,
    decode_process_json,
    prepare_hermes_home,
    run_local_process,
)


def _optional_token(value: object) -> int | None:
    return value if type(value) is int and value >= 0 else None


class HermesTextBackend:
    """Every generation uses the selected local model with no tools or fallback."""

    def __init__(self, config: LocalRuntimeConfig | None = None, *, spawn: ProcessSpawner | None = None) -> None:
        self._config = config or get_local_runtime_config()
        self._spawn = spawn

    @property
    def name(self) -> str:
        return HERMES_PROVIDER

    @property
    def model(self) -> str:
        return self._config.hermes_model

    @property
    def capabilities(self) -> set[TextCapability]:
        return {TextCapability.TEXT_GENERATION, TextCapability.STRUCTURED_OUTPUT, TextCapability.VISION}

    def _prepare_request(self, request: TextGenerationRequest) -> tuple[Path, Path]:
        home = self._config.hermes_home / "text" / uuid4().hex
        prepare_hermes_home(self._config, home, max_output_tokens=request.max_output_tokens)
        workspace = home / "workspace"
        workspace.mkdir()
        images: list[str] = []
        for image in request.images or []:
            if image.path is not None:
                path = image.path.resolve(strict=True)
                if not path.is_file():
                    raise ValueError("Hermes image input is not a local file")
                images.append(str(path))
            elif image.url and image.url.startswith("data:image/"):
                images.append(image.url)
            else:
                raise ValueError("Local Hermes accepts local image files or inline image data only")
        prompt = request.prompt
        response_schema = resolve_schema(request.response_schema) if request.response_schema is not None else None
        if response_schema is not None:
            schema = json.dumps(response_schema, ensure_ascii=False)
            prompt += "\n\nReturn only JSON matching this schema, without Markdown fences:\n" + schema
        payload = {
            "source": str(self._config.hermes_source),
            "model": self.model,
            "base_url": self._config.hermes_base_url,
            "prompt": prompt,
            "system_prompt": request.system_prompt,
            "images": images,
            "max_output_tokens": request.max_output_tokens or 65536,
            "response_schema": response_schema,
        }
        path = home / "request.json"
        atomic_write_json(path, payload)
        path.chmod(0o600)
        return home, path

    def _normalize_structured(self, text: str, schema: dict | type) -> str:
        body = strip_json_code_fence(strip_leading_think_block(text))
        try:
            if isinstance(schema, type) and issubclass(schema, BaseModel):
                return schema.model_validate_json(body).model_dump_json()
            value = json.loads(body)
            jsonschema.validate(value, resolve_schema(schema))
            return json.dumps(value, ensure_ascii=False)
        except (ValueError, ValidationError, jsonschema.ValidationError, jsonschema.SchemaError) as exc:
            raise StructuredOutputExhaustedError(
                provider=self.name,
                model=self.model,
                reason="local Hermes response did not satisfy the requested schema",
            ) from exc

    async def generate(self, request: TextGenerationRequest) -> TextGenerationResult:
        home, path = await run_sync_transaction(self._prepare_request, request)
        process = await run_local_process(
            (self._config.hermes_python, "-B", str(Path(__file__).with_name("hermes_text_worker.py")), str(path)),
            env=build_hermes_environment(self._config, home),
            cwd=home / "workspace",
            timeout_seconds=self._config.hermes_timeout_seconds,
            spawn=self._spawn,
        )
        payload = decode_process_json(process)
        if (
            payload.get("provider") != "omlx"
            or payload.get("model") != self.model
            or payload.get("base_url") != self._config.hermes_base_url
        ):
            raise LocalProcessError("Hermes worker identity did not match the selected local route")
        truncated = check_truncation(
            payload.get("finish_reason"),
            provider=self.name,
            model=self.model,
            output_tokens=_optional_token(payload.get("output_tokens")),
            structured=request.response_schema is not None,
            truncation_values=("length", "max_tokens", "incomplete"),
        )
        if any(payload.get(key) for key in ("failed", "interrupted")) or (
            not truncated and (not payload.get("completed") or payload.get("partial"))
        ):
            raise LocalProcessError("Hermes did not complete this local text invocation")
        text = payload.get("text")
        if not isinstance(text, str) or not text.strip():
            raise LocalProcessError("Hermes returned no text")
        if request.response_schema is not None:
            text = await asyncio.to_thread(self._normalize_structured, text, request.response_schema)
        return TextGenerationResult(
            text=text,
            provider=self.name,
            model=self.model,
            input_tokens=_optional_token(payload.get("input_tokens")),
            output_tokens=_optional_token(payload.get("output_tokens")),
            cache_read_tokens=_optional_token(payload.get("cache_read_tokens")),
            truncated=truncated,
        )
