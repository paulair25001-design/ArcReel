"""Keep ArcReel business rejections separate from native MCP transport failure.

The native handler, trust checks, timeout/reconnect handling and circuit
breaker remain in use. Only a typed error response from a completed RPC is
recognized as a healthy transport; model text cannot establish that fact.
"""

from __future__ import annotations

import contextvars
import functools
import importlib
import inspect
import json
from typing import Any


class _Call:
    def __init__(self, server_name: str) -> None:
        self.server_name = server_name
        self.error_result: Any = None


def install_mcp_business_errors(module: Any, *, result_type: type | None = None) -> None:
    """Install before the vision wrapper; apply only to the ``arcreel`` server.

    ``result_type`` is a test seam. Production recognizes the installed MCP
    SDK's real CallToolResult, across its snake/camel field aliases.
    """
    if getattr(module, "_arcreel_business_error_adapter", False):
        return
    sdk_result_type = result_type or importlib.import_module("mcp.types").CallToolResult
    original_factory = module._make_tool_handler
    original_field = module.mcp_field
    original_bump = module._bump_server_error
    original_reset = module._reset_server_error
    original_run = module._run_on_mcp_loop
    active: contextvars.ContextVar[_Call | None] = contextvars.ContextVar("arcreel_mcp_business_call", default=None)

    @functools.wraps(original_field)
    def field(obj: Any, snake: str, camel: str, default: Any = None) -> Any:
        value = original_field(obj, snake, camel, default)
        call = active.get()
        if (
            call is not None
            and snake == "is_error"
            and camel == "isError"
            and value is True
            and isinstance(obj, sdk_result_type)
        ):
            # The native handler reads this after session.call_tool returned.
            call.error_result = obj
        return value

    @functools.wraps(original_bump)
    def bump(server_name: str) -> Any:
        call = active.get()
        if call is not None and call.server_name == server_name and call.error_result is not None:
            return original_reset(server_name)
        return original_bump(server_name)

    @functools.wraps(original_run)
    def run_on_loop(coro_or_factory: Any, *args: Any, **kwargs: Any) -> Any:
        call = active.get()
        if call is None:
            return original_run(coro_or_factory, *args, **kwargs)
        started = False

        async def scoped() -> Any:
            nonlocal started
            started = True
            token = active.set(call)
            try:
                coro: Any = coro_or_factory() if callable(coro_or_factory) else coro_or_factory
                return await coro
            finally:
                active.reset(token)

        try:
            return original_run(scoped, *args, **kwargs)
        except BaseException:
            if not started and inspect.iscoroutine(coro_or_factory):
                coro_or_factory.close()
            raise

    @functools.wraps(original_factory)
    def factory(server_name: str, tool_name: str, tool_timeout: float) -> Any:
        original = original_factory(server_name, tool_name, tool_timeout)
        if server_name != "arcreel":
            return original

        @functools.wraps(original)
        def handler(*args: Any, **kwargs: Any) -> Any:
            call = _Call(server_name)
            token = active.set(call)
            try:
                result = original(*args, **kwargs)
            finally:
                active.reset(token)
            if call.error_result is None:
                return result
            # Preserve the native error text and failure shape. Native Hermes
            # omits machine fields on isError; retain the server's original
            # structured problem using the same standard field as successes.
            structured = original_field(call.error_result, "structured_content", "structuredContent")
            if structured is None:
                return result
            try:
                payload = json.loads(result)
                if not isinstance(payload, dict) or "error" not in payload or "structuredContent" in payload:
                    return result
                payload["structuredContent"] = structured
                return json.dumps(payload, ensure_ascii=False)
            except (TypeError, ValueError):
                return result

        return handler

    module.mcp_field = field
    module._bump_server_error = bump
    module._run_on_mcp_loop = run_on_loop
    module._make_tool_handler = factory
    module._arcreel_business_error_adapter = True
