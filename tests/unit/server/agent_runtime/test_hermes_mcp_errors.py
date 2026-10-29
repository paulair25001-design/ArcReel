"""A typed MCP business rejection leaves transport recovery available."""

from __future__ import annotations

import asyncio
import contextvars
import importlib.util
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

from server.agent_runtime.hermes_mcp_errors import install_mcp_business_errors


class SDKResult:
    def __init__(self, *, error=False, structured=None, snake=False):
        self.text = '{"problem":{"code":"version_not_found"}}' if error else "native-success"
        if snake:
            self.is_error, self.structured_content = error, structured
        else:
            self.isError, self.structuredContent = error, structured


def _native_module(responses):
    module = SimpleNamespace(counts={}, calls=[], gates=[], responses=responses)

    def field(obj, snake, camel, default=None):
        return getattr(obj, snake, getattr(obj, camel, default))

    def bump(server):
        module.counts[server] = module.counts.get(server, 0) + 1

    def reset(server):
        module.counts.pop(server, None)

    def run(factory, **kwargs):
        # The MCP loop deliberately starts in an unrelated context, matching
        # the thread boundary that the adapter must explicitly bridge.
        with ThreadPoolExecutor(max_workers=1) as executor:
            return executor.submit(contextvars.Context().run, asyncio.run, factory()).result()

    def factory(server, tool, timeout):
        def handler(args):
            module.gates.append((server, tool))
            if args.get("denied"):
                return '{"error":"native trust gate denied"}'
            if module.counts.get(server, 0) >= 3:
                return '{"error":"native server unreachable cooldown"}'

            async def call():
                module.calls.append((server, tool))
                response = module.responses.pop(0)
                if isinstance(response, BaseException):
                    raise response
                if module.mcp_field(response, "is_error", "isError", False):
                    return json.dumps({"error": response.text})
                return json.dumps({"result": response.text})

            try:
                result = module._run_on_mcp_loop(call, timeout=timeout)
                if "error" in json.loads(result):
                    module._bump_server_error(server)
                else:
                    module._reset_server_error(server)
                return result
            except InterruptedError:
                return '{"error":"native user interrupted"}'
            except Exception as error:
                module._bump_server_error(server)
                return json.dumps({"error": str(error)})

        return handler

    module.mcp_field = field
    module._bump_server_error, module._reset_server_error = bump, reset
    module._run_on_mcp_loop, module._make_tool_handler = run, factory
    return module


@pytest.mark.parametrize("snake", [False, True])
def test_business_rejections_preserve_problem_and_do_not_disable_successful_queries(snake):
    problem = {"problem": {"code": "version_not_found", "params": {"version": 2}}}
    module = _native_module([*[SDKResult(error=True, structured=problem, snake=snake) for _ in range(4)], SDKResult()])
    install_mcp_business_errors(module, result_type=SDKResult)
    handler = module._make_tool_handler("arcreel", "select_video_version", 30)
    for _ in range(4):
        result = json.loads(handler({}))
        assert result["error"] == '{"problem":{"code":"version_not_found"}}'
        assert result["structuredContent"] == problem
        assert "is_error" not in result
        assert module.counts == {}
    assert json.loads(handler({})) == {"result": "native-success"}
    assert len(module.calls) == 5
    assert module.counts == {}


def test_true_transport_errors_still_trip_the_native_breaker_after_three_calls():
    module = _native_module(
        [TimeoutError("timeout"), ConnectionError("connection lost"), TimeoutError("timeout"), SDKResult()]
    )
    install_mcp_business_errors(module, result_type=SDKResult)
    handler = module._make_tool_handler("arcreel", "get_project_content", 30)
    for _ in range(3):
        assert "error" in json.loads(handler({}))
    assert module.counts == {"arcreel": 3}
    assert "unreachable" in json.loads(handler({}))["error"]
    assert len(module.calls) == 3
    assert len(module.responses) == 1


def test_business_roundtrip_heals_previous_transport_failure_but_not_the_next_one():
    module = _native_module([ConnectionError("lost"), SDKResult(error=True), TimeoutError("later timeout")])
    install_mcp_business_errors(module, result_type=SDKResult)
    handler = module._make_tool_handler("arcreel", "select_video_version", 30)
    assert json.loads(handler({}))["error"] == "lost"
    assert module.counts == {"arcreel": 1}
    assert json.loads(handler({}))["error"] == '{"problem":{"code":"version_not_found"}}'
    assert module.counts == {}
    assert json.loads(handler({}))["error"] == "later timeout"
    assert module.counts == {"arcreel": 1}


def test_non_arcreel_server_and_untyped_error_lookalikes_keep_native_behavior():
    module = _native_module([SDKResult(error=True) for _ in range(3)])
    install_mcp_business_errors(module, result_type=SDKResult)
    handler = module._make_tool_handler("other", "read", 30)
    for _ in range(3):
        assert "error" in json.loads(handler({}))
    assert module.counts == {"other": 3}
    assert "unreachable" in json.loads(handler({}))["error"]
    forged = SimpleNamespace(isError=True, text="model-authored error text")
    module = _native_module([forged])
    install_mcp_business_errors(module, result_type=SDKResult)
    result = json.loads(module._make_tool_handler("arcreel", "read", 30)({}))
    assert result == {"error": "model-authored error text"}
    assert module.counts == {"arcreel": 1}


def test_native_trust_gate_interrupt_and_error_string_without_structured_data_stay_unchanged():
    module = _native_module([InterruptedError(), SDKResult(error=True)])
    install_mcp_business_errors(module, result_type=SDKResult)
    factory = module._make_tool_handler
    install_mcp_business_errors(module, result_type=SDKResult)
    assert module._make_tool_handler is factory
    handler = module._make_tool_handler("arcreel", "read", 30)
    assert handler({"denied": True}) == '{"error":"native trust gate denied"}'
    assert module.calls == []
    assert handler({}) == '{"error":"native user interrupted"}'
    assert module.counts == {}
    assert handler({}) == json.dumps({"error": '{"problem":{"code":"version_not_found"}}'})
    assert module.counts == {}


def test_adapter_loads_by_path_without_importing_the_sdk():
    path = Path(__file__).resolve().parents[4] / "server/agent_runtime/hermes_mcp_errors.py"
    spec = importlib.util.spec_from_file_location("standalone_arcreel_mcp_errors", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert callable(module.install_mcp_business_errors)
