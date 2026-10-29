from __future__ import annotations

import ast
import asyncio
import copy
import hashlib
import io
import json
import subprocess
from email import policy
from email.parser import BytesParser
from pathlib import Path

import httpx
import pytest
from PIL import Image

from arcreel_market_core.video_backend_contract import VideoGenerationRequest
from lib.backends.http_status_errors import ArtifactDownloadError
from lib.billing.call_failure import classify_call_failure
from lib.generation.task_failure_encoding import encode_task_failure_message
from lib.infra.ffmpeg import ffmpeg_executable
from lib.infra.media_probe import probe_media
from lib.local_runtime.l20_direct_backend import L20DirectVideoBackend
from lib.local_runtime.l20_direct_config import L20DirectConfig
from lib.local_runtime.l20_direct_gateway import L20DirectError, L20DirectGateway
from lib.local_runtime.l20_direct_graphs import (
    aligned_frames,
    build_image_graph,
    build_video_graph,
    normalized_workflow_sha256,
)
from tests.fakes import captured_provider_job_ids


def png(size=(64, 64)):
    buffer = io.BytesIO()
    Image.new("RGB", size, "red").save(buffer, format="PNG")
    return buffer.getvalue()


class Resources:
    def __init__(self):
        self.begun, self.finished = [], []

    def begin(self, project):
        self.begun.append(project)

    def finish(self, project):
        self.finished.append(project)
        return {"state": "RESTORED"}


class Comfy:
    """The native ComfyUI HTTP boundary; never installs or invokes a model."""

    def __init__(self):
        self.calls = []
        self.inputs = {}
        self.prompts = {}
        self.lose_submit_response = False
        self.duplicate_match = False
        self.history_mismatch = False
        self.external_busy = False
        self.free_error = False
        self.video_bytes = None
        self.coerce_numbers = False
        self.fail_download = False

    def entry(self, prompt_id):
        body = self.prompts[prompt_id]
        extra = {**body["extra_data"], "client_id": body["client_id"]}
        workflow = copy.deepcopy(body["prompt"])
        if self.coerce_numbers:
            workflow["sample"]["inputs"]["cfg"] = 1
        if self.history_mismatch:
            workflow = {"wrong": {"class_type": "SaveImage", "inputs": {}}}
        return {
            "prompt": [1, prompt_id, workflow, extra, ["output"]],
            "status": {"completed": True, "status_str": "success"},
            "outputs": {
                "output": {
                    "videos" if self.video_bytes else "images": [
                        {
                            "filename": "one.mp4" if self.video_bytes else "one.png",
                            "subfolder": "arcreel",
                            "type": "output",
                        }
                    ]
                }
            },
        }

    def request(self, request):
        path = request.url.path
        self.calls.append((request.method, path))
        if path == "/upload/image":
            message = BytesParser(policy=policy.default).parsebytes(
                b"Content-Type: " + request.headers["Content-Type"].encode() + b"\r\n\r\n" + request.content
            )
            part = next(item for item in message.iter_parts() if item.get_filename())
            self.inputs[part.get_filename()] = part.get_payload(decode=True)
            return httpx.Response(200, json={"name": part.get_filename(), "subfolder": "arcreel", "type": "input"})
        if path == "/view":
            if request.url.params["type"] == "output" and self.fail_download:
                return httpx.Response(503, text="temporary artifact transfer failure")
            return httpx.Response(
                200,
                content=self.inputs[request.url.params["filename"]]
                if request.url.params["type"] == "input"
                else (self.video_bytes or png()),
            )
        if path == "/object_info":
            graph = build_image_graph("cube", 64, 64, 1, [], "arcreel")
            edits = build_image_graph("cube", 64, 64, 1, ["ref"], "arcreel")
            video = build_video_graph("move", 64, 64, 1, ["first", "tail"], 24, "arcreel")
            return httpx.Response(
                200,
                json={
                    node["class_type"]: {}
                    for node in [*graph.workflow.values(), *edits.workflow.values(), *video.workflow.values()]
                },
            )
        if path == "/prompt":
            self.prompts["prompt-one"] = json.loads(request.content)
            if self.lose_submit_response:
                self.lose_submit_response = False
                raise httpx.ReadTimeout("accepted, response lost", request=request)
            return httpx.Response(200, json={"prompt_id": "prompt-one", "node_errors": {}})
        if path == "/queue":
            entries = []
            if self.duplicate_match:
                entries = [self.entry(key)["prompt"] for key in self.prompts]
                entries.append([2, "second-accepted-copy", *entries[0][2:]])
            if self.external_busy:
                entries.append([3, "other-job", {}, {}, []])
            return httpx.Response(200, json={"queue_running": entries, "queue_pending": []})
        if path == "/history":
            return httpx.Response(200, json={key: self.entry(key) for key in self.prompts})
        if path.startswith("/history/"):
            key = path.rsplit("/", 1)[1]
            return httpx.Response(200, json={key: self.entry(key)})
        if path == "/free":
            if self.free_error:
                return httpx.Response(500, json={"error": "free failed"})
            return httpx.Response(200, json={})
        raise AssertionError(f"Unexpected HTTP operation: {request.method} {path}")


def gateway(tmp_path):
    server, resources = Comfy(), Resources()
    config = L20DirectConfig("http://127.0.0.1:18191", "http://127.0.0.1:18080", tmp_path, 0.001, 10)
    transport = httpx.MockTransport(server.request)
    client = L20DirectGateway(config, resources, client_factory=lambda **kw: httpx.Client(transport=transport, **kw))
    return client, server, resources


def intent():
    return {
        "project": "demo",
        "media_type": "image",
        "prompt": "a red cube",
        "seed": 7,
        "canvas": [64, 64],
        "references": [],
    }


def test_native_image_http_lifecycle_retains_completed_artifact_without_resubmission(tmp_path):
    client, server, resources = gateway(tmp_path)
    handle = client.prepare(intent(), task_identity="one")
    artifact = client.run(handle)
    assert artifact.path.read_bytes() == png()
    assert artifact.sha256 == hashlib.sha256(png()).hexdigest()
    assert client.run(handle).path == artifact.path
    assert server.calls.count(("POST", "/prompt")) == 1
    assert server.calls.count(("POST", "/free")) == 1
    assert resources.begun == ["demo"]
    assert resources.finished == []
    assert client.finalize(project="demo")["state"] == "RESTORED"
    assert resources.finished == ["demo"]


def test_lost_submit_response_reconciles_exact_frozen_graph_and_identity_without_repost(tmp_path):
    client, server, _ = gateway(tmp_path)
    handle = client.prepare(intent(), task_identity="one")
    server.lose_submit_response = True
    with pytest.raises(httpx.ReadTimeout):
        client.run(handle)
    artifact = client.run(handle)
    assert artifact.path.read_bytes() == png()
    assert artifact.record["prompt_id"] == "prompt-one"
    assert server.calls.count(("POST", "/prompt")) == 1


def test_ambiguous_remote_identity_fails_closed_without_duplicate_post(tmp_path):
    client, server, _ = gateway(tmp_path)
    handle = client.prepare(intent(), task_identity="one")
    server.lose_submit_response = True
    with pytest.raises(httpx.ReadTimeout):
        client.run(handle)
    server.duplicate_match = True
    with pytest.raises(L20DirectError, match="single exact match"):
        client.run(handle)
    assert server.calls.count(("POST", "/prompt")) == 1


def test_completed_history_must_be_the_actual_frozen_graph(tmp_path):
    client, server, _ = gateway(tmp_path)
    handle = client.prepare(intent(), task_identity="one")
    server.history_mismatch = True
    with pytest.raises(L20DirectError, match="history differs"):
        client.run(handle)
    assert not (tmp_path / "jobs" / handle / "native.png").exists()


def test_comfy_lossless_numeric_coercion_preserves_wire_graph_and_records_both_raw_hashes(tmp_path):
    client, server, _ = gateway(tmp_path)
    handle = client.prepare(intent(), task_identity="one")
    server.coerce_numbers = True
    result = client.run(handle)
    comparison = result.record["history_graph_comparison"]
    assert comparison["equivalent"] is True
    assert comparison["policy"] == "json-integral-floats-v1"
    assert comparison["actual_raw_sha256"] != comparison["frozen_raw_sha256"]
    assert result.record["workflow"]["sample"]["inputs"]["cfg"] == 1.0
    assert type(result.record["workflow"]["sample"]["inputs"]["cfg"]) is float


def test_successful_inference_download_failure_uses_native_retry_download_classification(tmp_path):
    client, server, _ = gateway(tmp_path)
    handle = client.prepare(intent(), task_identity="one")
    server.fail_download = True
    with pytest.raises(ArtifactDownloadError) as caught:
        client.run(handle)
    assert encode_task_failure_message(caught.value).startswith("[artifact_download_failed]")
    assert classify_call_failure(caught.value).error_code == "download_failed"
    server.fail_download = False
    assert client.run(handle).path.read_bytes() == png()
    assert server.calls.count(("POST", "/prompt")) == 1


@pytest.mark.parametrize("actual", [True, "4", 5, 4.01, None])
def test_numeric_normalization_never_accepts_boolean_string_or_different_operation_value(actual):
    expected = {"node": {"class_type": "KSampler", "inputs": {"steps": 4}}}
    changed = {"node": {"class_type": "KSampler", "inputs": {"steps": actual}}}
    assert normalized_workflow_sha256(expected) != normalized_workflow_sha256(changed)


def test_numeric_normalization_accepts_only_identical_json_number_values():
    expected = {"node": {"class_type": "KSampler", "inputs": {"steps": 4, "enabled": True}}}
    actual = {"node": {"class_type": "KSampler", "inputs": {"steps": 4.0, "enabled": True}}}
    assert normalized_workflow_sha256(expected) == normalized_workflow_sha256(actual)
    actual["node"]["class_type"] = "SaveImage"
    assert normalized_workflow_sha256(expected) != normalized_workflow_sha256(actual)


def test_busy_comfy_queue_blocks_a_new_graph_without_posting_or_freeing(tmp_path):
    client, server, _ = gateway(tmp_path)
    handle = client.prepare(intent(), task_identity="one")
    server.external_busy = True
    with pytest.raises(L20DirectError, match="queue is not empty"):
        client.run(handle)
    assert ("POST", "/prompt") not in server.calls
    assert ("POST", "/free") not in server.calls


def test_cleanup_failure_preserves_the_completed_provider_artifact(tmp_path):
    client, server, _ = gateway(tmp_path)
    handle = client.prepare(intent(), task_identity="one")
    server.free_error = True
    artifact = client.run(handle)
    assert artifact.path.read_bytes() == png()
    assert artifact.record["state"] == "completed"
    assert artifact.record["cleanup_observation"] == "HTTPStatusError"
    assert client.run(handle).sha256 == artifact.sha256
    assert server.calls.count(("POST", "/prompt")) == 1


def test_input_snapshot_upload_and_response_reference_preserve_original_bytes(tmp_path):
    client, server, _ = gateway(tmp_path / "runtime")
    original = tmp_path / "source.png"
    original.write_bytes(png())
    request = intent()
    request["references"] = [{"path": str(original), "sha256": hashlib.sha256(png()).hexdigest()}]
    handle = client.prepare(request, task_identity="one")
    result = client.run(handle)
    assert original.read_bytes() == png()
    assert next(iter(server.inputs.values())) == png()
    assert result.record["workflow"]["condition"]["inputs"]["images.image_1"] == ["reference1", 0]
    assert result.record["workflow"]["reference1"]["inputs"]["image"].startswith("arcreel/arcreel-l20-")


def test_video_graph_uses_only_inference_nodes_and_explicit_both_endpoints():
    graph = build_video_graph("dolly in", 1024, 576, 4, ["first.png", "tail.png"], 192, "arcreel/one")
    kinds = {node["class_type"] for node in graph.workflow.values()}
    assert not any("Probe" in name or "Guard" in name or "Hold" in name for name in kinds)
    assert graph.workflow["condition"]["inputs"]["last_frame"] == ["tail", 0]
    assert graph.workflow["schedule"]["inputs"]["steps"] == 4
    assert graph.workflow["video"]["inputs"]["fps"] == 24
    assert graph.native_frames == 192
    assert aligned_frames(24) == 39
    with pytest.raises(ValueError, match="target-tail"):
        build_video_graph("move", 1024, 576, 4, ["first.png"], 192, "arcreel/one")


@pytest.mark.asyncio
@pytest.mark.parametrize("native_rate", [24, 30])
async def test_video_backend_persists_queue_identity_and_checks_real_decoding_and_delivery(tmp_path, native_rate):
    client, server, resources = gateway(tmp_path / "state")
    first, tail, fixture = tmp_path / "first.png", tmp_path / "tail.png", tmp_path / "fixture.mp4"

    def prepare():
        Image.new("RGB", (1024, 576), "red").save(first)
        Image.new("RGB", (1024, 576), "blue").save(tail)
        subprocess.run(
            [
                ffmpeg_executable(),
                "-v",
                "error",
                "-nostdin",
                "-f",
                "lavfi",
                "-i",
                f"color=c=red:s=1024x576:r={native_rate}",
                "-f",
                "lavfi",
                "-i",
                "anullsrc=r=48000:cl=stereo",
                "-frames:v",
                "39",
                "-t",
                str(39 / native_rate),
                "-c:v",
                "libx264",
                "-preset",
                "ultrafast",
                "-c:a",
                "aac",
                str(fixture),
            ],
            capture_output=True,
            check=True,
            timeout=30,
        )
        return fixture.read_bytes()

    server.video_bytes = await asyncio.to_thread(prepare)
    backend = L20DirectVideoBackend(client.config, gateway=client)
    request = VideoGenerationRequest(
        "move",
        tmp_path / "delivery.mp4",
        start_image=first,
        end_image=tail,
        aspect_ratio="16:9",
        duration_seconds=1,
        resolution="576p",
        task_id="worker-one",
        project_name="demo",
    )
    with captured_provider_job_ids() as persisted:
        if native_rate == 30:
            with pytest.raises(L20DirectError, match="native video canvas, frame count, duration or audio"):
                await backend.generate(request)
        else:
            result = await backend.generate(request)
            delivered = await probe_media(result.video_path)
            stream = delivered.first_stream("video")
            assert stream is not None
            assert stream.packet_count == 24
            assert delivered.first_stream("audio") is not None
            assert result.provenance is not None
            assert result.provenance["runtime"] == "arcreel-native-comfyui"
    assert len(persisted) == 1
    assert persisted[0]["job_id"].startswith("arcreel-l20-")
    assert server.calls.count(("POST", "/prompt")) == 1
    assert resources.finished == []


def test_previously_submitted_factory_handles_are_not_reinterpreted_by_the_new_provider(tmp_path):
    client, server, resources = gateway(tmp_path)
    with pytest.raises(L20DirectError, match="independent ArcReel"):
        client.run("a" * 64)
    assert server.calls == []
    assert resources.begun == []


@pytest.mark.asyncio
async def test_queue_identity_persistence_failure_prevents_all_remote_side_effects(tmp_path):
    client, server, resources = gateway(tmp_path / "state")
    first, tail = tmp_path / "first.png", tmp_path / "tail.png"

    def prepare():
        Image.new("RGB", (1024, 576), "red").save(first)
        Image.new("RGB", (1024, 576), "blue").save(tail)

    await asyncio.to_thread(prepare)
    backend = L20DirectVideoBackend(client.config, gateway=client)
    request = VideoGenerationRequest(
        "move",
        tmp_path / "delivery.mp4",
        start_image=first,
        end_image=tail,
        aspect_ratio="16:9",
        duration_seconds=1,
        resolution="576p",
        task_id="worker-one",
        project_name="demo",
    )
    with (
        captured_provider_job_ids(fail=RuntimeError("database unavailable")),
        pytest.raises(RuntimeError, match="database unavailable"),
    ):
        await backend.generate(request)
    assert server.calls == []
    assert resources.begun == []


@pytest.mark.parametrize(
    "value",
    [
        "https://cloud.example",
        "http://user:pass@localhost:123",
        "http://127.0.0.1:0",
        "http://127.0.0.1:99999",
        "http://127.0.0.1/path",
    ],
)
def test_runtime_connection_is_explicit_loopback_without_cloud_fallback(tmp_path, value):
    with pytest.raises(ValueError, match="explicit project-owned loopback"):
        L20DirectConfig(value, "http://127.0.0.1:18080", tmp_path)


def test_independent_provider_has_no_import_of_previous_execution_modules():
    root = Path(__file__).resolve().parents[4]
    for name in ("l20_direct_config.py", "l20_direct_graphs.py", "l20_direct_gateway.py", "l20_direct_backend.py"):
        tree = ast.parse((root / "lib/local_runtime" / name).read_bytes())
        imports = [node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
        assert not any(
            value and (value.endswith(("l20_bridge", "l20_backend")) or "factory" in value) for value in imports
        )
