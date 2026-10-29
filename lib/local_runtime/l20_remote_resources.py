"""Small SSH-side resource operations; never owns scripts, shots or media jobs.

Invoked with a JSON request on stdin. Only the explicitly named ArcReel ComfyUI
process is started/stopped. llama.cpp lifecycle uses its native router API.
"""

import hashlib
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path


def request(route, data=None, port=8080, timeout=20):
    raw = None if data is None else json.dumps(data).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}" + route, data=raw, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)


def identity(pid):
    root = Path("/proc") / str(pid)
    try:
        raw = (root / "cmdline").read_bytes()
        # comm may contain spaces; starttime is field 22 after the final ')'.
        fields = (root / "stat").read_text().rsplit(")", 1)[1].split()
        if fields[0] == "Z":
            return None
        birth = fields[19]
        return {
            "pid": int(pid),
            "birth": birth,
            "uid": root.stat().st_uid,
            "argv_sha256": hashlib.sha256(raw).hexdigest(),
        }
    except (OSError, ValueError, IndexError):
        return None


def router_identity():
    matches = []
    for path in Path("/proc").iterdir():
        if not path.name.isdigit():
            continue
        try:
            args = (path / "cmdline").read_bytes().decode().strip("\0").split("\0")
            if (
                args
                and "llama-server" in Path(args[0]).name
                and "--port" in args
                and args[args.index("--port") + 1] == "8080"
            ):
                matches.append(identity(path.name))
        except (OSError, ValueError, IndexError):
            pass
    if len(matches) != 1 or matches[0] is None:
        raise RuntimeError("Cannot identify one existing L20 router")
    return {"boot_id": Path("/proc/sys/kernel/random/boot_id").read_text(encoding="utf-8").strip(), **matches[0]}


def snapshot():
    body = request("/models")
    loaded = []
    for item in body["data"]:
        status = item["status"]
        if status["value"] in {"loaded", "loading", "sleeping"}:
            args = status.get("args", [])
            port = int(args[args.index("--port") + 1]) if "--port" in args else None
            loaded.append({"id": item["id"], "state": status["value"], "port": port})
    gpu = subprocess.check_output(
        ["/usr/bin/nvidia-smi", "--query-gpu=memory.free,memory.used", "--format=csv,noheader,nounits"], text=True
    ).strip()
    return {"router": router_identity(), "models": loaded, "health": request("/health"), "gpu_mib": gpu}


def slots(models):
    gpu = subprocess.check_output(
        ["/usr/bin/nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"], text=True
    )
    gpu_pids = {int(line.split(",")[0]): int(line.split(",")[1]) for line in gpu.splitlines() if line.strip()}
    receipts = []
    for item in models:
        if item["state"] != "loaded" or not item["port"]:
            raise RuntimeError("Original model is not stably loaded")
        value = request("/slots", port=item["port"])
        if not isinstance(value, list) or not value or any(x.get("is_processing") is not False for x in value):
            raise RuntimeError("Original model has an active or unknown slot")
        children = []
        for path in Path("/proc").iterdir():
            if not path.name.isdigit():
                continue
            try:
                args = (path / "cmdline").read_bytes().decode().strip("\0").split("\0")
                if (
                    args
                    and "llama-server" in Path(args[0]).name
                    and "--port" in args
                    and int(args[args.index("--port") + 1]) == item["port"]
                ):
                    children.append(int(path.name))
            except (OSError, ValueError, IndexError):
                pass
        if len(children) != 1 or gpu_pids.get(children[0], 0) <= 0:
            raise RuntimeError("Original model child is not resident on the GPU")
        receipts.append(
            {
                "model": item["id"],
                "process": identity(children[0]),
                "gpu_mib": gpu_pids[children[0]],
                "idle_slots": len(value),
            }
        )
    return {"idle": True, "children": receipts}


def runtime_root(value):
    root = Path(value)
    base = Path("/nvme1/.media-workspace/work/arcreel")
    if (
        root.parent != base
        or not root.name.startswith("session-")
        or not all(c.isalnum() or c == "-" for c in root.name)
    ):
        raise ValueError("Invalid ArcReel runtime root")
    base.mkdir(mode=0o700, exist_ok=True)
    if base.is_symlink() or base.stat().st_uid != os.getuid():
        raise RuntimeError("Runtime directory identity differs")
    return root


def start_comfy(data):
    root = runtime_root(data["root"])
    root.mkdir(mode=0o700, exist_ok=True)
    if root.is_symlink():
        raise RuntimeError("Runtime directory is redirected")
    marker = root / "process.json"
    if marker.exists():
        old = json.loads(marker.read_text())
        if identity(old["pid"]) == old:
            return {"root": str(root), "process": old, "reused": True}
        raise RuntimeError("Recorded ComfyUI process no longer exists; inspect before relaunch")
    # Fail before launch when a different service owns this dedicated port.
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 18191))
    python = data["python"]
    engine = Path(data["engine"])
    if not Path(python).is_file() or not (engine / "main.py").is_file():
        raise RuntimeError("Configured ComfyUI engine is not installed")
    for name in ("input", "output", "temp", "user"):
        (root / name).mkdir(mode=0o700, exist_ok=True)
    paths = root / "models.yaml"
    paths.write_text(
        "arcreel:\n  base_path: /nvme1/.media-workspace/models\n  diffusion_models: diffusion_models\n  text_encoders: text_encoders\n  vae: vae\n  loras: loras\n"
    )
    # Core inference nodes are part of the installed engine. External project
    # controllers/custom orchestration nodes are disabled for this service.
    args = [
        python,
        str(engine / "main.py"),
        "--listen",
        "127.0.0.1",
        "--port",
        "18191",
        "--input-directory",
        str(root / "input"),
        "--output-directory",
        str(root / "output"),
        "--temp-directory",
        str(root / "temp"),
        "--user-directory",
        str(root / "user"),
        "--extra-model-paths-config",
        str(paths),
        "--disable-auto-launch",
        "--disable-all-custom-nodes",
        "--disable-api-nodes",
    ]
    env = dict(os.environ)
    env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHONUNBUFFERED="1")
    with (root / "comfy.log").open("ab") as log:
        process = subprocess.Popen(
            args, cwd=engine, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=True
        )
    value = identity(process.pid)
    if value is None:
        raise RuntimeError("ComfyUI exited before its identity could be recorded")
    marker.write_text(json.dumps(value))
    return {"root": str(root), "process": value, "reused": False}


def comfy_identity(data):
    root = runtime_root(data["root"])
    saved = json.loads((root / "process.json").read_text())
    if identity(saved["pid"]) != saved:
        raise RuntimeError("The owned ComfyUI process exited or changed identity")
    inodes = set()
    for line in Path("/proc/net/tcp").read_text(encoding="utf-8").splitlines()[1:]:
        row = line.split()
        if row[1] == "0100007F:470F" and row[3] == "0A":
            inodes.add("socket:[" + row[9] + "]")
    fds = Path("/proc") / str(saved["pid"]) / "fd"
    owns = any(os.readlink(fd) in inodes for fd in fds.iterdir()) if inodes else False
    return {"process": saved, "listening": owns}


def stop_comfy(data):
    root = runtime_root(data["root"])
    path = root / "process.json"
    if not path.exists():
        return {"stopped": True, "was_started": False}
    saved = json.loads(path.read_text())
    actual = identity(saved["pid"])
    if actual is None:
        return {"stopped": True, "already_exited": True}
    if saved != actual or saved["uid"] != os.getuid():
        raise RuntimeError("Refusing to stop a different process")
    os.kill(saved["pid"], signal.SIGTERM)
    for _ in range(60):
        if identity(saved["pid"]) is None:
            return {"stopped": True}
        time.sleep(0.5)
    raise RuntimeError("Owned ComfyUI process did not stop")


def main():
    os.umask(0o077)
    if os.getuid() != 974 or not os.path.ismount("/nvme1"):
        raise RuntimeError("Configured L20 host/mount identity differs")
    data = json.load(sys.stdin)
    operation = data["operation"]
    if operation == "snapshot":
        result = snapshot()
    elif operation == "slots":
        result = slots(data["models"])
    elif operation in {"load", "unload"}:
        if router_identity() != data["router"]:
            raise RuntimeError("Original router identity changed")
        result = request("/models/" + operation, {"model": data["model"]}, timeout=240)
    elif operation == "start_comfy":
        result = start_comfy(data)
    elif operation == "stop_comfy":
        result = stop_comfy(data)
    elif operation == "comfy_identity":
        result = comfy_identity(data)
    elif operation == "log":
        result = {"tail": (runtime_root(data["root"]) / "comfy.log").read_text(errors="replace")[-10000:]}
    else:
        raise ValueError("Unknown resource operation")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
