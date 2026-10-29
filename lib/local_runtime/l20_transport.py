"""Project-owned SSH connection to an explicitly configured inference host."""

from __future__ import annotations

import json
import os
import shlex
import subprocess
from pathlib import Path
from urllib.parse import urlsplit


class L20SSHTransport:
    def __init__(self, config):
        self.config = config
        self.target = os.environ.get("ARCREEL_L20_SSH_TARGET", "")
        self.options = json.loads(os.environ.get("ARCREEL_L20_SSH_OPTIONS_JSON", "[]"))
        if not self.target or self.target.startswith("-") or not isinstance(self.options, list):
            raise ValueError("Configure the project's explicit L20 SSH connection")
        if any(not isinstance(x, str) for x in self.options):
            raise ValueError("L20 SSH options must be strings")
        if "StrictHostKeyChecking=yes" not in self.options:
            raise ValueError("L20 SSH requires a verified host key")
        config.state_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.socket = config.state_dir / "ssh.sock"
        self.base = ["ssh", "-T", "-S", str(self.socket), *self.options]

    def connect(self):
        check = subprocess.run([*self.base, "-O", "check", self.target], capture_output=True, timeout=20)
        if check.returncode == 0:
            return
        forwards = []
        for url, remote_port in ((self.config.comfy_url, 18191), (self.config.router_url, 8080)):
            parsed = urlsplit(url)
            if parsed.hostname != "127.0.0.1" or not parsed.port:
                raise ValueError("Managed L20 forwarding requires explicit IPv4 loopback ports")
            forwards += ["-L", f"127.0.0.1:{parsed.port}:127.0.0.1:{remote_port}"]
        # The detached SSH master keeps stderr open. A PIPE would keep the
        # parent's communicate() waiting after successful daemonization.
        with (self.config.state_dir / "ssh.log").open("ab") as diagnostic:
            result = subprocess.run(
                [
                    *self.base,
                    "-M",
                    "-o",
                    "ControlPersist=no",
                    "-o",
                    "ExitOnForwardFailure=yes",
                    "-fN",
                    *forwards,
                    self.target,
                ],
                stdin=subprocess.DEVNULL,
                stdout=diagnostic,
                stderr=diagnostic,
                start_new_session=True,
                timeout=40,
            )
        if result.returncode:
            raise RuntimeError("The project-owned L20 SSH forwarding could not be established")

    def rpc(self, operation, payload=None, *, timeout=120):
        self.connect()
        worker = Path(__file__).with_name("l20_remote_resources.py").read_text()
        command = "/usr/bin/python3 -B -c " + shlex.quote(worker)
        result = subprocess.run(
            [*self.base, self.target, command],
            input=json.dumps({"operation": operation, **(payload or {})}),
            text=True,
            capture_output=True,
            timeout=timeout,
        )
        if result.returncode:
            # Remote exception output may contain engine paths; keep it private.
            from lib.infra.json_io import atomic_write_json

            atomic_write_json(
                self.config.state_dir / "last-resource-error.json",
                {
                    "operation": operation,
                    "stderr": result.stderr[-12000:],
                    "returncode": result.returncode,
                },
            )
            raise RuntimeError("L20 resource operation failed; its private diagnostic has been retained")
        return json.loads(result.stdout)

    def close(self):
        subprocess.run([*self.base, "-O", "exit", self.target], capture_output=True, timeout=20)
