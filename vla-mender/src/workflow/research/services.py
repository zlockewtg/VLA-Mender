"""Task/GPU-scoped ownership for VLA-Mender's local tool services."""

from __future__ import annotations

import hashlib
import os
import signal
import socket
import subprocess
import time
from pathlib import Path
from typing import Any

import yaml

from .util import (
    add_loopback_no_proxy,
    atomic_write_json,
    atomic_write_text,
    read_json,
    utc_now,
)


PROFILE_SERVERS = {
    "default": (
        ("local.launch_sam3_server.main", "sam3", 14),
        ("local.launch_contact_graspnet_server.main", "graspnet", 15),
        ("local.launch_pyroki_server.main", "pyroki", 16),
    ),
    "minimal": (("local.launch_pyroki_server.main", "pyroki", 16),),
}


def _process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _fingerprint(pid: int) -> str | None:
    try:
        return hashlib.sha256(Path(f"/proc/{pid}/cmdline").read_bytes()).hexdigest()
    except OSError:
        return None


def _port_ready(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=1.0):
            return True
    except OSError:
        return False


class TaskServiceManager:
    def __init__(
        self,
        campaign_root: str | Path,
        *,
        project_root: str | Path,
        python: str | Path,
        gpu_id: int,
        gpu_slot: int,
        profile: str,
        port_base: int,
        port_stride: int,
        manage: bool,
        keep_alive: bool,
        startup_timeout_s: float,
        extra_env: dict[str, str],
    ):
        if profile not in PROFILE_SERVERS:
            raise ValueError(f"unsupported local service profile: {profile}")
        self.campaign_root = Path(campaign_root).resolve()
        self.project_root = Path(project_root).resolve()
        self.python = Path(python)
        self.gpu_id = int(gpu_id)
        self.gpu_slot = int(gpu_slot)
        self.profile = profile
        self.port_base = int(port_base)
        self.port_stride = int(port_stride)
        self.manage = bool(manage)
        self.keep_alive = bool(keep_alive)
        self.startup_timeout_s = float(startup_timeout_s)
        self.extra_env = dict(extra_env)
        self.root = self.campaign_root / "runtime" / "services" / f"gpu_{gpu_id}"
        self.root.mkdir(parents=True, exist_ok=True)
        self.state_path = self.root / "service_state.json"
        self.config_path = self.root / "servers.yaml"
        self.log_path = self.root / "launcher.log"
        self.process: subprocess.Popen[Any] | None = None

    def endpoints(self) -> dict[str, str]:
        base = self.port_base + self.gpu_slot * self.port_stride
        values: dict[str, str] = {}
        for _, name, offset in PROFILE_SERVERS[self.profile]:
            values[name] = f"http://127.0.0.1:{base + offset}"
        return values

    def environment(self) -> dict[str, str]:
        endpoints = self.endpoints()
        mapping = {
            "sam3": "SAM3_SERVICE_URL",
            "graspnet": "GRASPNET_SERVICE_URL",
            "pyroki": "PYROKI_SERVICE_URL",
        }
        return {mapping[name]: url for name, url in endpoints.items()}

    def _write_config(self) -> None:
        endpoints = self.endpoints()
        servers = []
        for target, name, _ in PROFILE_SERVERS[self.profile]:
            servers.append(
                {
                    "_target_": target,
                    "host": "127.0.0.1",
                    "port": int(endpoints[name].rsplit(":", 1)[1]),
                }
            )
        atomic_write_text(
            self.config_path,
            yaml.safe_dump({"api_servers": servers}, sort_keys=False),
        )

    def _compatible_running_state(self) -> bool:
        if not self.state_path.is_file():
            return False
        state = read_json(self.state_path)
        pid = int(state.get("pid", -1))
        if (
            pid < 1
            or not _process_alive(pid)
            or _fingerprint(pid) != state.get("process_fingerprint")
            or state.get("profile") != self.profile
            or state.get("endpoints") != self.endpoints()
        ):
            return False
        return all(_port_ready(int(url.rsplit(":", 1)[1])) for url in self.endpoints().values())

    def ensure(self) -> dict[str, str]:
        if self._compatible_running_state():
            return self.environment()
        if not self.manage:
            missing = [
                url
                for url in self.endpoints().values()
                if not _port_ready(int(url.rsplit(":", 1)[1]))
            ]
            if missing:
                raise RuntimeError(f"required external repair services are unavailable: {missing}")
            return self.environment()
        if self.state_path.is_file():
            self.stop(force=True)
        self._write_config()
        launcher = self.project_root / "vla-mender" / "tools" / "launch_servers.py"
        if not launcher.is_file():
            raise FileNotFoundError(launcher)
        command = [
            str(self.python),
            str(launcher),
            "--config-path",
            str(self.config_path),
            "--gpus",
            str(self.gpu_id),
            "--workers",
            "1",
            "--log-dir",
            str(self.root / "server_logs"),
            "--timeout",
            str(self.startup_timeout_s),
        ]
        environment = dict(os.environ)
        environment.update(self.extra_env)
        add_loopback_no_proxy(environment)
        environment["CUDA_VISIBLE_DEVICES"] = str(self.gpu_id)
        log_stream = self.log_path.open("a", encoding="utf-8")
        self.process = subprocess.Popen(
            command,
            cwd=self.project_root,
            env=environment,
            stdout=log_stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        log_stream.close()
        deadline = time.monotonic() + self.startup_timeout_s
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(f"repair service launcher exited with {self.process.returncode}")
            if all(_port_ready(int(url.rsplit(":", 1)[1])) for url in self.endpoints().values()):
                break
            time.sleep(1.0)
        else:
            if self.process is not None and self.process.poll() is None:
                try:
                    os.killpg(self.process.pid, signal.SIGTERM)
                except OSError:
                    pass
            raise TimeoutError("repair services did not become ready")
        atomic_write_json(
            self.state_path,
            {
                "schema_version": 1,
                "pid": self.process.pid,
                "process_fingerprint": _fingerprint(self.process.pid),
                "owned": True,
                "profile": self.profile,
                "gpu_id": self.gpu_id,
                "endpoints": self.endpoints(),
                "started_at": utc_now(),
            },
        )
        return self.environment()

    def stop(self, *, force: bool = False) -> None:
        if self.keep_alive and not force:
            return
        if not self.state_path.is_file():
            if self.process is not None and self.process.poll() is None:
                try:
                    os.killpg(self.process.pid, signal.SIGTERM)
                except OSError:
                    pass
            return
        state = read_json(self.state_path)
        pid = int(state.get("pid", -1))
        if (
            bool(state.get("owned"))
            and pid > 0
            and _process_alive(pid)
            and _fingerprint(pid) == state.get("process_fingerprint")
        ):
            try:
                os.killpg(pid, signal.SIGTERM)
            except OSError:
                pass
        state["stopped_at"] = utc_now()
        state["running"] = False
        atomic_write_json(self.state_path, state)
