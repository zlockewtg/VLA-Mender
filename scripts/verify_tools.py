#!/usr/bin/env python3
"""Verify VLA-Mender tools and optionally smoke-test their local servers."""
from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "vla-mender" / "tools"
CONTACT_ROOT = ROOT / "third_party" / "contact_graspnet_pytorch"
SAM3_ROOT = ROOT / "third_party" / "sam3"
LIBERO_ROOT = ROOT / "third_party" / "LIBERO-PRO"
ROBOSUITE_ROOT = ROOT / "third_party" / "libero_dependencies" / "robosuite"
RUNTIME_DIR = ROOT / ".runtime" / "tool_servers"
STATE_FILE = RUNTIME_DIR / "state.json"
SERVICES = (("sam3", 8114), ("graspnet", 8115), ("pyroki", 8116))


class VerificationError(RuntimeError):
    pass


class SkipVerification(RuntimeError):
    pass


def _prepend_paths(*paths: Path) -> None:
    for path in reversed(paths):
        value = str(path.resolve())
        if value not in sys.path:
            sys.path.insert(0, value)


def _import_vendor(name: str, root: Path):
    module = importlib.import_module(name)
    origin = getattr(module, "__file__", None)
    if origin is None or not Path(origin).resolve().is_relative_to(root.resolve()):
        raise VerificationError(f"{name} resolved outside {root}: {origin}")
    return module


def _version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError as exc:
        raise VerificationError(f"distribution {name!r} is not installed") from exc


def _check(label: str, fn: Callable[[], str | None]) -> None:
    try:
        detail = fn()
    except Exception as exc:
        print(f"FAIL {label}: {type(exc).__name__}: {exc}")
        raise VerificationError(label) from exc
    print(f"PASS {label}" + (f" ({detail})" if detail else ""))


def _quick_checks() -> None:
    if sys.version_info[:2] != (3, 12):
        raise VerificationError(f"expected Python 3.12, got {sys.version.split()[0]}")
    print(f"PASS Python {sys.version.split()[0]}")
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ.pop("PYTHONPATH", None)
    _prepend_paths(TOOLS, CONTACT_ROOT, SAM3_ROOT, LIBERO_ROOT, ROBOSUITE_ROOT)

    def contact() -> str:
        _import_vendor("contact_graspnet_pytorch", CONTACT_ROOT)
        estimator = importlib.import_module("contact_graspnet_pytorch.contact_grasp_estimator")
        if not hasattr(estimator, "GraspEstimator"):
            raise VerificationError("GraspEstimator is not exported")
        return str(Path(importlib.import_module("contact_graspnet_pytorch").__file__).resolve())

    def sam3() -> str:
        module = _import_vendor("sam3", SAM3_ROOT)
        if not hasattr(module, "build_sam3_image_model"):
            raise VerificationError("build_sam3_image_model is not exported")
        return str(Path(module.__file__).resolve())

    def pyroki() -> str:
        module = importlib.import_module("pyroki")
        origin = getattr(module, "__file__", None)
        if origin is None:
            raise VerificationError("pyroki has no __file__")
        return str(Path(origin).resolve())

    _check("Contact-GraspNet import", contact)
    _check("SAM3 import", sam3)
    _check("PyRoKi import", pyroki)
    for name in ("contact-graspnet-pytorch", "sam3", "pyroki"):
        _check(f"distribution {name}", lambda name=name: _version(name))
    try:
        import torch
        print(f"PASS torch CUDA available={torch.cuda.is_available()} visible={os.environ.get('CUDA_VISIBLE_DEVICES', '<all')}")
    except Exception as exc:
        print(f"FAIL torch runtime: {type(exc).__name__}: {exc}")
        raise VerificationError("torch runtime") from exc


def _tail_logs(log_dir: Path, lines: int = 40) -> str:
    chunks: list[str] = []
    for path in sorted(log_dir.glob("*.log")):
        try:
            content = path.read_text(errors="replace").splitlines()[-lines:]
        except OSError as exc:
            content = [f"<cannot read {path}: {exc}>"]
        chunks.append(f"--- {path.name} ---\n" + "\n".join(content))
    return "\n".join(chunks)


def _http_health(host: str, port: int) -> dict:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(f"http://{host}:{port}/health", timeout=2) as response:
        if response.status != 200:
            raise VerificationError(f"HTTP {response.status}")
        payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, dict):
        raise VerificationError("health response is not a JSON object")
    return payload


def _sam3_checkpoint_exists(path: Path) -> bool:
    if path.is_file():
        return path.name == "sam3.pt"
    if not path.is_dir():
        return False
    return (path / "sam3.pt").is_file() or any(
        (path / "snapshots").glob("*/sam3.pt")
    )


def _resolve_checkpoints() -> tuple[Path, Path]:
    sam3_candidates: list[Path] = []
    if os.environ.get("SAM3_CHECKPOINT_PATH"):
        sam3_candidates.append(Path(os.environ["SAM3_CHECKPOINT_PATH"]).expanduser())
    if os.environ.get("HF_HOME"):
        sam3_candidates.append(
            Path(os.environ["HF_HOME"]).expanduser()
            / "hub/models--facebook--sam3"
        )
    sam3_candidates.extend(
        [
            Path("/mnt/public/tgy/ckpts/huggingface/hub/models--facebook--sam3"),
            Path.home() / ".cache/huggingface/hub/models--facebook--sam3",
        ]
    )
    sam3_path = next(
        (path.resolve() for path in sam3_candidates if _sam3_checkpoint_exists(path)),
        None,
    )
    if sam3_path is None:
        checked = ", ".join(map(str, sam3_candidates))
        raise SkipVerification(
            "SAM3 checkpoint not found; set SAM3_CHECKPOINT_PATH "
            f"(checked: {checked})"
        )

    grasp_candidates: list[Path] = []
    if os.environ.get("CONTACT_GRASPNET_CHECKPOINT_DIR"):
        grasp_candidates.append(
            Path(os.environ["CONTACT_GRASPNET_CHECKPOINT_DIR"]).expanduser()
        )
    grasp_candidates.extend(
        [
            CONTACT_ROOT / "checkpoints/contact_graspnet/checkpoints",
            ROOT.parent
            / "capx-aspire/capx/third_party/contact_graspnet_pytorch"
            / "checkpoints/contact_graspnet/checkpoints",
        ]
    )
    grasp_dir = next(
        (
            path.resolve()
            for path in grasp_candidates
            if (path / "model.pt").is_file()
            and (path.parent / "config.yaml").is_file()
        ),
        None,
    )
    if grasp_dir is None:
        checked = ", ".join(map(str, grasp_candidates))
        raise SkipVerification(
            "Contact-GraspNet checkpoint not found; set "
            f"CONTACT_GRASPNET_CHECKPOINT_DIR (checked: {checked})"
        )

    os.environ["SAM3_CHECKPOINT_PATH"] = str(sam3_path)
    os.environ["CONTACT_GRASPNET_CHECKPOINT_DIR"] = str(grasp_dir)
    print(f"PASS SAM3 checkpoint ({sam3_path})")
    print(f"PASS Contact-GraspNet checkpoint ({grasp_dir})")
    return sam3_path, grasp_dir


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except (OSError, ProcessLookupError):
        return False
    return True


def _read_state() -> dict:
    if not STATE_FILE.is_file():
        return {}
    try:
        value = json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _healthy_services() -> list[str]:
    healthy: list[str] = []
    for name, port in SERVICES:
        try:
            payload = _http_health("127.0.0.1", port)
        except (OSError, urllib.error.URLError, VerificationError, json.JSONDecodeError):
            continue
        if payload.get("status") == "up" and payload.get("service") == name:
            healthy.append(name)
    return healthy


def _terminate_process_group(pgid: int, timeout: float = 15.0) -> None:
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            os.killpg(pgid, 0)
        except ProcessLookupError:
            return
        time.sleep(0.25)
    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def _stop_servers() -> None:
    state = _read_state()
    pgid = int(state.get("pgid", 0) or 0)
    if pgid > 0:
        _terminate_process_group(pgid)
    STATE_FILE.unlink(missing_ok=True)
    deadline = time.monotonic() + 10
    while _healthy_services() and time.monotonic() < deadline:
        time.sleep(0.25)
    remaining = _healthy_services()
    if remaining:
        raise VerificationError(
            "services are still running but are not owned by the recorded launcher: "
            + ", ".join(remaining)
        )
    print("PASS tool servers stopped")


def _server_smoke(timeout: float) -> None:
    uv = shutil.which("uv")
    if uv is None:
        raise VerificationError("uv executable not found")
    sam3_path, grasp_dir = _resolve_checkpoints()
    launcher = TOOLS / "launch_servers.py"
    if not launcher.is_file():
        raise VerificationError(f"launcher not found: {launcher}")

    healthy = _healthy_services()
    if len(healthy) == len(SERVICES):
        state = _read_state()
        if state:
            state["status"] = "ready"
            state.setdefault("ready_at", time.time())
            STATE_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")
        print("PASS tool servers already running in background")
        for name in healthy:
            print(f"PASS server {name} /health")
        if state:
            print(f"INFO state: {STATE_FILE}")
            print(f"INFO logs: {RUNTIME_DIR}")
        return
    if healthy:
        raise VerificationError(
            "only some service ports are occupied: " + ", ".join(healthy)
        )

    state = _read_state()
    stale_pid = int(state.get("pid", 0) or 0)
    if stale_pid and _pid_alive(stale_pid):
        raise VerificationError(
            f"recorded launcher PID {stale_pid} is alive but services are unhealthy; "
            "run --stop-servers first"
        )

    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    STATE_FILE.unlink(missing_ok=True)
    log_dir = RUNTIME_DIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    launcher_log = RUNTIME_DIR / "launcher.log"

    env = os.environ.copy()
    active_env = Path(sys.prefix).resolve()
    env["VIRTUAL_ENV"] = str(active_env)
    env["PATH"] = str(active_env / "bin") + os.pathsep + env.get("PATH", "")
    env["SAM3_CHECKPOINT_PATH"] = str(sam3_path)
    env["CONTACT_GRASPNET_CHECKPOINT_DIR"] = str(grasp_dir)
    command = [
        uv,
        "run",
        "--no-sync",
        "--active",
        str(launcher),
        "--profile",
        "default",
        "--log-dir",
        str(log_dir),
        "--timeout",
        str(timeout),
    ]
    print("INFO starting in background: " + " ".join(command))
    with launcher_log.open("w", encoding="utf-8") as output:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=output,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )

    state = {
        "pid": process.pid,
        "pgid": process.pid,
        "status": "starting",
        "command": command,
        "virtual_env": str(active_env),
        "sam3_checkpoint": str(sam3_path),
        "contact_graspnet_checkpoint_dir": str(grasp_dir),
        "launcher_log": str(launcher_log),
        "service_log_dir": str(log_dir),
        "started_at": time.time(),
    }
    STATE_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")

    try:
        deadline = time.monotonic() + timeout
        pending = list(SERVICES)
        while pending and time.monotonic() < deadline:
            if process.poll() is not None:
                launcher_output = launcher_log.read_text(errors="replace")[-4000:]
                raise VerificationError(
                    f"launcher exited with {process.returncode}\n{launcher_output}\n"
                    f"{_tail_logs(log_dir)}"
                )
            remaining = []
            for name, port in pending:
                try:
                    payload = _http_health("127.0.0.1", port)
                    if payload.get("status") != "up":
                        raise VerificationError(f"status={payload.get('status')!r}")
                    if payload.get("service") != name:
                        raise VerificationError(
                            f"port {port} belongs to {payload.get('service')!r}"
                        )
                    print(f"PASS server {name} /health")
                except (
                    OSError,
                    urllib.error.URLError,
                    VerificationError,
                    json.JSONDecodeError,
                ):
                    remaining.append((name, port))
            pending = remaining
            if pending:
                time.sleep(1)
        if pending:
            raise VerificationError(
                "server readiness timeout: "
                + ", ".join(name for name, _ in pending)
                + "\n"
                + _tail_logs(log_dir)
            )
    except Exception:
        _terminate_process_group(process.pid)
        STATE_FILE.unlink(missing_ok=True)
        raise

    state["status"] = "ready"
    state["ready_at"] = time.time()
    STATE_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")
    print(f"PASS tool servers running in background (launcher PID {process.pid})")
    print(f"INFO state: {STATE_FILE}")
    print(f"INFO logs: {RUNTIME_DIR}")


def _libero_smoke() -> None:
    assets = LIBERO_ROOT / "libero" / "libero" / "assets"
    robosuite_assets = ROBOSUITE_ROOT / "robosuite" / "models" / "assets"
    for path in (assets, robosuite_assets):
        if not path.exists():
            raise SkipVerification(f"required runtime asset directory missing: {path}")
    with tempfile.TemporaryDirectory(prefix="vla-mender-libero-") as temp:
        config_root = Path(temp)
        benchmark_root = LIBERO_ROOT / "libero" / "libero"
        (config_root / "config.yaml").write_text("\n".join([
            f"benchmark_root: {benchmark_root}",
            f"bddl_files: {benchmark_root / 'bddl_files'}",
            f"init_states: {benchmark_root / 'init_files'}",
            f"datasets: {benchmark_root.parent / 'datasets'}",
            f"assets: {benchmark_root / 'assets'}", "" ]))
        os.environ["LIBERO_CONFIG_PATH"] = str(config_root)
        _prepend_paths(LIBERO_ROOT, ROBOSUITE_ROOT)
        try:
            libero = _import_vendor("libero", LIBERO_ROOT)
            from libero import benchmark
            from libero.envs import OffScreenRenderEnv
            from libero.utils import get_libero_path
            suite = benchmark.get_benchmark_dict()["libero_10"]()
            task = suite.get_task(0)
            bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
            # LIBERO init-state files are trusted NumPy pickles. PyTorch 2.6+
            # defaults to weights_only=True, which rejects this legacy format.
            init_path = Path(get_libero_path("init_states")) / task.problem_folder / task.init_states_file
            try:
                import torch
                init_states = torch.load(init_path, weights_only=False)
            except TypeError:  # torch versions before the weights_only argument
                init_states = torch.load(init_path)
            if not bddl.is_file() or init_states is None or len(init_states) == 0:
                raise VerificationError(f"LIBERO task resources missing: {bddl}")
            env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=128, camera_widths=128)
            try:
                env.seed(0)
                env.reset()
                env.set_init_state(init_states[0])
                for _ in range(10):
                    env.step([0.0] * 7)
            finally:
                env.close()
            print(f"PASS LIBERO smoke ({task.name})")
        finally:
            os.environ.pop("LIBERO_CONFIG_PATH", None)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-smoke", action="store_true", help="start local tool servers, verify /health, and leave them running in background")
    parser.add_argument("--stop-servers", action="store_true", help="stop background tool servers started by this script")
    parser.add_argument("--libero-smoke", action="store_true", help="run the fixed LIBERO task-0 simulator smoke")
    parser.add_argument("--timeout", type=float, default=120.0, help="server readiness timeout in seconds")
    args = parser.parse_args()
    try:
        if args.stop_servers:
            _stop_servers()
            return 0
        _quick_checks()
        if args.server_smoke:
            _server_smoke(args.timeout)
        if args.libero_smoke:
            _libero_smoke()
    except SkipVerification as exc:
        print(f"SKIP {exc}")
        return 2
    except VerificationError as exc:
        print(f"FAIL {exc}")
        return 1
    print("environment and tools verification passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
