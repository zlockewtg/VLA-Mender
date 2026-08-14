#!/usr/bin/env python3
"""Execute one observation-only OSC policy over a materialized LIBERO reset bank.

Private MuJoCo state is consumed only by this trusted reset coordinator.  The code
policy is statically audited and executes in the CapX sandbox with only documented
observation APIs; reset payloads, episode identifiers, and evaluator predicates are
never injected into the policy namespace.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
from PIL import Image


DEFAULT_CAPX_ROOT = Path("/mnt/public/tgy/capx-aspire")
DEFAULT_PYTHON = DEFAULT_CAPX_ROOT / ".venv-libero/bin/python"
DEFAULT_ROBOSUITE_ROOT = DEFAULT_CAPX_ROOT / ".runtime/robosuite_capx_1_4_0_prefix_geom"
DEFAULT_SERVICE_PORT_PREFIXES = (141, 142, 143, 144)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_gate(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(
        "vla_mender_observation_only_gate", path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load observation-only gate: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.validate_observation_only_program


def _configure_capx_imports(capx_root: Path) -> None:
    root = str(capx_root.resolve())
    if root not in sys.path:
        sys.path.insert(0, root)


def _audit_program(
    program_path: Path, gate_path: Path, capx_root: Path
) -> dict[str, Any]:
    _configure_capx_imports(capx_root)
    from aspire.execution.policy import validate_program
    from aspire.vla_mender.failure_repair import (
        DEFAULT_REPAIR_DISABLED_APIS,
        validate_seed_specific_program,
    )

    program = program_path.read_text(encoding="utf-8")
    audits = {
        "observation_only": _load_gate(gate_path)(program),
        "hard_boundary": validate_program(program, list(DEFAULT_REPAIR_DISABLED_APIS)),
        "seed_specific": validate_seed_specific_program(program),
    }
    if (
        not audits["observation_only"].get("valid")
        or audits["hard_boundary"]
        or audits["seed_specific"]
    ):
        raise RuntimeError(f"repair program rejected: {audits}")
    return audits


def _episode_metadata(results: dict[str, Any], episode_index: int) -> dict[str, Any]:
    matches = [
        item
        for item in results.get("episodes", [])
        if int(item.get("dataset_episode_index", -1)) == episode_index
    ]
    if len(matches) != 1:
        raise ValueError(
            f"episode {episode_index} has {len(matches)} records in source results"
        )
    record = dict(matches[0])
    if bool(record.get("success")):
        raise ValueError(f"repair job points to successful episode {episode_index}")
    return record


def _source_public_state(
    results_root: Path, episode_index: int, frame_index: int
) -> np.ndarray:
    path = (
        results_root
        / "data"
        / f"chunk-{episode_index // 1000:03d}"
        / f"episode_{episode_index:06d}.parquet"
    )
    table = pq.read_table(path, columns=["state", "frame_index"])
    frame_indices = np.asarray(table["frame_index"].to_numpy(), dtype=np.int64)
    matches = np.flatnonzero(frame_indices == frame_index)
    if len(matches) != 1:
        raise ValueError(
            f"frame {frame_index} has {len(matches)} rows in episode {episode_index}"
        )
    states = table["state"].combine_chunks().to_pylist()
    state = np.asarray(states[int(matches[0])], dtype=np.float64)
    if state.shape != (8,) or not np.isfinite(state).all():
        raise ValueError(
            f"episode {episode_index} frame {frame_index} has invalid public state {state.shape}"
        )
    return state


def _normalized_private_descriptor(
    *,
    reset_bank: Path,
    private_root: Path,
    job: dict[str, Any],
    record: dict[str, Any],
    source_state: np.ndarray,
    sim_signature: dict[str, Any],
    tolerance: float,
) -> dict[str, Any]:
    """Translate reset storage schemas inside the coordinator-only boundary."""

    source_path = (reset_bank / str(job["reset_state"])).resolve()
    expected_root = reset_bank.resolve()
    try:
        source_path.relative_to(expected_root)
    except ValueError as exc:
        raise ValueError(f"reset state escapes reset bank: {source_path}") from exc
    if not source_path.is_file():
        raise FileNotFoundError(source_path)

    descriptor_dir = private_root / str(job["job_id"])
    descriptor_dir.mkdir(parents=True, exist_ok=True)
    os.chmod(descriptor_dir, 0o700)
    normalized_path = descriptor_dir / "state.npz"
    with np.load(source_path, allow_pickle=False) as payload:
        if "sim_state" not in payload.files:
            raise ValueError(f"reset payload has no sim_state: {source_path}")
        simulator_state = np.asarray(payload["sim_state"], dtype=np.float64).reshape(-1)
        gripper = (
            np.asarray(payload["gripper_controller_state"], dtype=np.float64)
            if "gripper_controller_state" in payload.files
            else np.empty((0,), dtype=np.float64)
        )
    np.savez_compressed(
        normalized_path,
        simulator_state=simulator_state,
        gripper_current_action=gripper,
    )
    os.chmod(normalized_path, 0o600)

    episode_index = int(job["episode_index"])
    frame_index = int(job["reset_frame_index"])
    descriptor = {
        "schema_version": 1,
        "dataset_episode_index": episode_index,
        "suite": str(record["suite"]),
        "task_id": int(record["task_id"]),
        "suite_episode_index": int(record["episode_index"]),
        "eval_seed": int(record["scene_model_seed"]),
        "scene_model_seed": int(record["scene_model_seed"]),
        "requested_frame_index": frame_index,
        "restored_frame_index": frame_index,
        "source_state": source_state.tolist(),
        "replay_state": source_state.tolist(),
        "max_abs_observation_error": 0.0,
        "tolerance": float(tolerance),
        "verified": True,
        "sim_signature": sim_signature,
        "private_simulator_state_path": str(normalized_path),
        "privacy": (
            "Coordinator-only exact-reset material; never exposed to repair code."
        ),
    }
    descriptor_path = descriptor_dir / "reset_descriptor.json"
    _write_json(descriptor_path, descriptor)
    os.chmod(descriptor_path, 0o600)
    return descriptor


def _scene_sim_signature(
    *,
    base_config: Path,
    suite: str,
    task_id: int,
    suite_episode_index: int,
    scene_model_seed: int,
    capx_root: Path,
) -> dict[str, Any]:
    """Build the strict compiled-model signature without exposing it to policy code."""

    _configure_capx_imports(capx_root)
    from aspire.execution.replay import build_env_config
    from aspire.vla_mender.failure_repair import (
        _sim_signature,
        ensure_vendored_libero_runtime,
    )
    from capx.envs.configs.instantiate import instantiate

    ensure_vendored_libero_runtime()
    import capx.integrations  # noqa: F401

    env = instantiate(build_env_config(base_config, suite, task_id))
    try:
        env.reset(seed=suite_episode_index + 1)
        low = env.low_level_env
        low.handle.env.seed(scene_model_seed)
        low._current_obs = low.handle.env.reset()
        return _sim_signature(low.handle.env)
    finally:
        env.close()


def _visible_image_audit(environment: Any, expected_path: Path) -> dict[str, Any]:
    observation = environment.low_level_env.get_observation()
    actual = np.asarray(observation["agentview"]["images"]["rgb"], dtype=np.uint8)
    expected = np.asarray(Image.open(expected_path).convert("RGB"), dtype=np.uint8)
    if actual.shape != expected.shape:
        raise RuntimeError(
            f"restored public image shape mismatch: actual={actual.shape}, expected={expected.shape}"
        )
    # The reset materializer stores LIBERO evidence with its historical 180-degree
    # image normalization, while CapX exposes native RGB with only the renderer's
    # vertical correction.  Compare the finite set of lossless axis conventions
    # and record which public convention matched; never inspect scene state.
    conventions = {
        "identity": expected,
        "horizontal_flip": expected[:, ::-1],
        "vertical_flip": expected[::-1],
        "rotate_180": expected[::-1, ::-1],
    }
    errors = {
        name: np.abs(actual.astype(np.int16) - value.astype(np.int16))
        for name, value in conventions.items()
    }
    convention = min(errors, key=lambda name: float(errors[name].mean()))
    absolute = errors[convention]
    return {
        "shape": list(actual.shape),
        "evidence_axis_convention": convention,
        "mean_absolute_error": float(absolute.mean()),
        "max_absolute_error": int(absolute.max()),
        "exact_pixel_fraction": float(np.mean(absolute == 0)),
        "expected_sha256": _sha256(expected_path),
        "actual_array_sha256": hashlib.sha256(actual.tobytes()).hexdigest(),
    }


def _run_job(args: argparse.Namespace, job: dict[str, Any]) -> dict[str, Any]:
    _configure_capx_imports(args.capx_root)
    from aspire.vla_mender.failure_repair import (
        execute_repair_program,
        restore_capx_env_from_replay,
    )

    reset_bank = args.reset_bank.resolve()
    output = args.output.resolve()
    attempts = output / "attempts"
    attempt_dir = attempts / str(job["job_id"])
    result_path = attempt_dir / "repair_result.json"
    manifest_path = attempt_dir / "attempt_manifest.json"
    if result_path.is_file() and manifest_path.is_file():
        manifest = _read_json(manifest_path)
        return {**job, **manifest["outcome"], "resumed": True}
    if attempt_dir.exists():
        raise FileExistsError(f"incomplete immutable attempt exists: {attempt_dir}")

    reset_manifest = _read_json(reset_bank / "reset_bank_manifest.json")
    results_path = Path(str(reset_manifest["source_results"])).resolve()
    results_root = results_path.parent
    results = _read_json(results_path)
    episode_index = int(job["episode_index"])
    frame_index = int(job["reset_frame_index"])
    record = _episode_metadata(results, episode_index)
    source_state = _source_public_state(results_root, episode_index, frame_index)
    sim_signature = _scene_sim_signature(
        base_config=args.base_config,
        suite=str(record["suite"]),
        task_id=int(record["task_id"]),
        suite_episode_index=int(record["episode_index"]),
        scene_model_seed=int(record["scene_model_seed"]),
        capx_root=args.capx_root,
    )
    descriptor = _normalized_private_descriptor(
        reset_bank=reset_bank,
        private_root=output / "private_reset_descriptors",
        job=job,
        record=record,
        source_state=source_state,
        sim_signature=sim_signature,
        tolerance=float(args.transfer_tolerance),
    )
    environment, transfer = restore_capx_env_from_replay(
        descriptor,
        base_config=args.base_config,
        transfer_tolerance=float(args.transfer_tolerance),
    )
    try:
        visible_audit = _visible_image_audit(
            environment, (reset_bank / str(job["agent_view"])).resolve()
        )
        if visible_audit["mean_absolute_error"] > float(args.max_image_mae):
            raise RuntimeError(
                "restored policy-visible image differs from reset evidence: "
                f"MAE={visible_audit['mean_absolute_error']:.6g} exceeds "
                f"{args.max_image_mae:.6g}"
            )
        low = environment.low_level_env
        low.max_steps = int(args.max_steps)
        low.handle.env.env.horizon = int(low.handle.env.env.timestep) + int(
            args.max_steps
        )
        result = execute_repair_program(
            env=environment,
            program=args.program.read_text(encoding="utf-8"),
            replay_report=descriptor,
            failure_mode={
                "mode_id": str(job["failure_mode_id"]),
                "category": str(job["failure_category"]),
            },
            output_dir=attempt_dir,
            save_failed_trajectory=True,
            video_fps=20,
        )
        outcome = {
            "success": bool(result.get("task_completed")),
            "task_completed": bool(result.get("task_completed")),
            "evaluator_task_completed": bool(result.get("evaluator_task_completed")),
            "sandbox_rc": result.get("sandbox_rc"),
            "failure_labels": result.get("failure_labels", []),
            "result_path": str(result_path),
        }
        public_transfer = {
            key: value
            for key, value in transfer.items()
            if key not in {"sim_signature", "scene_signature"}
        }
        _write_json(
            manifest_path,
            {
                "schema_version": 1,
                "job": job,
                "program": str(args.program.resolve()),
                "program_sha256": _sha256(args.program),
                "observation_only_audits": args.audits,
                "private_reset_actor_visible": False,
                "evaluator_truth_actor_visible": False,
                "public_restore_audit": public_transfer,
                "visible_image_audit": visible_audit,
                "outcome": outcome,
            },
        )
        return {**job, **outcome, "resumed": False}
    finally:
        environment.close()


def _single_job_subprocess(
    args: argparse.Namespace, job_id: str, port_prefix: int
) -> dict[str, Any]:
    command = [
        str(args.python),
        "-u",
        str(Path(__file__).resolve()),
        "--reset-bank",
        str(args.reset_bank.resolve()),
        "--output",
        str(args.output.resolve()),
        "--program",
        str(args.program.resolve()),
        "--observation-gate",
        str(args.observation_gate.resolve()),
        "--base-config",
        str(args.base_config.resolve()),
        "--capx-root",
        str(args.capx_root.resolve()),
        "--python",
        str(args.python),
        "--max-steps",
        str(args.max_steps),
        "--transfer-tolerance",
        str(args.transfer_tolerance),
        "--max-image-mae",
        str(args.max_image_mae),
        "--internal-job-id",
        job_id,
        "--service-port-prefix",
        str(port_prefix),
    ]
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(args.capx_root.resolve())
    environment.setdefault("ASPIRE_ROBOSUITE_ROOT", str(DEFAULT_ROBOSUITE_ROOT))
    for key in ("NO_PROXY", "no_proxy"):
        existing = [value for value in environment.get(key, "").split(",") if value]
        environment[key] = ",".join(
            dict.fromkeys(["127.0.0.1", "localhost", *existing])
        )
    completed = subprocess.run(
        command,
        cwd=args.capx_root,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=float(args.job_timeout),
        check=False,
    )
    log_path = args.output.resolve() / "logs" / f"{job_id}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(completed.stdout, encoding="utf-8")
    if completed.returncode != 0:
        return {
            "job_id": job_id,
            "success": False,
            "return_code": int(completed.returncode),
            "failure_labels": ["BATCH_PROCESS_FAILURE"],
            "log_path": str(log_path),
        }
    payload_lines = [
        line
        for line in completed.stdout.splitlines()
        if line.startswith("RESULT_JSON=")
    ]
    if not payload_lines:
        return {
            "job_id": job_id,
            "success": False,
            "return_code": 97,
            "failure_labels": ["MISSING_CHILD_RESULT"],
            "log_path": str(log_path),
        }
    result = json.loads(payload_lines[-1].split("=", 1)[1])
    return {**result, "return_code": 0, "log_path": str(log_path)}


def _worker_lane(
    args: argparse.Namespace, jobs: list[dict[str, Any]], port_prefix: int
) -> list[dict[str, Any]]:
    records = []
    for job in jobs:
        record = _single_job_subprocess(args, str(job["job_id"]), port_prefix)
        records.append(record)
        print(
            f"[repair] job={job['job_id']} success={record.get('success')} "
            f"rc={record.get('return_code')}",
            flush=True,
        )
    return records


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reset-bank", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--program", type=Path, required=True)
    parser.add_argument("--observation-gate", type=Path)
    parser.add_argument("--base-config", type=Path)
    parser.add_argument("--capx-root", type=Path, default=DEFAULT_CAPX_ROOT)
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--candidate-kind",
        choices=("all", "window_start", "window_midpoint"),
        default="all",
    )
    parser.add_argument(
        "--retry-failures-from",
        type=Path,
        help="Run only jobs marked unsuccessful in an earlier batch_summary.json.",
    )
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--job-timeout", type=float, default=1200.0)
    parser.add_argument("--transfer-tolerance", type=float, default=1e-2)
    parser.add_argument("--max-image-mae", type=float, default=2.0)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--internal-job-id")
    parser.add_argument("--service-port-prefix", type=int, default=145)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    args.reset_bank = args.reset_bank.resolve()
    args.output = args.output.resolve()
    args.program = args.program.resolve()
    args.capx_root = args.capx_root.resolve()
    # Preserve the virtual-environment launcher path.  Resolving its symlink can
    # invoke the base interpreter without the venv's site-packages.
    args.python = Path(os.path.abspath(args.python))
    args.observation_gate = (
        args.observation_gate.resolve()
        if args.observation_gate is not None
        else args.program.with_name("observation_only_gate.py")
    )
    args.base_config = (
        args.base_config.resolve()
        if args.base_config is not None
        else args.program.with_name("franka_libero_task0_20hz_osc.yaml")
    )
    for required in (
        args.reset_bank / "repair_jobs.json",
        args.reset_bank / "reset_bank_manifest.json",
        args.program,
        args.observation_gate,
        args.base_config,
        args.python,
    ):
        if not required.is_file():
            raise FileNotFoundError(required)
    if args.workers < 1:
        raise ValueError("--workers must be positive")

    audits = _audit_program(args.program, args.observation_gate, args.capx_root)
    args.audits = audits
    if args.audit_only:
        print(json.dumps(audits, indent=2, ensure_ascii=False))
        return

    jobs = list(_read_json(args.reset_bank / "repair_jobs.json")["jobs"])
    if args.retry_failures_from is not None:
        previous = _read_json(args.retry_failures_from.resolve())
        failed_ids = {
            str(record["job_id"])
            for record in previous.get("records", [])
            if not bool(record.get("success"))
        }
        known_ids = {str(job["job_id"]) for job in jobs}
        if not failed_ids or not failed_ids <= known_ids:
            raise ValueError(
                "retry summary has no failed jobs or references unknown reset jobs"
            )
        jobs = [job for job in jobs if str(job["job_id"]) in failed_ids]
    if args.candidate_kind != "all":
        jobs = [job for job in jobs if job.get("candidate_kind") == args.candidate_kind]
    if args.limit is not None:
        jobs = jobs[: args.limit]
    if args.internal_job_id is not None:
        matches = [job for job in jobs if str(job["job_id"]) == args.internal_job_id]
        if len(matches) != 1:
            raise ValueError(f"unknown or duplicate job id: {args.internal_job_id}")
        os.environ["SAM3_SERVICE_URL"] = (
            f"http://127.0.0.1:{args.service_port_prefix}14"
        )
        os.environ["GRASPNET_SERVICE_URL"] = (
            f"http://127.0.0.1:{args.service_port_prefix}15"
        )
        os.environ["PYROKI_SERVICE_URL"] = (
            f"http://127.0.0.1:{args.service_port_prefix}16"
        )
        result = _run_job(args, matches[0])
        print("RESULT_JSON=" + json.dumps(result, ensure_ascii=False), flush=True)
        return

    args.output.mkdir(parents=True, exist_ok=True)
    private_root = args.output / "private_reset_descriptors"
    private_root.mkdir(parents=True, exist_ok=True)
    os.chmod(private_root, 0o700)
    campaign = {
        "schema_version": 1,
        "reset_bank": str(args.reset_bank),
        "program": str(args.program),
        "program_sha256": _sha256(args.program),
        "observation_gate": str(args.observation_gate),
        "observation_gate_sha256": _sha256(args.observation_gate),
        "base_config": str(args.base_config),
        "job_count": len(jobs),
        "candidate_kind": args.candidate_kind,
        "observation_only_audits": audits,
        "policy_private_reset_access": False,
        "policy_evaluator_truth_access": False,
        "private_reset_use": "trusted coordinator exact restore only",
    }
    campaign_path = args.output / "campaign_manifest.json"
    if campaign_path.exists():
        previous = _read_json(campaign_path)
        if previous != campaign:
            raise ValueError("output campaign manifest does not match requested run")
    else:
        _write_json(campaign_path, campaign)

    worker_count = min(args.workers, len(DEFAULT_SERVICE_PORT_PREFIXES), len(jobs))
    lanes = [jobs[index::worker_count] for index in range(worker_count)]
    prefixes = DEFAULT_SERVICE_PORT_PREFIXES[:worker_count]
    records: list[dict[str, Any]] = []
    if worker_count == 1:
        records.extend(_worker_lane(args, lanes[0], prefixes[0]))
    else:
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=worker_count
        ) as executor:
            futures = [
                executor.submit(_worker_lane, args, lane, prefix)
                for lane, prefix in zip(lanes, prefixes, strict=True)
            ]
            for future in concurrent.futures.as_completed(futures):
                records.extend(future.result())
    records.sort(key=lambda item: str(item["job_id"]))
    success_count = sum(bool(item.get("success")) for item in records)
    summary = {
        "schema_version": 1,
        "expected_job_count": len(jobs),
        "attempted_count": len(records),
        "success_count": success_count,
        "failure_count": len(records) - success_count,
        "complete": len(records) == len(jobs),
        "all_successful": len(records) == len(jobs) and success_count == len(jobs),
        "records": records,
    }
    _write_json(args.output / "batch_summary.json", summary)
    print(
        json.dumps({key: summary[key] for key in summary if key != "records"}, indent=2)
    )


if __name__ == "__main__":
    main()
