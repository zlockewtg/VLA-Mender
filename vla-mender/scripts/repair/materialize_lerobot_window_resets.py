#!/usr/bin/env python3
"""Replay LeRobot LIBERO failures and materialize start/midpoint reset states."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
import pyarrow.parquet as pq

SOURCE_ROOT = Path(__file__).resolve().parents[2] / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from workflow.failure_diagnosis.failure_diagnosis import _restore_gripper  # noqa: E402
from workflow.libero_runtime import (  # noqa: E402
    PUBLIC_REPLAY_TOLERANCE,
    SIM_STATE_TOLERANCE,
    LiberoRuntime,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--initial-state-manifest", type=Path, required=True)
    parser.add_argument("--diagnosis", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--reset-dynamics",
        choices=("preserve_full_state", "quiescent_osc"),
        default="preserve_full_state",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def sha256_array(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).view(np.uint8)).hexdigest()


def load_trajectory(
    results_root: Path, episode_index: int
) -> tuple[np.ndarray, np.ndarray]:
    path = (
        results_root
        / "data"
        / f"chunk-{episode_index // 1000:03d}"
        / f"episode_{episode_index:06d}.parquet"
    )
    table = pq.read_table(path, columns=["state", "actions", "frame_index"])
    frame_indices = np.asarray(table["frame_index"].to_numpy(), dtype=np.int64)
    if not np.array_equal(frame_indices, np.arange(len(frame_indices), dtype=np.int64)):
        raise ValueError(f"episode {episode_index} frame indices are not contiguous")
    states = np.asarray(table["state"].combine_chunks().to_pylist(), dtype=np.float64)
    actions = np.asarray(
        table["actions"].combine_chunks().to_pylist(), dtype=np.float32
    )
    if states.shape != (len(frame_indices), 8) or actions.shape != (
        len(frame_indices),
        7,
    ):
        raise ValueError(
            f"episode {episode_index} has unexpected trajectory shapes "
            f"states={states.shape}, actions={actions.shape}"
        )
    if not np.isfinite(states).all() or not np.isfinite(actions).all():
        raise ValueError(
            f"episode {episode_index} contains non-finite trajectory values"
        )
    return states, actions


def observation_after_restore(env: Any, sim_state: np.ndarray) -> dict[str, Any]:
    if not hasattr(env, "regenerate_obs_from_state"):
        raise RuntimeError(
            "LIBERO environment does not expose regenerate_obs_from_state"
        )
    obs = env.regenerate_obs_from_state(sim_state)
    if not isinstance(obs, dict):
        raise RuntimeError(
            "regenerate_obs_from_state did not return an observation mapping"
        )
    return obs


def main() -> None:
    args = parse_args()
    results_root = args.results_root.resolve()
    state_manifest_path = args.initial_state_manifest.resolve()
    diagnosis_path = args.diagnosis.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to replace existing reset bank: {output}")

    results = read_json(results_root / "results.json")
    state_manifest = read_json(state_manifest_path)
    diagnosis = read_json(diagnosis_path)
    control_frequency = int(results["control_freq"])
    num_steps_wait = int(results["num_steps_wait"])
    if control_frequency != int(state_manifest["control_frequency"]):
        raise ValueError("rollout and initial-state control frequencies differ")
    if diagnosis.get("task") != state_manifest.get("task"):
        raise ValueError("diagnosis and initial-state tasks differ")

    state_file = (
        state_manifest_path.parent / str(state_manifest["state_file"])
    ).resolve()
    initial_states = np.load(state_file, allow_pickle=False)
    result_by_episode = {
        int(item["dataset_episode_index"]): item for item in results["episodes"]
    }
    failures = {
        index for index, item in result_by_episode.items() if not bool(item["success"])
    }
    diagnosed = {int(item["episode_index"]) for item in diagnosis["episodes"]}
    if failures != diagnosed:
        raise ValueError(
            f"failure coverage mismatch: rollout={sorted(failures)}, diagnosis={sorted(diagnosed)}"
        )

    mode_by_id = {str(item["id"]): item for item in diagnosis["failure_modes"]}
    candidates: list[dict[str, Any]] = []
    for item in sorted(
        diagnosis["episodes"], key=lambda value: int(value["episode_index"])
    ):
        start, stop = (int(value) for value in item["failure_window"])
        midpoint = (start + stop) // 2
        if not (0 <= start < midpoint <= stop):
            raise ValueError(
                f"invalid start/midpoint selection for episode {item['episode_index']}"
            )
        for rank, (kind, frame) in enumerate(
            (("window_start", start), ("window_midpoint", midpoint))
        ):
            candidates.append(
                {
                    "episode_index": int(item["episode_index"]),
                    "initial_state_index": int(item["initial_state_index"]),
                    "candidate_rank": rank,
                    "candidate_kind": kind,
                    "requested_frame_index": frame,
                    "window_start": start,
                    "window_stop": stop,
                    "failure_mode_id": str(item["failure_mode"]),
                    "causal_frame": int(item["causal_frame"]),
                }
            )

    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    reports: list[dict[str, Any]] = []
    bank_sim_states: list[np.ndarray] = []
    bank_gripper_states: list[np.ndarray] = []
    try:
        runtime = LiberoRuntime(
            str(state_manifest["suite"]),
            int(state_manifest["task_id"]),
            "osc",
            control_frequency,
        )
        candidates_by_episode: dict[int, list[dict[str, Any]]] = {}
        for candidate in candidates:
            candidates_by_episode.setdefault(candidate["episode_index"], []).append(
                candidate
            )

        for episode_index, episode_candidates in candidates_by_episode.items():
            result = result_by_episode[episode_index]
            initial_state_index = int(result["initial_state_index"])
            if initial_state_index != int(episode_candidates[0]["initial_state_index"]):
                raise ValueError(
                    f"episode {episode_index} initial-state index mismatch"
                )
            states, actions = load_trajectory(results_root, episode_index)
            selected_frames = {
                int(item["requested_frame_index"]) for item in episode_candidates
            }
            if max(selected_frames) >= len(states):
                raise ValueError(
                    f"episode {episode_index} reset frame exceeds trajectory"
                )
            scene_seed = int(result["scene_model_seed"])
            env = runtime.new_env(scene_seed)
            captured: dict[
                int, tuple[np.ndarray, np.ndarray | None, np.ndarray, float]
            ] = {}
            try:
                obs = env.set_init_state(initial_states[initial_state_index])
                for _ in range(num_steps_wait):
                    obs, _, done, _ = env.step(runtime.neutral_action(env).tolist())
                    if done:
                        raise RuntimeError(
                            f"episode {episode_index} terminated during replay stabilization"
                        )
                initial_error = float(
                    np.max(np.abs(runtime.public_state(obs) - states[0]))
                )
                if initial_error > PUBLIC_REPLAY_TOLERANCE:
                    raise RuntimeError(
                        f"episode {episode_index} initial replay error {initial_error} exceeds "
                        f"{PUBLIC_REPLAY_TOLERANCE}"
                    )
                max_error = initial_error
                for frame_index in range(max(selected_frames) + 1):
                    if frame_index in selected_frames:
                        agent_view, _ = runtime.observation_images(obs)
                        captured[frame_index] = (
                            np.asarray(env.get_sim_state(), dtype=np.float64).copy(),
                            runtime.capture_gripper_controller(env),
                            agent_view,
                            max_error,
                        )
                    if frame_index == max(selected_frames):
                        break
                    obs, _, done, _ = env.step(actions[frame_index].tolist())
                    if done:
                        raise RuntimeError(
                            f"episode {episode_index} ended before reset frame "
                            f"{max(selected_frames)}"
                        )
                    error = float(
                        np.max(
                            np.abs(runtime.public_state(obs) - states[frame_index + 1])
                        )
                    )
                    max_error = max(max_error, error)
                    if error > PUBLIC_REPLAY_TOLERANCE:
                        raise RuntimeError(
                            f"episode {episode_index} replay diverged at frame "
                            f"{frame_index + 1}: {error}"
                        )
            finally:
                env.close()

            for candidate in episode_candidates:
                frame_index = int(candidate["requested_frame_index"])
                sim_state, gripper_state, agent_view, prefix_max_error = captured[
                    frame_index
                ]
                state_name = f"episode_{episode_index:06d}_frame_{frame_index:06d}.npz"
                image_name = f"episode_{episode_index:06d}_frame_{frame_index:06d}.png"
                private_path = temporary / "private_reset_states" / state_name
                image_path = temporary / "agent_views" / image_name
                image_path.parent.mkdir(parents=True, exist_ok=True)
                imageio.imwrite(image_path, agent_view)

                target_env = runtime.new_env(scene_seed)
                try:
                    target_env.set_init_state(initial_states[initial_state_index])
                    runtime.set_sim_state(target_env, sim_state)
                    _restore_gripper(target_env, gripper_state)
                    dynamics_audit = runtime.apply_reset_dynamics(
                        target_env, args.reset_dynamics
                    )
                    restored_state = np.asarray(
                        target_env.get_sim_state(), dtype=np.float64
                    ).copy()
                    restored_gripper = runtime.capture_gripper_controller(target_env)
                    restored_obs = observation_after_restore(target_env, restored_state)
                    restored_public_error = float(
                        np.max(
                            np.abs(
                                runtime.public_state(restored_obs) - states[frame_index]
                            )
                        )
                    )
                    state_restore_error = float(
                        np.max(np.abs(restored_state - sim_state))
                    )
                    if args.reset_dynamics == "preserve_full_state" and (
                        state_restore_error > SIM_STATE_TOLERANCE
                    ):
                        raise RuntimeError(
                            f"episode {episode_index} frame {frame_index} sim-state "
                            f"restore error {state_restore_error} exceeds {SIM_STATE_TOLERANCE}"
                        )
                    runtime.write_private_state(
                        private_path,
                        sim_state=restored_state,
                        gripper_state=restored_gripper,
                    )
                finally:
                    target_env.close()

                with np.load(private_path, allow_pickle=False) as payload:
                    saved_state = np.asarray(payload["sim_state"], dtype=np.float64)
                    saved_gripper = np.asarray(
                        payload.get("gripper_controller_state", np.empty((0,))),
                        dtype=np.float64,
                    )
                mode = mode_by_id[candidate["failure_mode_id"]]
                report = {
                    **candidate,
                    "verified": True,
                    "replayed_action_count": frame_index,
                    "max_public_state_replay_error": prefix_max_error,
                    "restored_public_state_error": restored_public_error,
                    "restored_public_state_within_replay_tolerance": bool(
                        restored_public_error <= PUBLIC_REPLAY_TOLERANCE
                    ),
                    "sim_state_restore_error": state_restore_error,
                    "public_tolerance": PUBLIC_REPLAY_TOLERANCE,
                    "sim_state_tolerance": SIM_STATE_TOLERANCE,
                    "source_control_space": "osc",
                    "target_control_space": "osc",
                    "reset_dynamics": args.reset_dynamics,
                    "dynamics_audit": dynamics_audit,
                    "private_state": f"private_reset_states/{state_name}",
                    "private_state_sha256": sha256_array(saved_state),
                    "agent_view": f"agent_views/{image_name}",
                    "agent_view_sha256": sha256_array(agent_view),
                    "failure_mode_label": str(mode["label"]),
                    "failure_category": str(mode["category"]),
                }
                reports.append(report)
                bank_sim_states.append(saved_state)
                bank_gripper_states.append(saved_gripper)

        order = sorted(
            range(len(reports)),
            key=lambda index: (
                int(reports[index]["episode_index"]),
                int(reports[index]["candidate_rank"]),
            ),
        )
        reports = [reports[index] for index in order]
        bank_sim_states = [bank_sim_states[index] for index in order]
        bank_gripper_states = [bank_gripper_states[index] for index in order]
        combined_path = temporary / "reset_state_bank.npz"
        np.savez_compressed(
            combined_path,
            sim_states=np.stack(bank_sim_states),
            gripper_controller_states=np.stack(bank_gripper_states),
            episode_indices=np.asarray(
                [item["episode_index"] for item in reports], dtype=np.int64
            ),
            frame_indices=np.asarray(
                [item["requested_frame_index"] for item in reports], dtype=np.int64
            ),
            candidate_kinds=np.asarray(
                [item["candidate_kind"] for item in reports], dtype="<U32"
            ),
        )
        combined_sha256 = hashlib.sha256(combined_path.read_bytes()).hexdigest()
        manifest = {
            "schema_version": 1,
            "selection": "two frames per failed episode: inclusive window start and floor midpoint",
            "midpoint_formula": "floor((window_start + window_stop) / 2)",
            "trajectory_state_timing": "pre_action",
            "count": len(reports),
            "episode_count": len(candidates_by_episode),
            "control_frequency_hz": control_frequency,
            "num_steps_wait": num_steps_wait,
            "reset_dynamics": args.reset_dynamics,
            "source_results": str(results_root / "results.json"),
            "source_initial_state_manifest": str(state_manifest_path),
            "source_diagnosis": str(diagnosis_path),
            "combined_state_bank": "reset_state_bank.npz",
            "combined_state_bank_sha256": combined_sha256,
            "resets": reports,
        }
        jobs = {
            "schema_version": 1,
            "jobs": [
                {
                    "job_id": (
                        f"e{item['episode_index']:06d}-"
                        f"{item['candidate_kind']}-f{item['requested_frame_index']:06d}"
                    ),
                    "episode_index": item["episode_index"],
                    "reset_frame_index": item["requested_frame_index"],
                    "candidate_kind": item["candidate_kind"],
                    "reset_state": item["private_state"],
                    "agent_view": item["agent_view"],
                    "failure_mode_id": item["failure_mode_id"],
                    "failure_mode": item["failure_mode_label"],
                    "failure_category": item["failure_category"],
                    "target_control_space": item["target_control_space"],
                    "reset_dynamics": item["reset_dynamics"],
                }
                for item in reports
            ],
        }
        LiberoRuntime.write_json(temporary / "reset_bank_manifest.json", manifest)
        LiberoRuntime.write_json(
            temporary / "replay_verification.json",
            {"schema_version": 1, "all_verified": True, "reports": reports},
        )
        LiberoRuntime.write_json(temporary / "repair_jobs.json", jobs)
        temporary.replace(output)
        print(
            json.dumps(
                {
                    "output": str(output),
                    "episodes": len(candidates_by_episode),
                    "reset_states": len(reports),
                    "combined_state_bank": str(output / "reset_state_bank.npz"),
                    "all_verified": True,
                },
                indent=2,
            )
        )
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


if __name__ == "__main__":
    main()
