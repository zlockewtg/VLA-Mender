"""Agent handoff, failure windows, prefix replay and reset-bank materialization."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from ..libero_runtime import LiberoRuntime, PUBLIC_REPLAY_TOLERANCE
from ..parameters import ExperimentSettings
from ..trajectory_protocol import (
    diagnosis_evidence_metadata,
    validate_episode,
    validate_rollout_contract,
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _validate_materialization_rollout_contract(
    settings: ExperimentSettings, rollout: Path, summary: dict[str, Any]
) -> tuple[str, bool]:
    """Validate an exact rollout or a frozen rollout reused across reset-only changes."""

    requested_fingerprint = settings.fingerprint()
    source_fingerprint = str(summary.get("settings_fingerprint", ""))
    if source_fingerprint == requested_fingerprint:
        validate_rollout_contract(summary, requested_fingerprint)
        return source_fingerprint, False

    validate_rollout_contract(summary)
    run_manifest_path = rollout.parent / "run_manifest.json"
    if not run_manifest_path.is_file():
        raise ValueError(
            "rollout settings fingerprint differs from the materialization contract "
            f"and source run_manifest.json is missing: {run_manifest_path}"
        )
    run_manifest = _read_json(run_manifest_path)
    if str(run_manifest.get("settings_fingerprint", "")) != source_fingerprint:
        raise ValueError("rollout summary/run manifest fingerprint mismatch")
    source_settings = run_manifest.get("requested_settings")
    if not isinstance(source_settings, dict):
        raise ValueError("source run manifest lacks requested_settings")

    # Reset selection and reset dynamics do not affect the already-frozen
    # policy trajectory. Every rollout-affecting field must remain identical.
    source_rollout_contract = dict(source_settings)
    requested_rollout_contract = settings.as_dict()
    source_rollout_contract.pop("reset", None)
    requested_rollout_contract.pop("reset", None)
    if source_rollout_contract != requested_rollout_contract:
        raise ValueError(
            "frozen rollout may be reused only when reset settings changed; "
            "rollout-affecting contract drift was detected"
        )
    return source_fingerprint, True


def _episode_path(rollout_dir: Path, episode_index: int) -> Path:
    return rollout_dir / "episodes" / f"episode_{episode_index:06d}.json"


def _reset_prompt_values(settings: ExperimentSettings) -> dict[str, str]:
    selection = settings.reset.candidate_selection
    stage_discovery = ""
    if selection == "per_episode_stage_entry_only":
        selection_contract = "exactly 1 (`stage_entry`)"
        prevention_contract = ""
        policy = """This command selects one independently observed behavior-stage
entry for each failed trajectory:
`stage_entry = intervention_stage_start_frame_index`. It replays that episode's
exact action prefix, verifies public-state agreement, applies
`{{RESET_DYNAMICS}}`, and atomically publishes one final handoff directory. The
diagnosis agent does not access or manually select private reset payloads.

First infer a task-specific observable behavior-stage graph from the task
instruction and multiple successful trajectories. Do not assume a fixed stage
vocabulary or copy a decomposition from another task. Then inspect each failed
episode independently, localize candidate change-points from its public state
and executed actions, and confirm the boundary in original-resolution
neighboring frames from every available camera.

The diagnosis must record the preceding stage, successful references, inspected
frames, per-camera evidence paths, observable and state/action transitions,
persistence evidence, and why an earlier or later boundary is wrong. Do not
infer a boundary from another episode or from a fixed offset. Do not materialize
`pre_causal`, `window_stop`, `pre_window`, or an interior frame."""
        stage_discovery = r"""### Per-episode behavior-stage discovery and intervention entry

For each task, infer a task-specific observable behavior-stage graph before
selecting reset states. Do not assume a fixed stage vocabulary, fixed frame
number, fixed temporal offset, or a stage decomposition copied from another
task.

#### A. Build the observable stage graph

Use the task instruction and multiple successful trajectories to identify the
ordered behavior stages needed to complete the task.

A stage boundary is an observable change in one or more of:

- the immediate control objective;
- the task entity currently being approached, engaged, or manipulated;
- the contact or attachment relationship;
- the dominant end-effector or object motion regime;
- the gripper or actuator mode;
- the manipulated object's task-relevant state;
- the prerequisite that has just become satisfied.

For every inferred stage, record `stage_name`, `observable_entry_condition`,
`observable_exit_condition`, `required_prerequisites`, `relevant_entities`, and
`supporting_successful_episode_indices` in the top-level
`observable_stage_graph` list.

Use the most detailed stage decomposition supported by the available evidence.
Do not merge distinct approach, alignment, engagement, manipulation, transfer,
placement, release, or recovery behaviors merely because they occur close
together in time.

#### B. Determine the failed stage

For each failed episode:

1. Compare it with phase-aligned successful trajectories.
2. Verify the prerequisites chronologically.
3. Find the earliest task transition whose expected result is no longer
   supported by public evidence.
4. Assign the behavior stage containing that transition as
   `intervention_stage`.

Do not label a downstream stage as failed when an earlier prerequisite was
never observably completed.

#### C. Locate that episode's stage entry

Inspect the failed episode independently. Never reuse a frame number or offset
from another episode.

First use public state and executed actions to locate candidate behavior
change-points. Then inspect original-resolution neighboring frames from every
available camera around each candidate.

Frame `t` is the stage-entry state only when:

1. the preceding stage is still complete or valid at frame `t`;
2. the action/state transition beginning at frame `t` initiates behavior
   specific to the new stage;
3. the new behavior continues sufficiently to distinguish a real phase
   transition from noise, hesitation, or a transient command;
4. the stage's required entities and interaction state are visibly consistent
   with successful references;
5. `t` precedes the first causal failure frame.

Choose the earliest frame satisfying all five conditions. Do not choose a
generic trajectory start, the middle of an already active stage, the clearest
consequence of the failure, a matching index from another trajectory, or a
boundary inferred from robot motion without confirming the relevant entity and
interaction state.

#### D. Required evidence

For every failed episode, record the inferred intervention stage, selected
stage-entry frame, immediately preceding stage, successful reference episodes,
all inspected neighboring frames, original-resolution evidence paths for every
camera, the observable boundary transition, supporting public state/action
transition, persistence evidence, why the preceding frame is too early, and why
the next plausible candidate would be too late.

Store these in `stage_entry_evidence` using `preceding_stage`,
`inspected_frame_indices`, `camera_evidence`, `observable_transition`,
`state_action_transition`, `persistence_evidence`,
`successful_reference_episode_indices`, `why_previous_frame_is_too_early`, and
`why_later_frame_is_too_late`. A combined `contact_sheet` may additionally be
recorded for convenient review.

If the evidence cannot resolve the boundary, do not guess or materialize a
reset. Mark the episode as ambiguous and acquire denser public evidence.

#### E. Output convention

`intervention_stage_start_frame_index = t` means frame `t` is the saved state
immediately before the action/state transition that initiates the selected
stage. Require
`0 <= intervention_stage_start_frame_index < first_causal_frame_index`.
"""
        selection_record = {
            "method": "per-episode observed behavior stage entry only",
            "intervention_points": ["stage_entry"],
            "frames_per_failure": 1,
        }
        completion = (
            "`summary.failure_episode_count` matches the validated diagnosis and "
            "every diagnosed failure produced exactly one reset at its own "
            "frame-inspected behavior-stage entry."
        )
    elif selection == "failed_stage_entry_only":
        selection_contract = "exactly 1 (`stage_entry`)"
        prevention_contract = ""
        policy = """This command selects only the entry state of the broad
behavior stage in which the failure develops:
`stage_entry = intervention_stage_start_frame_index`. It replays that action
prefix, verifies public-state agreement, applies `{{RESET_DYNAMICS}}`, and
atomically publishes one final handoff directory. The diagnosis agent does not
access or manually select private reset payloads.

Every failed trajectory contributes exactly one intervention point. First use
successful trajectories and both camera views to identify the failed behavior
stage, then move the reset back to the entry of that stage. For this task,
`pick` includes target approach, alignment, and grasp acquisition; its entry is
the first recorded policy frame after simulator stabilization. `transport`
begins once pick is observably successful and the object is securely retained
as lift/carry starts; it includes basket-directed carrying and alignment before
release or containment. Thus an off-center basket approach or rim contact is a
transport-stage failure and resets to transport entry, not to the later contact
frame. Do not materialize `pre_causal`, `window_start`, `window_stop`,
`pre_window`, or an interior frame."""
        selection_record = {
            "method": "failed behavior stage entry only",
            "intervention_points": ["stage_entry"],
            "frames_per_failure": 1,
        }
        completion = (
            "`summary.failure_episode_count` matches the validated diagnosis and "
            "every diagnosed failure produced exactly one reset entry at its "
            "declared failed behavior stage entry."
        )
    elif selection == "pre_causal_only":
        selection_contract = "exactly 1 (`pre_causal`)"
        prevention_contract = ""
        policy = """This command selects only the last normal state immediately
before the first observable abnormal transition:
`pre_causal = first_causal_frame_index - 1`. It replays that action prefix,
verifies public-state agreement, applies `{{RESET_DYNAMICS}}`, and atomically
publishes one final handoff directory. The diagnosis agent does not access or
manually select private reset payloads.

Every failed trajectory contributes exactly one intervention point. The
diagnosis must establish `first_causal_frame_index` by phase-aligning both
camera views with explicitly cited successful trajectories and locating the
first failed transition that departs from those references. It must also record
the successful episode indices and a concrete comparison for every failure.
Do not substitute a nominal subtask boundary, a fixed lead, or a later visible
consequence. Do not materialize `window_start`, `window_stop`, `pre_window`, or
any interior frame; materialize only the immediately preceding normal frame."""
        selection_record = {
            "method": "successful-reference-aligned pre-causal state only",
            "intervention_points": ["pre_causal"],
            "frames_per_failure": 1,
        }
        completion = (
            "`summary.failure_episode_count` matches the validated diagnosis and "
            "every diagnosed failure produced exactly one reset entry at "
            "`first_causal_frame_index - 1`."
        )
    elif selection == "window_start_only":
        selection_contract = "exactly 1 (`window_start`)"
        prevention_contract = ""
        policy = """This command selects only the failure window's
`recoverable_window_start_frame_index`, replays that action prefix, verifies
public-state agreement, applies `{{RESET_DYNAMICS}}`, and atomically publishes
one final handoff directory. The diagnosis agent does not access or manually
select private reset payloads.

Every failed trajectory contributes exactly one intervention point:
`window_start = recoverable_window_start_frame_index`. It is the earliest
directly correctable state inside the semantic subtask phase that fails, before
the diagnosed erroneous transition. Do not materialize `pre_window`,
`window_stop`, or any interior frame. The diagnosis must still record and
validate `window_stop == first_causal_frame_index`; only reset selection changes."""
        selection_record = {
            "method": "failure window start only",
            "intervention_points": ["window_start"],
            "frames_per_failure": 1,
        }
        completion = (
            "`summary.failure_episode_count` matches the validated diagnosis and "
            "every diagnosed failure produced exactly one reset entry at its "
            "window start."
        )
    elif selection == "window_endpoints":
        selection_contract = "exactly 2 (`window_start`, `window_stop`)"
        prevention_contract = ""
        policy = """This command selects both failure-window endpoints, replays
both selected action prefixes, verifies public-state agreement, applies
`{{RESET_DYNAMICS}}`, and atomically publishes one final handoff directory. The
diagnosis agent does not access or manually select private reset payloads.

Every failed trajectory contributes exactly two intervention points: its
`recoverable_window_start_frame_index` and
`recoverable_window_stop_frame_index`. No interior stride sampling is allowed.
The start endpoint tests repair from the earliest correctable state in the
failed subtask phase; the stop endpoint tests intervention at the first
observable erroneous transition."""
        selection_record = {
            "method": "failure window endpoints",
            "intervention_points": ["window_start", "window_stop"],
            "frames_per_failure": 2,
        }
        completion = (
            "`summary.failure_episode_count` matches the validated diagnosis and "
            "every diagnosed failure produced exactly two reset entries: one at "
            "its window start and one at its window stop."
        )
    else:
        steps = settings.reset.prevention_steps
        selection_contract = "exactly 3 (`pre_window`, `window_start`, `window_stop`)"
        prevention_contract = (
            f"- Preventive lead: `{steps}` policy steps before `window_start`"
        )
        policy = f"""This command selects one preventive point before the failure
window plus both failure-window endpoints, replays all three selected action
prefixes, verifies public-state agreement, applies `{{RESET_DYNAMICS}}`, and
atomically publishes one final handoff directory. The diagnosis agent does not
access or manually select private reset payloads.

Every failed trajectory contributes exactly three intervention points:

1. `pre_window = recoverable_window_start_frame_index - {steps}`, exactly
   {steps} policy steps before and strictly outside the diagnosed failure window;
2. `window_start = recoverable_window_start_frame_index`;
3. `window_stop = recoverable_window_stop_frame_index`.

Do not clamp or silently shift `pre_window`. If an episode cannot provide the
exact {steps}-step lead, materialization must fail. No interior stride sampling
is allowed. The preventive point tests proactive intervention, the start
endpoint tests repair from the earliest correctable state in the failed subtask
phase, and the stop endpoint tests intervention at the first observable
erroneous transition. The preventive point is intentionally outside the
same-phase failure-window constraint; the diagnosed window itself must continue
to satisfy every Stage 3 requirement unchanged."""
        selection_record = {
            "method": "pre-window prevention plus failure window endpoints",
            "intervention_points": ["pre_window", "window_start", "window_stop"],
            "prevention_steps": steps,
            "frames_per_failure": 3,
        }
        completion = (
            "`summary.failure_episode_count` matches the validated diagnosis and "
            f"every diagnosed failure produced exactly three reset entries: one "
            f"exactly {steps} steps before its window start, one at its window "
            "start, and one at its window stop."
        )
    selection_json = "  " + json.dumps(
        selection_record, indent=2, ensure_ascii=False
    ).replace("\n", "\n  ") + ","
    return {
        "{{RESET_CANDIDATE_CONTRACT}}": selection_contract,
        "{{RESET_PREVENTION_CONTRACT}}": prevention_contract,
        "{{RESET_MATERIALIZATION_POLICY}}": policy.replace(
            "{{RESET_DYNAMICS}}", settings.reset.dynamics
        ),
        "{{PER_EPISODE_STAGE_DISCOVERY}}": stage_discovery,
        "{{RESET_SELECTION_JSON}}": selection_json,
        "{{RESET_COMPLETION_REQUIREMENT}}": completion,
    }


def _render_prompt(
    settings: ExperimentSettings, *, run_root: Path, settings_path: Path
) -> str:
    """Render the checked-in task-level prompt with the resolved contract."""

    template_candidates = (
        Path(__file__).resolve().parents[2] / "run" / "pre_repair" / "prompt.md",
        Path(__file__).with_name("prompt.md"),
    )
    template_path = next((path for path in template_candidates if path.is_file()), None)
    if template_path is None:
        raise FileNotFoundError("pre-repair prompt template not found")
    template = template_path.read_text(encoding="utf-8")
    values = {
        "{{SUITE}}": settings.task.suite,
        "{{TASK_ID}}": str(settings.task.task_id),
        "{{TASK_DESCRIPTION}}": settings.task.task_description
        or "(use the task instruction from the rollout)",
        "{{CHECKPOINT}}": str(settings.task.checkpoint),
        "{{RUNTIME_BACKEND}}": settings.backend.name,
        "{{OPENPI_COMMIT}}": settings.backend.openpi_commit or "(vendored pin)",
        "{{OPENPI_ENVIRONMENT}}": str(
            settings.backend.openpi_environment or Path(sys.prefix).resolve()
        ),
        "{{LIBERO_ROOT}}": str(settings.backend.libero_root or "(LIBERO default)"),
        "{{TRAJECTORY_PROTOCOL}}": "vla-mender.libero.openpi/v2",
        "{{STATE_PROVIDER}}": settings.initial_states.provider,
        "{{STATE_COUNT}}": str(settings.initial_states.count),
        "{{STATE_MANIFEST}}": str(settings.initial_states.state_manifest or "(none)"),
        "{{CONTROL_FREQUENCY_HZ}}": str(settings.rollout.control_frequency_hz),
        "{{MAX_STEPS}}": str(settings.rollout.max_steps),
        "{{POLICY_SEED}}": str(settings.rollout.policy_seed),
        "{{GPUS}}": ", ".join(str(gpu) for gpu in settings.rollout.gpus),
        "{{WORKERS_PER_GPU}}": str(settings.rollout.workers_per_gpu),
        "{{ACTION_CHUNK}}": str(settings.rollout.action_chunk),
        "{{INFERENCE_STEPS}}": str(settings.rollout.inference_steps),
        "{{NUM_STEPS_WAIT}}": str(settings.rollout.num_steps_wait),
        "{{BINARY_GRIPPER}}": str(settings.rollout.binary_gripper).lower(),
        "{{GRIPPER_HYSTERESIS_THRESHOLD}}": str(
            settings.rollout.gripper_hysteresis_threshold
        ),
        "{{SOURCE_CONTROL_SPACE}}": settings.controller.source_control_space,
        "{{TARGET_CONTROL_SPACE}}": settings.controller.target_control_space,
        "{{RESET_DYNAMICS}}": settings.reset.dynamics,
        "{{RESET_CANDIDATE_SELECTION}}": settings.reset.candidate_selection,
        "{{PREVENTION_STEPS}}": str(settings.reset.prevention_steps),
        "{{OUTPUT_DIR}}": str(run_root),
        "{{SETTINGS_PATH}}": str(settings_path),
        **_reset_prompt_values(settings),
    }
    for placeholder, value in values.items():
        template = template.replace(placeholder, value)
    unresolved = sorted(set(re.findall(r"\{\{[^{}]+\}\}", template)))
    if unresolved:
        raise ValueError(f"unresolved prompt placeholders: {unresolved}")
    return template


def write_task_prompt(
    settings: ExperimentSettings,
    output_dir: str | Path,
    filename: str = "prompt.md",
    *,
    run_root: str | Path | None = None,
    settings_path: str | Path | None = None,
) -> Path:
    """Write the complete config-rendered pre-repair prompt without a rollout."""

    output = Path(output_dir).resolve()
    root = Path(run_root).resolve() if run_root is not None else output
    source = (
        Path(settings_path).resolve()
        if settings_path is not None
        else root / "experiment.resolved.yaml"
    )
    output.mkdir(parents=True, exist_ok=True)
    destination = output / filename
    destination.write_text(
        _render_prompt(settings, run_root=root, settings_path=source),
        encoding="utf-8",
    )
    return destination


def build_agent_prompt(
    rollout_dir: str | Path,
    output_dir: str | Path,
    settings: ExperimentSettings | None = None,
) -> dict[str, Any]:
    """Export only policy-visible evidence and a vendor-neutral agent prompt."""

    rollout = Path(rollout_dir).resolve()
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    summary = _read_json(rollout / "summary.json")
    if settings is None:
        from ..parameters import load_settings

        resolved = rollout.parent / "experiment.resolved.yaml"
        if not resolved.is_file():
            resolved = rollout.parent.parent / "experiment.resolved.yaml"
        if not resolved.is_file():
            raise ValueError(
                "settings are required unless experiment.resolved.yaml exists"
            )
        settings = load_settings(resolved)
    _validate_materialization_rollout_contract(settings, rollout, summary)
    failures: list[dict[str, Any]] = []
    successes: list[dict[str, Any]] = []
    for record in summary.get("episodes", []):
        index = int(record["episode_index"])
        episode = _read_json(_episode_path(rollout, index))
        validate_episode(episode)
        states = np.asarray(episode["states"], dtype=np.float64)
        actions = np.asarray(episode["actions"], dtype=np.float64)
        count = len(states)
        selected = sorted(
            set(np.linspace(0, max(0, count - 1), min(24, count), dtype=int).tolist())
        )
        item = {
            "episode_index": index,
            "outcome": "success" if bool(record.get("success")) else "failure",
            "num_frames": count,
            "wide_video": str(rollout / "videos" / f"episode_{index:06d}_wide.mp4"),
            "wrist_video": str(rollout / "videos" / f"episode_{index:06d}_wrist.mp4"),
            "timeline": [
                {
                    "frame_index": frame,
                    "state": states[frame].tolist(),
                    "action": actions[frame].tolist(),
                }
                for frame in selected
            ],
        }
        (successes if bool(record.get("success")) else failures).append(item)
    evidence = {
        "schema_version": 2,
        "observation_only": True,
        **diagnosis_evidence_metadata(summary),
        "successful_episode_count": len(successes),
        "failure_episode_count": len(failures),
        "successes": successes,
        "failures": failures,
    }
    LiberoRuntime.write_json(output / "agent_input.json", evidence)
    assert settings is not None
    run_root = rollout.parent
    write_task_prompt(
        settings,
        output,
        run_root=run_root,
        settings_path=run_root / "experiment.resolved.yaml",
    )
    return evidence


def validate_diagnosis(
    rollout_dir: str | Path, diagnosis_path: str | Path
) -> dict[str, Any]:
    rollout = Path(rollout_dir).resolve()
    summary = _read_json(rollout / "summary.json")
    if "trajectory_protocol" in summary:
        validate_rollout_contract(summary)
    expected = {
        int(item["episode_index"]): int(item["num_steps"])
        for item in summary.get("episodes", [])
        if not bool(item.get("success"))
    }
    diagnosis = _read_json(Path(diagnosis_path).resolve())
    if int(diagnosis.get("schema_version", 0)) != 1 or not isinstance(
        diagnosis.get("episodes"), list
    ):
        raise ValueError("diagnosis must use schema_version=1 and contain episodes[]")
    actual = [int(item.get("episode_index", -1)) for item in diagnosis["episodes"]]
    if len(actual) != len(set(actual)) or set(actual) != set(expected):
        raise ValueError(
            f"diagnosis episode coverage mismatch: expected={sorted(expected)}, actual={sorted(actual)}"
        )
    successful_indices = {
        int(item["episode_index"])
        for item in summary.get("episodes", [])
        if bool(item.get("success"))
    }
    references = diagnosis.get("successful_reference_episodes")
    if expected and (not isinstance(references, list) or not references):
        raise ValueError("diagnosis must cite successful_reference_episodes[]")
    for reference in references or []:
        reference_index = int(reference.get("episode_index", -1))
        if reference_index not in successful_indices:
            raise ValueError(
                f"successful reference is not a successful episode: {reference_index}"
            )
        if not str(reference.get("reason", "")).strip():
            raise ValueError(
                f"successful reference {reference_index} must contain a reason"
            )
    mode_records = diagnosis.get("failure_modes")
    if not isinstance(mode_records, list) or (expected and not mode_records):
        raise ValueError("diagnosis must contain failure_modes[]")
    mode_ids = {str(mode.get("failure_mode_id", "")) for mode in mode_records}
    if "" in mode_ids:
        raise ValueError("failure_modes[] contains an empty failure_mode_id")
    declared_mode_episodes: dict[str, set[int]] = {}
    for mode in mode_records:
        mode_id = str(mode.get("failure_mode_id", ""))
        if (
            not str(mode.get("label", "")).strip()
            or not str(mode.get("category", "")).strip()
        ):
            raise ValueError(f"failure mode {mode_id} must have label and category")
        declared = mode.get("episode_indices")
        if not isinstance(declared, list):
            raise ValueError(f"failure mode {mode_id} must contain episode_indices[]")
        declared_mode_episodes[mode_id] = {int(index) for index in declared}
        if not declared_mode_episodes[mode_id].issubset(expected):
            raise ValueError(f"failure mode {mode_id} references an unknown episode")
    actual_mode_episodes: dict[str, set[int]] = {}
    for item in diagnosis["episodes"]:
        index = int(item["episode_index"])
        causal = int(item["first_causal_frame_index"])
        start = int(item["recoverable_window_start_frame_index"])
        stop = int(item["recoverable_window_stop_frame_index"])
        if not str(item.get("failure_phase", "")).strip():
            raise ValueError(f"episode {index} has empty failure_phase")
        mode_id = str(item.get("failure_mode_id", ""))
        if mode_id not in mode_ids:
            raise ValueError(
                f"episode {index} references unknown failure_mode_id: {mode_id!r}"
            )
        if not str(item.get("failure_category", "")).strip():
            raise ValueError(f"episode {index} has empty failure_category")
        if not str(item.get("failure_mode", "")).strip():
            raise ValueError(f"episode {index} has empty failure_mode")
        actual_mode_episodes.setdefault(mode_id, set()).add(index)
        if not (0 <= start < stop == causal < expected[index]):
            raise ValueError(f"episode {index} has invalid causal/window indices")
        confidence = float(item.get("confidence", -1.0))
        if not 0.0 <= confidence <= 1.0:
            raise ValueError(f"episode {index} confidence must be in [0, 1]")
        if not isinstance(item.get("evidence"), list) or not item["evidence"]:
            raise ValueError(f"episode {index} must cite public evidence")
    if declared_mode_episodes != actual_mode_episodes:
        raise ValueError(
            f"failure mode episode membership mismatch: declared={declared_mode_episodes}, "
            f"actual={actual_mode_episodes}"
        )
    return diagnosis


def select_reset_candidates(
    settings: ExperimentSettings, diagnosis: dict[str, Any]
) -> dict[str, Any]:
    """Select reset points under the configured intervention contract."""

    selection = settings.reset.candidate_selection
    if selection not in {
        "per_episode_stage_entry_only",
        "failed_stage_entry_only",
        "pre_causal_only",
        "window_start_only",
        "window_endpoints",
        "pre_window_and_endpoints",
    }:
        raise ValueError(f"unsupported reset candidate selection: {selection!r}")

    selected: list[dict[str, Any]] = []
    declared_reference_indices = {
        int(reference["episode_index"])
        for reference in diagnosis.get("successful_reference_episodes", [])
        if isinstance(reference, dict) and "episode_index" in reference
    }
    if selection == "per_episode_stage_entry_only":
        stage_graph = diagnosis.get("observable_stage_graph")
        if not isinstance(stage_graph, list) or not stage_graph:
            raise ValueError(
                "per-episode stage selection requires observable_stage_graph[]"
            )
        required_stage_fields = {
            "stage_name",
            "observable_entry_condition",
            "observable_exit_condition",
            "required_prerequisites",
            "relevant_entities",
            "supporting_successful_episode_indices",
        }
        for position, stage in enumerate(stage_graph):
            if not isinstance(stage, dict) or not required_stage_fields.issubset(stage):
                raise ValueError(
                    f"observable_stage_graph[{position}] lacks required stage fields"
                )
    for item in sorted(
        diagnosis["episodes"], key=lambda value: int(value["episode_index"])
    ):
        start = int(item["recoverable_window_start_frame_index"])
        stop = int(item["recoverable_window_stop_frame_index"])
        reference_values: list[int] = []
        reference_comparison = ""
        intervention_stage = ""
        stage_entry = -1
        stage_evidence: dict[str, Any] = {}
        if selection == "per_episode_stage_entry_only":
            intervention_stage = str(item.get("intervention_stage", "")).strip()
            if not intervention_stage:
                raise ValueError(
                    f"episode {item['episode_index']} must declare intervention_stage"
                )
            if "intervention_stage_start_frame_index" not in item:
                raise ValueError(
                    f"episode {item['episode_index']} must declare "
                    "intervention_stage_start_frame_index"
                )
            stage_entry = int(item["intervention_stage_start_frame_index"])
            causal = int(item["first_causal_frame_index"])
            if not 0 < stage_entry < causal:
                raise ValueError(
                    f"episode {item['episode_index']} has invalid per-episode stage "
                    f"entry {stage_entry} for causal frame {causal}"
                )
            raw_stage_evidence = item.get("stage_entry_evidence")
            if not isinstance(raw_stage_evidence, dict):
                raise ValueError(
                    f"episode {item['episode_index']} must record stage_entry_evidence"
                )
            inspected = raw_stage_evidence.get("inspected_frame_indices")
            if not isinstance(inspected, list) or not inspected:
                raise ValueError(
                    f"episode {item['episode_index']} must list inspected frame indices"
                )
            inspected_values = {int(value) for value in inspected}
            if stage_entry not in inspected_values or stage_entry - 1 not in inspected_values:
                raise ValueError(
                    f"episode {item['episode_index']} stage evidence must include "
                    "the boundary and preceding frame"
                )
            required_evidence_strings = (
                "preceding_stage",
                "observable_transition",
                "state_action_transition",
                "persistence_evidence",
                "why_previous_frame_is_too_early",
                "why_later_frame_is_too_late",
            )
            for evidence_key in required_evidence_strings:
                if not str(raw_stage_evidence.get(evidence_key, "")).strip():
                    raise ValueError(
                        f"episode {item['episode_index']} stage evidence must record "
                        f"{evidence_key}"
                    )
            camera_evidence = raw_stage_evidence.get("camera_evidence")
            if not isinstance(camera_evidence, dict) or not camera_evidence:
                raise ValueError(
                    f"episode {item['episode_index']} must cite per-camera stage evidence"
                )
            if any(not str(path).strip() for path in camera_evidence.values()):
                raise ValueError(
                    f"episode {item['episode_index']} has an empty camera evidence path"
                )
            stage_references = raw_stage_evidence.get(
                "successful_reference_episode_indices"
            )
            if not isinstance(stage_references, list) or not stage_references:
                raise ValueError(
                    f"episode {item['episode_index']} must cite stage-entry successful references"
                )
            if not {int(value) for value in stage_references}.issubset(
                declared_reference_indices
            ):
                raise ValueError(
                    f"episode {item['episode_index']} cites an undeclared stage-entry reference"
                )
            stage_evidence = dict(raw_stage_evidence)
            points = (("stage_entry", stage_entry),)
        elif selection == "failed_stage_entry_only":
            intervention_stage = str(item.get("intervention_stage", "")).strip()
            if not intervention_stage:
                raise ValueError(
                    f"episode {item['episode_index']} must declare intervention_stage"
                )
            if "intervention_stage_start_frame_index" not in item:
                raise ValueError(
                    f"episode {item['episode_index']} must declare "
                    "intervention_stage_start_frame_index"
                )
            stage_entry = int(item["intervention_stage_start_frame_index"])
            causal = int(item["first_causal_frame_index"])
            if not 0 <= stage_entry < causal:
                raise ValueError(
                    f"episode {item['episode_index']} has invalid stage entry "
                    f"{stage_entry} for causal frame {causal}"
                )
            points = (("stage_entry", stage_entry),)
        elif selection == "pre_causal_only":
            causal = int(item["first_causal_frame_index"])
            raw_references = item.get("successful_reference_episode_indices")
            if not isinstance(raw_references, list) or not raw_references:
                raise ValueError(
                    f"episode {item['episode_index']} must cite "
                    "successful_reference_episode_indices for pre-causal selection"
                )
            reference_values = [int(value) for value in raw_references]
            if not set(reference_values).issubset(declared_reference_indices):
                raise ValueError(
                    f"episode {item['episode_index']} cites an undeclared successful reference"
                )
            reference_comparison = str(
                item.get("successful_reference_comparison", "")
            ).strip()
            if not reference_comparison:
                raise ValueError(
                    f"episode {item['episode_index']} must record a concrete "
                    "successful_reference_comparison for pre-causal selection"
                )
            if causal != stop:
                raise ValueError(
                    f"episode {item['episode_index']} causal/window-stop mismatch"
                )
            pre_causal = causal - 1
            if pre_causal < 0:
                raise ValueError(
                    f"episode {item['episode_index']} has no frame before causal frame {causal}"
                )
            points = (("pre_causal", pre_causal),)
        elif selection == "pre_window_and_endpoints":
            prevention = start - settings.reset.prevention_steps
            if prevention < 0:
                raise ValueError(
                    f"episode {item['episode_index']} window start {start} cannot provide "
                    f"an exact {settings.reset.prevention_steps}-step pre-window intervention"
                )
            points = (
                ("pre_window", prevention),
                ("window_start", start),
                ("window_stop", stop),
            )
        elif selection == "window_start_only":
            points = (("window_start", start),)
        else:
            points = (("window_start", start), ("window_stop", stop))
        for rank, (role, frame) in enumerate(points):
            selected.append(
                {
                    "episode_index": int(item["episode_index"]),
                    "candidate_rank": rank,
                    "intervention_point": role,
                    "requested_frame_index": frame,
                    "failure_phase": item["failure_phase"],
                    "failure_mode_id": item.get("failure_mode_id", ""),
                    "failure_category": item.get(
                        "failure_category", item.get("failure_phase", "")
                    ),
                    "failure_mode": item.get(
                        "failure_mode", item.get("failure_phase", "")
                    ),
                    "window_start": start,
                    "window_stop": stop,
                    **(
                        {
                            "intervention_stage": intervention_stage,
                            "stage_entry_frame_index": stage_entry,
                            "first_causal_frame_index": int(
                                item["first_causal_frame_index"]
                            ),
                            "stage_entry_evidence": stage_evidence,
                        }
                        if selection == "per_episode_stage_entry_only"
                        else {}
                    ),
                    **(
                        {
                            "intervention_stage": intervention_stage,
                            "stage_entry_frame_index": stage_entry,
                            "first_causal_frame_index": int(
                                item["first_causal_frame_index"]
                            ),
                        }
                        if selection == "failed_stage_entry_only"
                        else {}
                    ),
                    **(
                        {
                            "first_causal_frame_index": stop,
                            "successful_reference_episode_indices": reference_values,
                            "successful_reference_comparison": reference_comparison,
                        }
                        if selection == "pre_causal_only"
                        else {}
                    ),
                }
            )
    if selection == "per_episode_stage_entry_only":
        return {
            "schema_version": 1,
            "selection": "per-episode observed behavior stage entry only",
            "intervention_points": ["stage_entry"],
            "frames_per_failure": 1,
            "candidates": selected,
        }
    if selection == "failed_stage_entry_only":
        return {
            "schema_version": 1,
            "selection": "failed behavior stage entry only",
            "intervention_points": ["stage_entry"],
            "frames_per_failure": 1,
            "candidates": selected,
        }
    if selection == "pre_causal_only":
        return {
            "schema_version": 1,
            "selection": "successful-reference-aligned pre-causal state only",
            "intervention_points": ["pre_causal"],
            "frames_per_failure": 1,
            "candidates": selected,
        }
    if selection == "pre_window_and_endpoints":
        return {
            "schema_version": 1,
            "selection": "pre-window prevention plus failure window endpoints",
            "intervention_points": ["pre_window", "window_start", "window_stop"],
            "prevention_steps": settings.reset.prevention_steps,
            "frames_per_failure": 3,
            "candidates": selected,
        }
    if selection == "window_start_only":
        return {
            "schema_version": 1,
            "selection": "failure window start only",
            "intervention_points": ["window_start"],
            "frames_per_failure": 1,
            "candidates": selected,
        }
    return {
        "schema_version": 1,
        "selection": "failure window endpoints",
        "intervention_points": ["window_start", "window_stop"],
        "frames_per_failure": 2,
        "candidates": selected,
    }


def _restore_gripper(env: Any, state: np.ndarray | None) -> None:
    if state is None:
        return
    gripper = env.env.robots[0].gripper
    desired = np.asarray(state, dtype=np.float64)
    gripper.current_action = desired.copy()


def _replay_one(
    settings: ExperimentSettings,
    rollout: Path,
    initial_states: np.ndarray,
    candidate: dict[str, Any],
    private_path: Path,
    agent_view_path: Path,
) -> dict[str, Any]:
    index = int(candidate["episode_index"])
    requested = int(candidate["requested_frame_index"])
    episode = _read_json(_episode_path(rollout, index))
    actions = np.asarray(episode["actions"], dtype=np.float32)
    recorded = np.asarray(episode["states"], dtype=np.float64)
    source = LiberoRuntime(
        settings.task.suite,
        settings.task.task_id,
        settings.controller.source_control_space,
        settings.rollout.control_frequency_hz,
        libero_root=settings.backend.libero_root,
    )
    scene_seed = int(episode["scene_model_seed"])
    env = source.new_env(scene_seed)
    try:
        obs = env.set_init_state(initial_states[int(episode["initial_state_index"])])
        for _ in range(10):
            obs, _, done, _ = env.step(source.neutral_action(env).tolist())
            if done:
                raise RuntimeError(
                    f"episode {index} terminated during replay stabilization"
                )
        initial_error = float(np.max(np.abs(source.public_state(obs) - recorded[0])))
        if initial_error > PUBLIC_REPLAY_TOLERANCE:
            raise RuntimeError(
                f"episode {index} initial replay error {initial_error} exceeds tolerance"
            )
        max_error = initial_error
        for action_index in range(requested):
            obs, _, done, _ = env.step(actions[action_index].tolist())
            if done:
                raise RuntimeError(
                    f"episode {index} ended before requested frame {requested}"
                )
            error = float(
                np.max(np.abs(source.public_state(obs) - recorded[action_index + 1]))
            )
            max_error = max(max_error, error)
            if error > PUBLIC_REPLAY_TOLERANCE:
                raise RuntimeError(
                    f"episode {index} diverged at frame {action_index + 1}: {error}"
                )
        import imageio.v2 as imageio

        agent_view, _ = source.observation_images(obs)
        agent_view_path.parent.mkdir(parents=True, exist_ok=True)
        imageio.imwrite(agent_view_path, agent_view)
        agent_view_sha256 = hashlib.sha256(
            np.ascontiguousarray(agent_view).view(np.uint8)
        ).hexdigest()
        sim_state = np.asarray(env.get_sim_state(), dtype=np.float64).copy()
        gripper_state = source.capture_gripper_controller(env)
    finally:
        env.close()

    target = LiberoRuntime(
        settings.task.suite,
        settings.task.task_id,
        settings.controller.target_control_space,
        settings.rollout.control_frequency_hz,
        libero_root=settings.backend.libero_root,
    )
    target_env = target.new_env(scene_seed)
    try:
        target_env.set_init_state(initial_states[int(episode["initial_state_index"])])
        target.set_sim_state(target_env, sim_state)
        _restore_gripper(target_env, gripper_state)
        dynamics = target.apply_reset_dynamics(target_env, settings.reset.dynamics)
        final_state = np.asarray(target_env.get_sim_state(), dtype=np.float64).copy()
        final_gripper = target.capture_gripper_controller(target_env)
        target.write_private_state(
            private_path, sim_state=final_state, gripper_state=final_gripper
        )
    finally:
        target_env.close()
    return {
        **candidate,
        "verified": True,
        "replayed_action_count": requested,
        "max_public_state_error": max_error,
        "public_tolerance": PUBLIC_REPLAY_TOLERANCE,
        "source_control_space": settings.controller.source_control_space,
        "target_control_space": settings.controller.target_control_space,
        "reset_dynamics": settings.reset.dynamics,
        "dynamics_audit": dynamics,
        "private_state": str(private_path.name),
        "private_state_sha256": target.state_hash(final_state),
        "agent_view": str(agent_view_path.name),
        "agent_view_sha256": agent_view_sha256,
    }


def materialize_reset_bank(
    settings: ExperimentSettings,
    rollout_dir: str | Path,
    diagnosis_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Validate all failures and atomically publish one repair handoff bundle."""

    rollout = Path(rollout_dir).resolve()
    output = Path(output_dir).resolve()
    summary = _read_json(rollout / "summary.json")
    source_rollout_fingerprint, reused_rollout = (
        _validate_materialization_rollout_contract(settings, rollout, summary)
    )
    diagnosis = validate_diagnosis(rollout, diagnosis_path)
    candidates = select_reset_candidates(settings, diagnosis)
    initial_states = np.load(rollout / "initial_states.npy", allow_pickle=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        (temporary / "private_reset_states").mkdir(parents=True)
        (temporary / "agent_views").mkdir(parents=True)
        reports: list[dict[str, Any]] = []
        for candidate in candidates["candidates"]:
            state_name = (
                f"episode_{candidate['episode_index']:06d}_"
                f"frame_{candidate['requested_frame_index']:06d}.npz"
            )
            reports.append(
                _replay_one(
                    settings,
                    rollout,
                    initial_states,
                    candidate,
                    temporary / "private_reset_states" / state_name,
                    temporary / "agent_views" / state_name.replace(".npz", ".png"),
                )
            )
        resets: list[dict[str, Any]] = []
        for report in reports:
            private_name = str(report.pop("private_state"))
            view_name = str(report.pop("agent_view"))
            private_relative = Path("private_reset_states") / private_name
            view_relative = Path("agent_views") / view_name
            reset = {
                **report,
                "job_id": (
                    f"e{report['episode_index']:06d}-"
                    f"f{report['requested_frame_index']:06d}"
                ),
                "reset_frame_index": report["requested_frame_index"],
                "reset_state": str(private_relative),
                "reset_state_file_sha256": hashlib.sha256(
                    (temporary / private_relative).read_bytes()
                ).hexdigest(),
                "agent_view": str(view_relative),
                "agent_view_file_sha256": hashlib.sha256(
                    (temporary / view_relative).read_bytes()
                ).hexdigest(),
            }
            resets.append(reset)
        manifest = {
            "schema_version": 1,
            "artifact_type": "vla_mender.repair_handoff",
            "complete": True,
            "settings_fingerprint": settings.fingerprint(),
            "source": {
                "run_root": str(rollout.parent),
                "rollout_dir": str(rollout),
                "diagnosis_working_file": str(Path(diagnosis_path).resolve()),
                "rollout_settings_fingerprint": source_rollout_fingerprint,
                "reused_frozen_rollout": reused_rollout,
            },
            "diagnosis": diagnosis,
            "selection": {
                "method": candidates["selection"],
                "intervention_points": candidates["intervention_points"],
                "frames_per_failure": candidates["frames_per_failure"],
                **(
                    {"prevention_steps": candidates["prevention_steps"]}
                    if "prevention_steps" in candidates
                    else {}
                ),
            },
            "summary": {
                "failure_episode_count": len(diagnosis["episodes"]),
                "failure_mode_count": len(diagnosis["failure_modes"]),
                "reset_count": len(resets),
                "replay_verified_count": sum(
                    1 for item in resets if bool(item.get("verified"))
                ),
                "all_replays_verified": all(
                    bool(item.get("verified")) for item in resets
                ),
            },
            "resets": resets,
        }
        LiberoRuntime.write_json(temporary / "manifest.json", manifest)
        if output.exists():
            raise FileExistsError(
                f"refusing to replace existing repair handoff: {output}; "
                "use a fresh run root or explicitly archive the old handoff"
            )
        temporary.replace(output)
        return manifest
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
