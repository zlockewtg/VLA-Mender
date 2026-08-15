"""Simulator and visual continuity checks used at repair admission time."""

from __future__ import annotations

import hashlib
import importlib
import json
import platform
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

import cv2
import numpy as np


MODEL_NUMERIC_ARRAY_FIELDS = (
    "body_parentid", "body_pos", "body_quat", "body_ipos", "body_iquat",
    "body_mass", "body_inertia", "jnt_type", "jnt_bodyid", "jnt_qposadr",
    "jnt_dofadr", "jnt_pos", "jnt_axis", "jnt_range", "jnt_stiffness",
    "dof_damping", "dof_frictionloss", "dof_armature", "geom_type",
    "geom_bodyid", "geom_dataid", "geom_pos", "geom_quat", "geom_size",
    "geom_rgba", "geom_group", "geom_friction", "geom_solref", "geom_solimp",
    "geom_contype", "geom_conaffinity", "cam_bodyid", "cam_pos", "cam_quat",
    "cam_fovy", "light_bodyid", "light_pos", "light_dir", "light_diffuse",
    "light_ambient", "light_specular", "mesh_vertadr", "mesh_vertnum",
    "mesh_faceadr", "mesh_facenum", "mesh_vert", "mesh_face", "mat_texid",
    "mat_rgba", "mat_texrepeat", "mat_specular", "mat_shininess",
    "mat_reflectance", "tex_type", "tex_height", "tex_width", "tex_adr",
    "tex_rgb",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def canonical_array_sha256(value: np.ndarray, *, dtype: str | None = None) -> str:
    array = np.asarray(value, dtype=dtype)
    canonical = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(canonical.dtype).encode())
    digest.update(np.asarray(canonical.shape, dtype="<i8").tobytes())
    digest.update(canonical.tobytes())
    return digest.hexdigest()


def simulator_state_sha256(state: np.ndarray) -> str:
    canonical = np.ascontiguousarray(np.asarray(state, dtype="<f8").reshape(-1))
    return hashlib.sha256(canonical.tobytes()).hexdigest()


def model_numeric_sha256(
    model: Any, fields: Iterable[str] = MODEL_NUMERIC_ARRAY_FIELDS
) -> str:
    """Hash compiled model values, including randomized static scene geometry."""

    digest = hashlib.sha256()
    found = 0
    for name in fields:
        value = getattr(model, name, None)
        if value is None:
            continue
        array = np.asarray(value)
        if array.dtype.kind in "iu":
            canonical = np.ascontiguousarray(array, dtype="<i8")
        elif array.dtype.kind == "b":
            canonical = np.ascontiguousarray(array, dtype=np.uint8)
        elif array.dtype.kind in "fc":
            canonical = np.ascontiguousarray(array, dtype="<f8")
        else:
            continue
        digest.update(name.encode())
        digest.update(b"\0")
        digest.update(json.dumps(list(canonical.shape)).encode())
        digest.update(b"\0")
        digest.update(canonical.tobytes())
        found += 1
    require(found > 0, "simulator model exposes no configured numeric arrays")
    return digest.hexdigest()


def simulator_signature(env_or_sim: Any) -> dict[str, Any]:
    """Describe state layout and compiled model without depending on LIBERO imports."""

    sim = getattr(env_or_sim, "sim", env_or_sim)
    model = sim.model
    names: list[str] = []
    names_by_kind: dict[str, list[str]] = {}
    for count_name, lookup_name in (
        ("nbody", "body_id2name"),
        ("njnt", "joint_id2name"),
        ("ngeom", "geom_id2name"),
    ):
        lookup = getattr(model, lookup_name, None)
        group = (
            [f"{count_name}:{lookup(index)}" for index in range(int(getattr(model, count_name, 0)))]
            if lookup is not None
            else []
        )
        names.extend(group)
        names_by_kind[count_name] = group
    body_joint_names = names_by_kind["nbody"] + names_by_kind["njnt"]
    geom_names = names_by_kind["ngeom"]
    state_width = int(np.asarray(sim.get_state().flatten()).size) if hasattr(sim, "get_state") else (
        int(np.asarray(env_or_sim.get_sim_state()).size)
    )
    return {
        "nq": int(model.nq),
        "nv": int(model.nv),
        "na": int(getattr(model, "na", 0)),
        "state_width": state_width,
        "model_names_sha256": hashlib.sha256("\n".join(names).encode()).hexdigest(),
        "body_joint_names_sha256": hashlib.sha256(
            "\n".join(body_joint_names).encode()
        ).hexdigest(),
        "geom_names_sha256": hashlib.sha256("\n".join(geom_names).encode()).hexdigest(),
        "model_numeric_sha256": model_numeric_sha256(model),
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "model_names": names,
        "model_names_by_kind": names_by_kind,
        **_runtime_module_paths(),
    }


def _runtime_module_paths() -> dict[str, str]:
    result: dict[str, str] = {}
    for name in ("libero", "robosuite"):
        try:
            module = importlib.import_module(name)
        except ImportError:
            continue
        origin = getattr(module, "__file__", None)
        if origin:
            result[f"{name}_module_path"] = str(Path(origin).resolve())
        version = getattr(module, "__version__", None)
        if version is not None:
            result[f"{name}_version"] = str(version)
    return result


def signature_mismatches(
    source: Mapping[str, Any], target: Mapping[str, Any], fields: Iterable[str]
) -> dict[str, tuple[Any, Any]]:
    missing = "<missing>"
    return {
        field: (source.get(field, missing), target.get(field, missing))
        for field in fields
        if field not in source or field not in target or source[field] != target[field]
    }


def flow_metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    require(reference.shape == candidate.shape, "continuity images have different shapes")
    gray0 = cv2.cvtColor(reference, cv2.COLOR_RGB2GRAY)
    gray1 = cv2.cvtColor(candidate, cv2.COLOR_RGB2GRAY)
    flow = cv2.calcOpticalFlowFarneback(
        gray0, gray1, None, 0.5, 3, 25, 5, 7, 1.5, 0
    )
    magnitude = np.linalg.norm(flow, axis=2)
    return {
        "mae": float(np.mean(np.abs(reference.astype(np.float32) - candidate.astype(np.float32)))),
        "flow_median_px": float(np.median(magnitude)),
        "flow_p90_px": float(np.quantile(magnitude, 0.9)),
        "flow_p99_px": float(np.quantile(magnitude, 0.99)),
        "flow_max_px": float(np.max(magnitude)),
    }


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"expected JSON object: {path}")
    return value


def validate_simulator_evidence(
    continuity: Mapping[str, Any], *, fields: tuple[str, ...], tolerance: float
) -> dict[str, Any]:
    """Verify reset descriptor, repair runtime signature, and exact repair row 0."""

    descriptor_path = Path(str(continuity["reset_descriptor"])).resolve()
    attempt_path = Path(str(continuity["attempt_manifest"])).resolve()
    result_path = Path(str(continuity["result"])).resolve()
    descriptor = read_json(descriptor_path)
    attempt = read_json(attempt_path)
    result = read_json(result_path)
    source_signature = descriptor.get("sim_signature") or {}
    # Older repair artifacts retained the complete target-side signature under
    # ``transfer``.  Observation-only repair artifacts intentionally keep that
    # private and publish a coordinator-generated strict verification audit.
    # Both formats are accepted, but the attested form is tied back to the
    # descriptor by scene identity and exact simulator-state hash below.
    transfer = attempt.get("transfer") or attempt.get("public_restore_audit") or {}
    target_signature = transfer.get("sim_signature") or {}
    verification_mode = "target_signature_comparison"
    if target_signature:
        mismatches = signature_mismatches(source_signature, target_signature, fields)
        require(not mismatches, f"reset/repair simulator signature mismatch: {mismatches}")
    else:
        missing_source = [field for field in fields if field not in source_signature]
        require(not missing_source, f"reset signature is missing fields: {missing_source}")
        require(
            "public_restore_audit" in attempt,
            f"repair artifact has neither target signature nor public restore audit: {attempt_path}",
        )
        identity_pairs = (
            ("suite", "suite"),
            ("task_id", "task_id"),
            ("suite_episode_index", "suite_episode_index"),
            ("scene_model_seed", "scene_model_seed"),
            ("restored_frame_index", "frame_index"),
        )
        identity_mismatches = {
            source_key: (descriptor.get(source_key), transfer.get(target_key))
            for source_key, target_key in identity_pairs
            if source_key not in descriptor
            or target_key not in transfer
            or descriptor[source_key] != transfer[target_key]
        }
        require(
            not identity_mismatches,
            f"reset/repair public audit identity mismatch: {identity_mismatches}",
        )
        verification_mode = "trusted_runtime_attestation"
    require(
        transfer.get("strict_sim_signature_verified") is True,
        f"repair did not verify strict simulator signature: {attempt_path}",
    )
    recorded_fields = tuple(transfer.get("strict_sim_signature_fields", ()))
    require(recorded_fields == fields, f"unexpected signature policy: {recorded_fields}")
    expected_source_python = continuity.get("expected_source_python")
    if expected_source_python is not None:
        expected_source_python = str(
            Path(str(expected_source_python)).expanduser().absolute()
        )
        actual_source_python = str(
            Path(str(source_signature.get("python_executable", "")))
            .expanduser()
            .absolute()
        )
        require(
            actual_source_python == expected_source_python,
            f"reset replay Python differs from source rollout: {actual_source_python} != {expected_source_python}",
        )
        robosuite_path = Path(str(source_signature.get("robosuite_module_path", ""))).resolve()
        runtime_root = Path(expected_source_python).parent.parent.resolve()
        try:
            robosuite_path.relative_to(runtime_root)
        except ValueError as exc:
            raise RuntimeError(
                f"reset replay robosuite is outside source runtime: {robosuite_path}"
            ) from exc

    evidence = result.get("repair_row0_simulator_state") or {}
    require(evidence.get("verified") is True, f"missing verified repair row-0 evidence: {result_path}")
    reset_path = Path(str(descriptor["private_simulator_state_path"])).resolve()
    repair_path = Path(str(evidence["path"])).resolve()
    reset_key = str(continuity.get("reset_state_key", "simulator_state"))
    repair_key = str(evidence.get("key", continuity.get("repair_state_key", "simulator_state")))
    with np.load(reset_path, allow_pickle=False) as payload:
        require(reset_key in payload.files, f"missing {reset_key}: {reset_path}")
        reset_state = np.asarray(payload[reset_key], dtype=np.float64).reshape(-1)
    with np.load(repair_path, allow_pickle=False) as payload:
        require(repair_key in payload.files, f"missing {repair_key}: {repair_path}")
        repair_state = np.asarray(payload[repair_key], dtype=np.float64).reshape(-1)
    require(reset_state.shape == repair_state.shape, "reset and repair row-0 state shapes differ")
    error = float(np.max(np.abs(reset_state - repair_state))) if reset_state.size else 0.0
    require(error <= tolerance, f"reset/repair row-0 state error {error} exceeds {tolerance}")
    reset_hash = simulator_state_sha256(reset_state)
    repair_hash = simulator_state_sha256(repair_state)
    require(reset_hash == repair_hash, "reset/repair row-0 simulator-state hash mismatch")
    require(evidence.get("reset_sha256") == reset_hash, "recorded reset state hash mismatch")
    require(evidence.get("sha256") == repair_hash, "recorded repair state hash mismatch")
    if verification_mode == "trusted_runtime_attestation":
        require(
            transfer.get("simulator_state_width") == int(reset_state.size),
            "public restore audit state width mismatch",
        )
        require(
            transfer.get("simulator_state_sha256") == reset_hash,
            "public restore audit state hash mismatch",
        )
        target_signature = source_signature
    return {
        "verified": True,
        "verification_mode": verification_mode,
        "signature_fields": list(fields),
        "source_signature": {field: source_signature[field] for field in fields},
        "repair_signature": {field: target_signature[field] for field in fields},
        "state_width": int(reset_state.size),
        "state_sha256": reset_hash,
        "max_abs_error": error,
        "tolerance": tolerance,
        "reset_descriptor": str(descriptor_path),
        "attempt_manifest": str(attempt_path),
        "result": str(result_path),
        "expected_source_python": expected_source_python,
    }
