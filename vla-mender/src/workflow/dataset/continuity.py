"""Simulator and visual continuity checks used at repair admission time."""

from __future__ import annotations

import hashlib
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
        if not hasattr(model, name):
            continue
        array = np.ascontiguousarray(np.asarray(getattr(model, name)))
        digest.update(name.encode())
        digest.update(str(array.dtype).encode())
        digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
        digest.update(array.tobytes())
        found += 1
    require(found > 0, "simulator model exposes no configured numeric arrays")
    return digest.hexdigest()


def _names(model: Any, attribute: str) -> list[str]:
    values = getattr(model, attribute, ())
    return [str(value) for value in values]


def _string_list_sha256(values: Iterable[str]) -> str:
    return hashlib.sha256("\0".join(values).encode()).hexdigest()


def simulator_signature(env_or_sim: Any) -> dict[str, Any]:
    """Describe state layout and compiled model without depending on LIBERO imports."""

    sim = getattr(env_or_sim, "sim", env_or_sim)
    model = sim.model
    names = _names(model, "names") or (
        [f"body:{item}" for item in _names(model, "body_names")]
        + [f"joint:{item}" for item in _names(model, "joint_names")]
        + [f"geom:{item}" for item in _names(model, "geom_names")]
    )
    body_joint_names = _names(model, "body_names") + _names(model, "joint_names")
    geom_names = _names(model, "geom_names")
    state_width = int(np.asarray(sim.get_state().flatten()).size) if hasattr(sim, "get_state") else (
        int(np.asarray(env_or_sim.get_sim_state()).size)
    )
    return {
        "nq": int(model.nq),
        "nv": int(model.nv),
        "na": int(getattr(model, "na", 0)),
        "state_width": state_width,
        "model_names_sha256": _string_list_sha256(names),
        "body_joint_names_sha256": _string_list_sha256(body_joint_names),
        "geom_names_sha256": _string_list_sha256(geom_names),
        "model_numeric_sha256": model_numeric_sha256(model),
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
    }


def signature_mismatches(
    source: Mapping[str, Any], target: Mapping[str, Any], fields: Iterable[str]
) -> dict[str, tuple[Any, Any]]:
    return {
        field: (source.get(field), target.get(field))
        for field in fields
        if source.get(field) != target.get(field)
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
    transfer = attempt.get("transfer") or {}
    target_signature = transfer.get("sim_signature") or {}
    mismatches = signature_mismatches(source_signature, target_signature, fields)
    require(not mismatches, f"reset/repair simulator signature mismatch: {mismatches}")
    require(
        transfer.get("strict_sim_signature_verified") is True,
        f"repair did not verify strict simulator signature: {attempt_path}",
    )
    recorded_fields = tuple(transfer.get("strict_sim_signature_fields", ()))
    require(recorded_fields == fields, f"unexpected signature policy: {recorded_fields}")

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
    return {
        "verified": True,
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
    }
