import math
import pathlib
import time
from typing import Any

import numpy as np
import open3d as o3d
import viser.transforms as vtf
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation as SciRotation

from knowledge.api.env_protocol import (
    BaseEnv,
)
from knowledge.api.base_api import ApiBase
from knowledge.api.franka.common import (
    apply_tcp_offset,
)
from knowledge.api.franka.libero_reduced import FrankaLiberoApiReduced
from knowledge.api.motion.pyroki import init_pyroki
from knowledge.api.motion.pyroki_context import get_pyroki_context  # type: ignore
from knowledge.api.vision.graspnet import init_contact_graspnet
from knowledge.api.vision.molmo import init_molmo

# from knowledge.api.vision.owlvit import init_owlvit
# from knowledge.api.vision.sam2 import init_sam2
from knowledge.api.vision.sam3 import init_sam3, init_sam3_point_prompt
from knowledge.api.utils.camera_utils import obs_get_rgb
from knowledge.api.utils.depth_utils import depth_color_to_pointcloud, depth_to_pointcloud, depth_to_rgb


def _classify_grasp_resume_phase(
    current_position: np.ndarray,
    current_quaternion_wxyz: np.ndarray,
    pregrasp_position: np.ndarray,
    grasp_position: np.ndarray,
    grasp_quaternion_wxyz: np.ndarray,
    *,
    position_tolerance_m: float,
    orientation_tolerance_rad: float,
    contact_ready_tolerance_m: float | None = None,
) -> dict[str, Any]:
    """Classify an observed handoff pose without regressing grasp progress.

    A reset-suffix policy can take over after the VLA has already entered the
    pregrasp-to-contact segment.  Sending that pose back to pregrasp creates an
    artificial up-down loop.  This classifier uses only public Cartesian state
    and the observation-derived policy targets to distinguish contact-ready,
    descent-corridor, and genuinely unapproached states.
    """
    current = np.asarray(current_position, dtype=np.float64).reshape(3)
    pregrasp = np.asarray(pregrasp_position, dtype=np.float64).reshape(3)
    grasp = np.asarray(grasp_position, dtype=np.float64).reshape(3)
    current_quaternion = np.asarray(current_quaternion_wxyz, dtype=np.float64).reshape(4)
    grasp_quaternion = np.asarray(grasp_quaternion_wxyz, dtype=np.float64).reshape(4)
    if not all(
        np.all(np.isfinite(value))
        for value in (current, pregrasp, grasp, current_quaternion, grasp_quaternion)
    ):
        raise ValueError("grasp resume inputs must contain finite values")
    current_quaternion /= max(float(np.linalg.norm(current_quaternion)), 1.0e-12)
    grasp_quaternion /= max(float(np.linalg.norm(grasp_quaternion)), 1.0e-12)
    quaternion_dot = float(
        np.clip(abs(np.dot(current_quaternion, grasp_quaternion)), 0.0, 1.0)
    )
    orientation_error = float(2.0 * np.arccos(quaternion_dot))
    orientation_aligned = orientation_error <= orientation_tolerance_rad
    pregrasp_error = float(np.linalg.norm(current - pregrasp))
    grasp_error = float(np.linalg.norm(current - grasp))

    segment = grasp - pregrasp
    segment_length_squared = float(np.dot(segment, segment))
    if segment_length_squared <= 1.0e-12:
        segment_progress = 1.0
        corridor_error = grasp_error
    else:
        segment_progress = float(np.dot(current - pregrasp, segment) / segment_length_squared)
        projection = pregrasp + np.clip(segment_progress, 0.0, 1.0) * segment
        corridor_error = float(np.linalg.norm(current - projection))

    if contact_ready_tolerance_m is None:
        contact_ready_tolerance_m = position_tolerance_m
    contact_ready = bool(
        orientation_aligned and grasp_error <= contact_ready_tolerance_m
    )
    in_descent_corridor = bool(
        orientation_aligned
        and -0.10 <= segment_progress <= 1.15
        and corridor_error <= position_tolerance_m
    )
    resume_phase = (
        "contact_ready"
        if contact_ready
        else "descent_corridor"
        if in_descent_corridor
        else "pregrasp_required"
    )
    return {
        "resume_phase": resume_phase,
        "pregrasp_error_initial_m": pregrasp_error,
        "grasp_error_initial_m": grasp_error,
        "orientation_error_initial_rad": orientation_error,
        "descent_segment_progress": segment_progress,
        "descent_corridor_error_m": corridor_error,
        "phase_regression_avoided": resume_phase != "pregrasp_required",
    }


# ------------------------------- Control API ------------------------------
class FrankaLiberoApiReducedSkillLibrary(FrankaLiberoApiReduced):
    """
    Robot control helpers for Franka.
    """

    _VLAMENDER_XY_RELEASE_LIMIT_M = 0.030
    # Close settling is observation-bounded: stop after two consecutive
    # closed-aperture observations (and at least three control frames), while
    # retaining the audited 12-frame fallback for slow or unavailable state.
    _VLAMENDER_GRASP_CLOSE_MIN_STEPS = 3
    _VLAMENDER_GRASP_CLOSE_STABLE_STEPS = 2
    _VLAMENDER_GRASP_CLOSE_SETTLE_STEPS = 16
    # A grasped object normally prevents the fingers from reaching the
    # library's coarse "closed" aperture.  Detect that contact plateau from
    # public normalized width, after observing meaningful closure, rather than
    # appending the full fallback wait to every successful grasp.
    _VLAMENDER_GRASP_CONTACT_MIN_STEPS = 5
    _VLAMENDER_GRASP_CONTACT_STABLE_STEPS = 2
    _VLAMENDER_GRASP_CONTACT_WIDTH_DELTA = 1.5e-3
    _VLAMENDER_GRASP_CONTACT_MIN_CLOSURE = 0.05
    _VLAMENDER_GRASP_CONTACT_READY_TOLERANCE_M = 0.006
    _VLAMENDER_GRIPPER_OPEN_THRESHOLD = 0.85

    def __init__(
        self,
        env: BaseEnv,
        *,
        enable_ik: bool = True,
    ) -> None:
        super().__init__(env, enable_ik=enable_ik)
        self._vlamender_mask_commits: dict[str, dict[str, Any]] = {}
        self._vlamender_grounded_targets: dict[str, dict[str, Any]] = {}

    def functions(self) -> dict[str, Any]:
        fns = super().functions()
        fns["get_robot_state"] = self.get_robot_state
        fns["estimate_grasp_state"] = self.estimate_grasp_state
        fns["grasp_if_unheld"] = self.grasp_if_unheld
        fns["ground_placement_target"] = self.ground_placement_target
        fns["guarded_open_gripper"] = self.guarded_open_gripper
        fns["rotation_matrix_to_quaternion"] = self.rotation_matrix_to_quaternion
        fns["decompose_transform"] = self.decompose_transform
        fns["depth_to_point_cloud"] = self.depth_to_point_cloud
        fns["mask_to_world_points"] = self.mask_to_world_points
        fns["pixel_to_world_point"] = self.pixel_to_world_point
        fns["transform_points"] = self.transform_points
        fns["interpolate_segment"] = self.interpolate_segment
        fns["normalize_vector"] = self.normalize_vector
        fns["select_top_down_grasp"] = self.select_top_down_grasp

        return fns

    # VLA-Mender reset-suffix safety APIs
    # ====================================

    @staticmethod
    def _normalize_vlamender_prompts(object_prompts: Any) -> list[str]:
        values = [object_prompts] if isinstance(object_prompts, str) else list(object_prompts or [])
        prompts: list[str] = []
        for value in values:
            prompt = str(value).strip()
            if prompt and prompt not in prompts:
                prompts.append(prompt)
        if not prompts:
            raise ValueError("object_prompts must contain at least one non-empty semantic prompt")
        if len(prompts) > 8:
            raise ValueError("object_prompts is bounded to at most eight prompts")
        return prompts

    def get_robot_state(self, observation: dict[str, Any] | None = None) -> dict[str, Any]:
        """Return a validated, versioned robot-state view for reset-suffix policies.

        This is the only VLA-Mender API that should decode ``robot_cartesian_pos``.  The LIBERO
        layout is ``XYZ + quaternion_wxyz + gripper_width``.  In particular, index 6 is a
        quaternion component and must never be combined with the final gripper-width element.

        Args:
            observation: A value returned by ``get_observation``.  When omitted, a fresh
                observation is acquired.

        Returns:
            A plain dictionary containing the observed panda-hand EEF position, the Cartesian
            ``motion_target_position`` expressed in the same TCP frame accepted by ``goto_pose``,
            normalized quaternion, seven arm joints, normalized gripper width, a coarse aperture
            state, and a schema identifier.  Use ``motion_target_position`` when deriving a new
            ``goto_pose`` waypoint from the current robot pose; feeding ``eef_position`` back into
            ``goto_pose`` would apply the controller's TCP offset a second time.
        """
        obs = self.get_observation() if observation is None else observation
        if not isinstance(obs, dict):
            raise ValueError("observation must be a dictionary returned by get_observation")
        cart = np.asarray(obs.get("robot_cartesian_pos", []), dtype=np.float64).reshape(-1)
        if cart.size != 8 or not np.all(np.isfinite(cart)):
            raise ValueError(
                "robot_cartesian_pos must contain eight finite values: "
                "XYZ + quaternion_wxyz + gripper_width"
            )
        quaternion = cart[3:7].copy()
        quaternion_norm = float(np.linalg.norm(quaternion))
        if quaternion_norm < 1e-8:
            raise ValueError("robot_cartesian_pos contains a degenerate quaternion")
        quaternion /= quaternion_norm

        joints = np.asarray(obs.get("robot_joint_pos", []), dtype=np.float64).reshape(-1)
        if joints.size < 7 or not np.all(np.isfinite(joints[:7])):
            raise ValueError("robot_joint_pos must contain at least seven finite arm joints")

        raw_width = float(cart[-1])
        if not np.isfinite(raw_width) or raw_width < -0.05 or raw_width > 1.05:
            raise ValueError(
                f"gripper width {raw_width!r} is outside the documented normalized range"
            )
        width = float(np.clip(raw_width, 0.0, 1.0))
        if width <= 0.15:
            aperture = "closed"
        elif width >= self._VLAMENDER_GRIPPER_OPEN_THRESHOLD:
            aperture = "open"
        else:
            aperture = "intermediate"
        motion_target_position = apply_tcp_offset(
            cart[:3], quaternion, -np.asarray(self._TCP_OFFSET, dtype=np.float64)
        )
        return {
            "schema_version": "libero_xyz_quaternion_wxyz_gripper_v2",
            "eef_position": cart[:3].tolist(),
            "motion_target_position": motion_target_position.tolist(),
            "eef_quaternion_wxyz": quaternion.tolist(),
            "arm_joint_positions": joints[:7].tolist(),
            "gripper_width_normalized": width,
            "gripper_aperture_state": aperture,
        }

    def _collect_vlamender_semantic_candidates(
        self,
        observation: dict[str, Any],
        prompts: list[str],
        *,
        max_candidates_per_prompt: int = 5,
        min_score: float = 0.05,
    ) -> tuple[list[dict[str, Any]], list[str]]:
        robot = self.get_robot_state(observation)
        eef = np.asarray(robot["eef_position"], dtype=np.float64)
        candidates: list[dict[str, Any]] = []
        queried_cameras: list[str] = []
        for camera_name in ("agentview", "robot0_eye_in_hand"):
            camera = observation.get(camera_name)
            if not isinstance(camera, dict):
                continue
            try:
                rgb = np.asarray(camera["images"]["rgb"], dtype=np.uint8)
                depth = np.asarray(camera["images"]["depth"], dtype=np.float64)
                if depth.ndim == 3:
                    depth = depth[:, :, 0]
                intrinsics = np.asarray(camera["intrinsics"], dtype=np.float64).reshape(3, 3)
                extrinsics = np.asarray(camera["pose_mat"], dtype=np.float64).reshape(4, 4)
            except Exception:
                continue
            queried_cameras.append(camera_name)
            for prompt in prompts:
                try:
                    proposals = list(self.segment_sam3_text_prompt(rgb, prompt) or [])
                except Exception:
                    proposals = []
                proposals.sort(key=lambda item: float(item.get("score", 0.0)), reverse=True)
                for proposal in proposals[:max_candidates_per_prompt]:
                    score = float(proposal.get("score", 0.0))
                    if not np.isfinite(score) or score < min_score:
                        continue
                    try:
                        mask = np.asarray(proposal["mask"], dtype=np.uint8)
                        if mask.shape != depth.shape:
                            continue
                        points = np.asarray(
                            self.mask_to_world_points(mask, depth, intrinsics, extrinsics),
                            dtype=np.float64,
                        ).reshape(-1, 3)
                    except Exception:
                        continue
                    points = points[np.all(np.isfinite(points), axis=1)]
                    if len(points) < 15:
                        continue
                    center = np.median(points, axis=0)
                    lower = np.percentile(points, 5, axis=0)
                    upper = np.percentile(points, 95, axis=0)
                    extent = upper - lower
                    distances = np.linalg.norm(points - eef[None, :], axis=1)
                    near_point_distance = float(np.percentile(distances, 5))
                    center_distance = float(np.linalg.norm(center - eef))
                    xy_distance = float(np.linalg.norm(center[:2] - eef[:2]))
                    relative_z = float(center[2] - eef[2])
                    near_eef = bool(
                        near_point_distance <= 0.105
                        or (
                            xy_distance <= 0.14
                            and -0.28 <= relative_z <= 0.10
                            and center_distance <= 0.28
                        )
                    )
                    candidates.append(
                        {
                            "camera": camera_name,
                            "prompt": prompt,
                            "score": score,
                            "center": center,
                            "extent": extent,
                            "points": points,
                            "near_point_distance": near_point_distance,
                            "center_distance": center_distance,
                            "xy_distance": xy_distance,
                            "relative_z": relative_z,
                            "near_eef": near_eef,
                        }
                    )
        return candidates, queried_cameras

    def _estimate_grasp_state_with_candidates(
        self,
        observation: dict[str, Any],
        object_prompts: Any,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        prompts = self._normalize_vlamender_prompts(object_prompts)
        robot = self.get_robot_state(observation)
        candidates, queried_cameras = self._collect_vlamender_semantic_candidates(
            observation, prompts
        )
        near = [item for item in candidates if item["near_eef"]]
        near.sort(
            key=lambda item: (
                item["near_point_distance"],
                item["center_distance"],
                -item["score"],
            )
        )
        detected_cameras = sorted({item["camera"] for item in candidates})
        near_cameras = sorted({item["camera"] for item in near})
        width = float(robot["gripper_width_normalized"])
        aperture = str(robot["gripper_aperture_state"])
        evidence: list[str] = [
            f"gripper_width={width:.6f}",
            f"gripper_aperture={aperture}",
            f"queried_cameras={','.join(queried_cameras) or 'none'}",
            f"semantic_candidates={len(candidates)}",
            f"near_eef_candidates={len(near)}",
        ]

        if aperture == "open":
            state = "not_held"
            confidence = 0.98 if near else 0.93 if candidates else 0.80
            evidence.append(
                "a fully open parallel-jaw gripper cannot retain the requested object"
            )
        elif near:
            state = "held"
            confidence = 0.96 if len(near_cameras) >= 2 else 0.88
            evidence.append("semantic object geometry is adjacent to the end effector")
        else:
            state = "unknown"
            confidence = 0.30
            evidence.append(
                "a closed or intermediate gripper without near-object evidence is ambiguous"
            )

        nearest = near[0] if near else None
        public_nearest = None
        if nearest is not None:
            public_nearest = {
                "camera": nearest["camera"],
                "prompt": nearest["prompt"],
                "score": nearest["score"],
                "center": nearest["center"].tolist(),
                "extent": nearest["extent"].tolist(),
                "near_point_distance": nearest["near_point_distance"],
                "center_distance": nearest["center_distance"],
                "xy_distance": nearest["xy_distance"],
                "relative_z": nearest["relative_z"],
            }
        return (
            {
                "state": state,
                "confidence": confidence,
                "gripper_width_normalized": width,
                "gripper_aperture_state": aperture,
                "queried_cameras": queried_cameras,
                "detected_cameras": detected_cameras,
                "near_eef_cameras": near_cameras,
                "semantic_candidates_found": len(candidates),
                "near_eef_candidates": len(near),
                "nearest_object": public_nearest,
                "evidence": evidence,
            },
            candidates,
        )

    def estimate_grasp_state(
        self,
        observation: dict[str, Any],
        object_prompts: Any,
    ) -> dict[str, Any]:
        """Estimate ``held``, ``not_held`` or ``unknown`` from public observations only.

        Gripper width is only one piece of evidence.  The estimator also grounds the requested
        semantic object in agent and wrist views, projects masks into world geometry, and checks
        object-to-EEF proximity.  Closed-without-geometry is deliberately ``unknown`` rather than
        proof of a grasp, and uncertainty must preserve the current gripper command.

        The internally detected masks are state-estimation evidence only.  Manipulation code must
        still call the normal SAM3 API and ``commit_target_mask`` for its chosen target instance.
        """
        result, _ = self._estimate_grasp_state_with_candidates(observation, object_prompts)
        return result

    def _vlamender_open_raw(self) -> None:
        super().open_gripper()

    def _vlamender_goto_grasp_pose(
        self,
        position: np.ndarray,
        quaternion_wxyz: np.ndarray,
    ) -> None:
        """Move only forward to the grasp pose on the active controller."""
        self.goto_pose(position, quaternion_wxyz)

    def _vlamender_close_raw(self) -> None:
        initial_width: float | None = None
        try:
            initial_robot = self.get_robot_state(self.get_observation())
            initial_width = float(initial_robot["gripper_width_normalized"])
        except Exception:
            pass
        self._env._set_gripper(0.0)
        closed_streak = 0
        contact_streak = 0
        executed_steps = 0
        last_width: float | None = None
        previous_width = initial_width
        total_closure = 0.0
        reason = "maximum_settle_fallback"
        for _ in range(self._VLAMENDER_GRASP_CLOSE_SETTLE_STEPS):
            self._env._step_once()
            executed_steps += 1
            try:
                robot = self.get_robot_state(self.get_observation())
                last_width = float(robot["gripper_width_normalized"])
                width_delta = (
                    float("inf")
                    if previous_width is None
                    else abs(last_width - previous_width)
                )
                total_closure = (
                    0.0
                    if initial_width is None
                    else max(0.0, initial_width - last_width)
                )
                closed_streak = (
                    closed_streak + 1
                    if robot["gripper_aperture_state"] == "closed"
                    else 0
                )
                contact_streak = (
                    contact_streak + 1
                    if (
                        robot["gripper_aperture_state"] == "intermediate"
                        and total_closure >= self._VLAMENDER_GRASP_CONTACT_MIN_CLOSURE
                        and width_delta <= self._VLAMENDER_GRASP_CONTACT_WIDTH_DELTA
                    )
                    else 0
                )
                previous_width = last_width
            except Exception:
                # Observation failure cannot weaken the old conservative
                # behavior: retain the close command for the full fallback.
                closed_streak = 0
                contact_streak = 0
            if (
                executed_steps >= self._VLAMENDER_GRASP_CLOSE_MIN_STEPS
                and closed_streak >= self._VLAMENDER_GRASP_CLOSE_STABLE_STEPS
            ):
                reason = "observed_closed_aperture_stable"
                break
            if (
                executed_steps >= self._VLAMENDER_GRASP_CONTACT_MIN_STEPS
                and contact_streak >= self._VLAMENDER_GRASP_CONTACT_STABLE_STEPS
            ):
                reason = "observed_contact_width_stable"
                break
        self._vlamender_last_close_audit = {
            "executed_steps": executed_steps,
            "maximum_steps": self._VLAMENDER_GRASP_CLOSE_SETTLE_STEPS,
            "closed_streak": closed_streak,
            "contact_streak": contact_streak,
            "initial_gripper_width_normalized": initial_width,
            "last_gripper_width_normalized": last_width,
            "total_closure_normalized": total_closure,
            "reason": reason,
        }
        print("[vlamender_close_settle] " + repr(self._vlamender_last_close_audit), flush=True)

    def commit_target_mask(
        self,
        rgb: np.ndarray,
        candidate: dict[str, Any],
        role: str,
    ) -> dict[str, Any]:
        """Audit a SAM3 choice and register its exact mask for geometry grounding."""
        committed = super().commit_target_mask(rgb, candidate, role)
        if not hasattr(self, "_vlamender_mask_commits"):
            self._vlamender_mask_commits = {}
        commit_id = f"target_mask_{len(self._vlamender_mask_commits):03d}"
        self._vlamender_mask_commits[commit_id] = {
            "mask": np.asarray(committed["mask"], dtype=bool).copy(),
            "role": str(committed["role"]),
        }
        committed["target_mask_commit_id"] = commit_id
        return committed

    def ground_placement_target(
        self,
        observation: dict[str, Any],
        camera_name: str,
        committed_target: dict[str, Any],
    ) -> dict[str, Any]:
        """Ground one audited target mask into stable world geometry for later release checks.

        Placement targets are commonly occluded by the held object at release time. Re-running a
        text prompt at that point can select a different instance and make a correct release look
        metres away. This API accepts only the normalized result of ``commit_target_mask``,
        projects its mask using the supplied observation, and stores the geometry internally. The
        returned identifier is an opaque handle: ``guarded_open_gripper`` ignores caller-provided
        coordinates and resolves the original stored geometry.
        """
        if not isinstance(observation, dict):
            raise ValueError("observation must be a dictionary returned by get_observation")
        camera_key = str(camera_name).strip()
        camera = observation.get(camera_key)
        if not isinstance(camera, dict):
            raise ValueError(f"observation does not contain camera {camera_key!r}")
        if not isinstance(committed_target, dict) or "mask" not in committed_target:
            raise ValueError("committed_target must be returned by commit_target_mask")
        role = str(committed_target.get("role", "")).strip()
        if not role:
            raise ValueError("committed_target is missing its audited role")
        mask_commit_id = committed_target.get("target_mask_commit_id")
        registered = getattr(self, "_vlamender_mask_commits", {}).get(mask_commit_id)
        if registered is None:
            raise ValueError("committed_target was not returned by commit_target_mask in this run")
        try:
            rgb = np.asarray(camera["images"]["rgb"], dtype=np.uint8)
            depth = np.asarray(camera["images"]["depth"], dtype=np.float64)
            if depth.ndim == 3:
                depth = depth[:, :, 0]
            intrinsics = np.asarray(camera["intrinsics"], dtype=np.float64).reshape(3, 3)
            extrinsics = np.asarray(camera["pose_mat"], dtype=np.float64).reshape(4, 4)
            mask = np.asarray(committed_target["mask"], dtype=np.uint8)
        except Exception as exc:
            raise ValueError("camera calibration or committed target mask is malformed") from exc
        if rgb.ndim != 3 or mask.shape != rgb.shape[:2] or not np.any(mask):
            raise ValueError("committed target mask must be non-empty and match the camera image")
        if role != registered["role"] or not np.array_equal(mask.astype(bool), registered["mask"]):
            raise ValueError("committed target mask or role was modified after audit")
        points = np.asarray(
            self.mask_to_world_points(mask, depth, intrinsics, extrinsics), dtype=np.float64
        ).reshape(-1, 3)
        points = points[np.all(np.isfinite(points), axis=1)]
        if len(points) < 15:
            raise ValueError("committed target mask has insufficient valid depth geometry")
        lower = np.percentile(points, 5, axis=0)
        upper = np.percentile(points, 95, axis=0)
        target_id = f"placement_target_{len(getattr(self, '_vlamender_grounded_targets', {})):03d}"
        if not hasattr(self, "_vlamender_grounded_targets"):
            self._vlamender_grounded_targets = {}
        stored = {
            "camera": camera_key,
            "prompt": f"committed:{role}",
            "score": float(committed_target.get("score", 0.0)),
            "center": np.median(points, axis=0),
            "extent": upper - lower,
            "points": points,
            "committed": True,
        }
        self._vlamender_grounded_targets[target_id] = stored
        return {
            "schema_version": "vlamender_placement_target_v1",
            "target_commit_id": target_id,
            "role": role,
            "camera": camera_key,
            "center": stored["center"].tolist(),
            "extent": stored["extent"].tolist(),
        }

    def _resolve_grounded_target(self, target_commit: Any) -> dict[str, Any] | None:
        if target_commit is None:
            return None
        target_id = (
            target_commit.get("target_commit_id")
            if isinstance(target_commit, dict)
            else target_commit
        )
        if not isinstance(target_id, str) or not target_id:
            return None
        return getattr(self, "_vlamender_grounded_targets", {}).get(target_id)

    @staticmethod
    def _vlamender_placement_evaluations(
        object_candidates: list[dict[str, Any]],
        target_candidates: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Score conservative object-on-target geometry without trusting one broad mask."""
        evaluations: list[dict[str, Any]] = []
        for object_candidate in object_candidates:
            object_extent = np.asarray(object_candidate["extent"], dtype=np.float64)
            object_radius = float(max(0.015, 0.5 * max(object_extent[:2])))
            object_height = float(max(0.01, object_extent[2]))
            object_bottom = float(np.percentile(object_candidate["points"][:, 2], 5))
            for target in target_candidates:
                target_extent = np.asarray(target["extent"], dtype=np.float64)
                target_radius = float(max(0.015, 0.5 * max(target_extent[:2])))
                target_surface = float(np.percentile(target["points"][:, 2], 85))
                xy_distance = float(
                    np.linalg.norm(object_candidate["center"][:2] - target["center"][:2])
                )
                z_clearance = object_bottom - target_surface
                fit_margin = target_radius - object_radius
                # Keep the geometry-derived value for diagnostics, but use one explicit total
                # release threshold. The target remains the initially committed target; this
                # tolerance only governs the live held-object-to-target XY residual.
                base_xy_limit = float(
                    np.clip(0.65 * max(0.0, fit_margin), 0.004, 0.03)
                )
                xy_limit = float(
                    FrankaLiberoApiReducedSkillLibrary._VLAMENDER_XY_RELEASE_LIMIT_M
                )
                xy_relaxation = max(0.0, xy_limit - base_xy_limit)
                max_z_clearance = float(np.clip(0.25 * object_height, 0.008, 0.02))
                target_scale_valid = bool(
                    target_radius >= 0.75 * object_radius
                    and target_radius <= max(0.20, 4.0 * object_radius)
                )
                target_below_object = bool(
                    target_surface
                    <= float(object_candidate["center"][2]) - max(0.01, 0.15 * object_height)
                )
                vertical_contact_ready = bool(
                    -0.005 <= z_clearance <= max_z_clearance
                )
                # Keep vertical proximity as a diagnostic, but do not make it a release gate.
                # The VLA-Mender policy remains responsible for choosing its descent height.
                release_ready = bool(
                    target_scale_valid
                    and target_below_object
                    and xy_distance <= xy_limit
                )
                evaluations.append(
                    {
                        "release_ready": release_ready,
                        "xy_distance": xy_distance,
                        "base_xy_limit": base_xy_limit,
                        "xy_relaxation": xy_relaxation,
                        "xy_limit": xy_limit,
                        "xy_limit_policy": "fixed_total_limit",
                        "z_clearance": z_clearance,
                        "max_z_clearance": max_z_clearance,
                        "object_radius": object_radius,
                        "target_radius": target_radius,
                        "target_scale_valid": target_scale_valid,
                        "target_below_object": target_below_object,
                        "vertical_contact_ready": vertical_contact_ready,
                        "object_camera": object_candidate["camera"],
                        "object_prompt": object_candidate.get("prompt"),
                        "object_center": np.asarray(
                            object_candidate["center"], dtype=np.float64
                        ).copy(),
                        "object_extent": object_extent.copy(),
                        "target_camera": target["camera"],
                        "target_prompt": target.get("prompt"),
                    }
                )
        evaluations.sort(
            key=lambda item: (
                not item["release_ready"],
                not item["target_scale_valid"],
                not item["target_below_object"],
                item["xy_distance"] / max(item["xy_limit"], 1e-6),
                not item["vertical_contact_ready"],
                abs(item["z_clearance"]),
            )
        )
        return evaluations

    def grasp_if_unheld(
        self,
        observation: dict[str, Any],
        object_prompts: Any,
        pregrasp_position: Any,
        grasp_position: Any,
        grasp_quaternion_wxyz: Any,
    ) -> dict[str, Any]:
        """Execute one grasp only when public evidence positively says the object is unheld.

        ``held`` returns ``already_held`` without any robot command.  ``unknown`` returns
        ``ambiguous_hold`` and also issues no command.  Only ``not_held`` may open for acquisition,
        move through the supplied observation-derived pregrasp/grasp poses, and close the gripper.
        """
        before = self.estimate_grasp_state(observation, object_prompts)
        if before["state"] == "held":
            return {
                "status": "already_held",
                "executed": False,
                "grasp_state_before": before,
            }
        if before["state"] != "not_held":
            return {
                "status": "ambiguous_hold",
                "executed": False,
                "grasp_state_before": before,
            }

        pregrasp = np.asarray(pregrasp_position, dtype=np.float64).reshape(3)
        grasp = np.asarray(grasp_position, dtype=np.float64).reshape(3)
        quaternion = np.asarray(grasp_quaternion_wxyz, dtype=np.float64).reshape(4)
        if not (
            np.all(np.isfinite(pregrasp))
            and np.all(np.isfinite(grasp))
            and np.all(np.isfinite(quaternion))
        ):
            raise ValueError("grasp poses must contain finite values")
        quaternion_norm = float(np.linalg.norm(quaternion))
        if quaternion_norm <= 1e-12:
            raise ValueError("grasp quaternion must be non-zero")
        quaternion /= quaternion_norm
        pregrasp_motion_skipped = False
        grasp_motion_skipped = False
        resume_audit: dict[str, Any] = {"resume_phase": "unclassified"}
        prompts = object_prompts if isinstance(object_prompts, (list, tuple)) else [object_prompts]
        handle_alignment = any("handle" in str(prompt).casefold() for prompt in prompts)
        previous_handle_alignment = bool(
            getattr(self, "_handle_alignment_active", False)
        )
        if handle_alignment and hasattr(self, "_handle_alignment_active"):
            self._handle_alignment_active = True
        try:
            current_observation = self.get_observation()
            current_state = self.get_robot_state(current_observation)
            current_position = np.asarray(
                current_state["motion_target_position"], dtype=np.float64
            )
            current_quaternion = np.asarray(
                current_state["eef_quaternion_wxyz"], dtype=np.float64
            )
            resume_audit = _classify_grasp_resume_phase(
                current_position,
                current_quaternion,
                pregrasp,
                grasp,
                quaternion,
                position_tolerance_m=self._GOTO_POSITION_TOLERANCE_M,
                orientation_tolerance_rad=self._GOTO_ORIENTATION_TOLERANCE_RAD,
                contact_ready_tolerance_m=(
                    self._VLAMENDER_GRASP_CONTACT_READY_TOLERANCE_M
                ),
            )
            pregrasp_motion_skipped = resume_audit["resume_phase"] != "pregrasp_required"
            grasp_motion_skipped = resume_audit["resume_phase"] == "contact_ready"
            print("[vlamender_grasp_resume] " + repr(resume_audit), flush=True)

            # Preserve partial closure and contact at a progressed handoff.
            # Opening here would itself erase VLA progress before the repair
            # policy gets a chance to continue it.
            if (
                resume_audit["resume_phase"] == "pregrasp_required"
                and before["gripper_aperture_state"] != "open"
            ):
                self._vlamender_open_raw()
            if not pregrasp_motion_skipped:
                self.goto_pose(pregrasp, quaternion)
            # Re-read the public robot pose after the optional pregrasp move.
            # Observation-derived recovery may intentionally bind both poses
            # to an already achieved safe contact.  Re-solving IK for that
            # identical pose adds motion and consumes repair budget without
            # changing the commanded grasp, so close directly when both
            # position and orientation are already within the audited limits.
            grasp_observation = self.get_observation()
            grasp_state = self.get_robot_state(grasp_observation)
            grasp_current_position = np.asarray(
                grasp_state["motion_target_position"], dtype=np.float64
            )
            grasp_current_quaternion = np.asarray(
                grasp_state["eef_quaternion_wxyz"], dtype=np.float64
            )
            grasp_position_error = float(
                np.linalg.norm(grasp_current_position - grasp)
            )
            grasp_quaternion_dot = float(
                np.clip(abs(np.dot(grasp_current_quaternion, quaternion)), 0.0, 1.0)
            )
            grasp_orientation_error = float(
                2.0 * np.arccos(grasp_quaternion_dot)
            )
            grasp_motion_skipped = bool(
                grasp_motion_skipped
                or (
                    grasp_position_error <= self._GOTO_POSITION_TOLERANCE_M
                    and grasp_orientation_error <= self._GOTO_ORIENTATION_TOLERANCE_RAD
                )
            )
            if not grasp_motion_skipped:
                self._vlamender_goto_grasp_pose(grasp, quaternion)
            self._vlamender_close_raw()
        except (RuntimeError, ValueError) as exc:
            return {
                "status": "motion_failed",
                "executed": True,
                "failure_reason": str(exc),
                "pregrasp_motion_skipped": pregrasp_motion_skipped,
                "grasp_motion_skipped": grasp_motion_skipped,
                "resume_audit": resume_audit,
                "grasp_state_before": before,
            }
        finally:
            if handle_alignment and hasattr(self, "_handle_alignment_active"):
                self._handle_alignment_active = previous_handle_alignment

        after_observation = self.get_observation()
        after = self.estimate_grasp_state(after_observation, object_prompts)
        return {
            "status": "grasped" if after["state"] == "held" else "grasp_unverified",
            "executed": True,
            "pregrasp_motion_skipped": pregrasp_motion_skipped,
            "grasp_motion_skipped": grasp_motion_skipped,
            "resume_audit": resume_audit,
            "close_audit": dict(getattr(self, "_vlamender_last_close_audit", {})),
            "grasp_state_before": before,
            "grasp_state_after": after,
        }

    def guarded_open_gripper(
        self,
        observation: dict[str, Any],
        object_prompts: Any,
        target_prompts: Any,
        phase_id: str,
        target_commit: Any = None,
        include_live_target_candidates: bool = False,
    ) -> dict[str, Any]:
        """Open only for an observation-confirmed placement release.

        The phase name must explicitly be a release phase, the semantic object must be estimated as
        held near the EEF, and current object/target geometry must show the object aligned above the
        target. A valid committed target is authoritative by default; live target candidates are
        collected only without a commit or when ``include_live_target_candidates`` is explicitly
        enabled. Missing or conflicting evidence fails closed without issuing a gripper command.
        """
        phase = str(phase_id).strip().lower()
        if "release" not in phase or "grasp" in phase or "acquisition" in phase:
            return {
                "status": "blocked_invalid_phase",
                "executed": False,
                "phase_id": str(phase_id),
            }

        grasp_state, object_candidates = self._estimate_grasp_state_with_candidates(
            observation, object_prompts
        )
        if grasp_state["state"] == "unknown":
            return {
                "status": "blocked_ambiguous_grasp",
                "executed": False,
                "phase_id": str(phase_id),
                "grasp_state": grasp_state,
            }

        grounded_target = self._resolve_grounded_target(target_commit)
        if target_commit is not None and grounded_target is None:
            return {
                "status": "blocked_invalid_target_commit",
                "executed": False,
                "phase_id": str(phase_id),
                "grasp_state": grasp_state,
            }
        if grounded_target is not None:
            # Committed placement geometry is authoritative by default, preventing a fresh
            # semantic false positive from switching target instances during descent. Live target
            # candidates remain available only as explicit, opt-in corroborating evidence.
            targets = [grounded_target]
            if include_live_target_candidates:
                live_targets, _ = self._collect_vlamender_semantic_candidates(
                    observation,
                    self._normalize_vlamender_prompts(target_prompts),
                )
                targets.extend(live_targets)
        else:
            # Compatibility fallback when the caller did not ground a visible target beforehand.
            targets, _ = self._collect_vlamender_semantic_candidates(
                observation,
                self._normalize_vlamender_prompts(target_prompts),
            )
        relevant_objects = (
            [item for item in object_candidates if item["near_eef"]]
            if grasp_state["state"] == "held"
            else object_candidates
        )
        if not relevant_objects or not targets:
            return {
                "status": "blocked_missing_geometry",
                "executed": False,
                "phase_id": str(phase_id),
                "grasp_state": grasp_state,
            }

        # A release retry must evaluate the same held-object instance as the
        # preceding blocked guard call. In particular, an eye-in-hand view can
        # expose a different, lower-distance semantic candidate after the EEF
        # moves. Track only public geometry relative to the observed EEF; no
        # simulator identity or private object state is involved.
        robot_state = self.get_robot_state(observation)
        current_eef = np.asarray(robot_state["eef_position"], dtype=np.float64)
        target_id = (
            target_commit.get("target_commit_id")
            if isinstance(target_commit, dict)
            else target_commit
        )
        continuity_key = (
            tuple(self._normalize_vlamender_prompts(object_prompts)),
            str(target_id) if target_id is not None else None,
            phase,
        )
        track = getattr(self, "_vlamender_guard_object_track", None)
        continuity_used = False
        continuity_error = None
        continuity_limit = None
        if isinstance(track, dict) and track.get("key") == continuity_key:
            previous_center = np.asarray(track["object_center"], dtype=np.float64)
            previous_extent = np.asarray(track["object_extent"], dtype=np.float64)
            previous_eef = np.asarray(track["eef_position"], dtype=np.float64)
            predicted_center = previous_center + (current_eef - previous_eef)
            previous_xy_scale = float(max(previous_extent[:2]))
            continuity_limit = float(max(0.025, 0.50 * previous_xy_scale))
            extent_limit = float(max(0.018, 0.50 * previous_xy_scale))
            continuous_candidates: list[tuple[float, dict[str, Any]]] = []
            for candidate in relevant_objects:
                if (
                    candidate.get("camera") != track.get("object_camera")
                    or candidate.get("prompt") != track.get("object_prompt")
                ):
                    continue
                center = np.asarray(candidate["center"], dtype=np.float64)
                extent = np.asarray(candidate["extent"], dtype=np.float64)
                prediction_error = float(np.linalg.norm(center - predicted_center))
                extent_error = float(np.linalg.norm(extent - previous_extent))
                if prediction_error <= continuity_limit and extent_error <= extent_limit:
                    continuous_candidates.append((prediction_error, candidate))
            if not continuous_candidates:
                return {
                    "status": "blocked_candidate_continuity",
                    "executed": False,
                    "phase_id": str(phase_id),
                    "grasp_state": grasp_state,
                    "candidate_continuity_used": True,
                    "candidate_continuity_valid": False,
                    "candidate_continuity_limit": continuity_limit,
                }
            continuous_candidates.sort(key=lambda item: item[0])
            continuity_error, continuous_object = continuous_candidates[0]
            relevant_objects = [continuous_object]
            continuity_used = True

        evaluations = self._vlamender_placement_evaluations(
            relevant_objects,
            targets,
        )
        best = evaluations[0]
        best["candidate_continuity_used"] = continuity_used
        best["candidate_continuity_valid"] = True
        best["candidate_continuity_error"] = continuity_error
        best["candidate_continuity_limit"] = continuity_limit
        if grasp_state["state"] == "held":
            self._vlamender_guard_object_track = {
                "key": continuity_key,
                "eef_position": current_eef.copy(),
                "object_center": np.asarray(best["object_center"], dtype=np.float64).copy(),
                "object_extent": np.asarray(best["object_extent"], dtype=np.float64).copy(),
                "object_camera": best.get("object_camera"),
                "object_prompt": best.get("object_prompt"),
            }
        if grasp_state["state"] == "not_held":
            return {
                "status": (
                    "already_released"
                    if best["release_ready"]
                    else "blocked_not_held_away_from_target"
                ),
                "executed": False,
                "phase_id": str(phase_id),
                "grasp_state": grasp_state,
                "placement_geometry": best,
            }
        if not best["release_ready"]:
            return {
                "status": "blocked_not_release_ready",
                "executed": False,
                "phase_id": str(phase_id),
                "grasp_state": grasp_state,
                "placement_geometry": best,
            }

        self._vlamender_open_raw()
        self._vlamender_guard_object_track = None
        after = self.get_robot_state(self.get_observation())
        return {
            "status": "opened",
            "executed": True,
            "phase_id": str(phase_id),
            "grasp_state": grasp_state,
            "placement_geometry": best,
            "robot_state_after": after,
        }

    # SKILL LIBRARY - Reusable Functions from LLM Robot Code Generation
    # ======================================================================
    # Source: reduced_api and reduced_api_exampleless experiments
    # Total unique functions analyzed: 182
    # Functions after filtering: 73
    # Minimum occurrence threshold: 2
    # ======================================================================

    # Here is a curated library of reusable robotics skills derived from the provided code generations.

    # I have categorized them into **Coordinate Transforms**, **Vision & Perception**, and **Geometry & Math**. I selected implementations that are vectorized (for performance), numerically stable (especially for quaternion conversion), and decoupled from specific environment dictionaries to ensure maximum reusability.

    ### 1. Category: Coordinate Transformations
    # These functions were the most frequent across all experiments (occurring >80 times in total). They are essential because planners often output matrices, but controllers (like `solve_ik`) often require quaternions.
    # **Why Reusable:** Converting between rotation matrices, quaternions, and homogeneous transformation matrices is a fundamental requirement for almost every manipulation task.

    def rotation_matrix_to_quaternion(self, R: np.ndarray) -> np.ndarray:
        """
        Convert a 3x3 rotation matrix to a unit quaternion [w, x, y, z].

        Implements the robust Sheppard's method (checking trace and diagonal elements)
        to avoid numerical instability when the trace is close to zero.

        Args:
            R: (3, 3) rotation matrix.

        Returns:
            np.array: [w, x, y, z] unit quaternion.

        """
        tr = np.trace(R)
        if tr > 0:
            S = np.sqrt(tr + 1.0) * 2
            w = 0.25 * S
            x = (R[2, 1] - R[1, 2]) / S
            y = (R[0, 2] - R[2, 0]) / S
            z = (R[1, 0] - R[0, 1]) / S
        elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
            S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
            w = (R[2, 1] - R[1, 2]) / S
            x = 0.25 * S
            y = (R[0, 1] + R[1, 0]) / S
            z = (R[0, 2] + R[2, 0]) / S
        elif R[1, 1] > R[2, 2]:
            S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
            w = (R[0, 2] - R[2, 0]) / S
            x = (R[0, 1] + R[1, 0]) / S
            y = 0.25 * S
            z = (R[1, 2] + R[2, 1]) / S
        else:
            S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
            w = (R[1, 0] - R[0, 1]) / S
            x = (R[0, 2] + R[2, 0]) / S
            y = (R[1, 2] + R[2, 1]) / S
            z = 0.25 * S
        return np.array([w, x, y, z])

    def decompose_transform(self, T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Decompose a 4x4 homogeneous transformation matrix into position and quaternion.

        Args:
            T: (4, 4) homogeneous transformation matrix.

        Returns:
            tuple:
                - position: (3,) np.array
                - quaternion: (4,) np.array [w, x, y, z]

        """
        position = T[:3, 3]
        R = T[:3, :3]
        quat = self.rotation_matrix_to_quaternion(R)
        return position, quat

    ### 2. Category: Vision & Perception (Depth to 3D)
    # These functions bridge the gap between 2D camera data and 3D robot actions. They are crucial for converting segmentation masks into grasp targets.

    # **Why Reusable:** They encapsulate the pinhole camera model math, handling intrinsics (projection) and extrinsics (camera pose), allowing the agent to reason in the World Frame.

    def depth_to_point_cloud(self, depth_img: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
        """
        Convert a depth image to a 3D point cloud in the Camera Frame.

        Args:
            depth_img: (H, W) depth map in meters.
            intrinsics: (3, 3) camera intrinsic matrix.

        Returns:
            np.array: (H, W, 3) image of 3D coordinates.

        """
        if depth_img.ndim == 3:
            depth_img = depth_img[:, :, 0]

        h, w = depth_img.shape
        fx = intrinsics[0, 0]
        fy = intrinsics[1, 1]
        cx = intrinsics[0, 2]
        cy = intrinsics[1, 2]

        # Vectorized grid generation
        y_grid, x_grid = np.mgrid[0:h, 0:w]

        z = depth_img
        x = (x_grid - cx) * z / fx
        y = (y_grid - cy) * z / fy

        return np.dstack((x, y, z))

    def mask_to_world_points(
        self, mask: np.ndarray, depth: np.ndarray, intrinsics: np.ndarray, extrinsics: np.ndarray
    ) -> np.ndarray:
        """
        Convert specific pixels defined by a binary mask into 3D points in the World Frame.

        Args:
            mask: (H, W) binary mask (0 or 1).
            depth: (H, W) depth map.
            intrinsics: (3, 3) camera intrinsics.
            extrinsics: (4, 4) camera-to-world pose matrix.

        Returns:
            np.array: (N, 3) array of valid 3D points in world coordinates.

        """
        # Get pixel coordinates
        ys, xs = np.where(mask > 0)
        if len(ys) == 0:
            return np.empty((0, 3))

        if depth.ndim == 3:
            depth = depth[:, :, 0]

        z_vals = depth[ys, xs]

        # Filter invalid depth
        valid = z_vals > 0
        ys = ys[valid]
        xs = xs[valid]
        z = z_vals[valid]

        fx = intrinsics[0, 0]
        fy = intrinsics[1, 1]
        cx = intrinsics[0, 2]
        cy = intrinsics[1, 2]

        # Deproject to Camera Frame
        x_cam = (xs - cx) * z / fx
        y_cam = (ys - cy) * z / fy

        # Stack to (N, 3)
        points_cam = np.stack([x_cam, y_cam, z], axis=-1)

        # Transform to World Frame
        # Create homogeneous coordinates (N, 4)
        points_cam_hom = np.hstack([points_cam, np.ones((len(points_cam), 1))])
        points_world_hom = (extrinsics @ points_cam_hom.T).T

        return points_world_hom[:, :3]

    def pixel_to_world_point(
        self, u: int, v: int, z: float, intrinsics: np.ndarray, extrinsics: np.ndarray
    ) -> np.ndarray:
        """
        Deproject a single pixel to a 3D world point.

        Args:
            u, v: Pixel coordinates (col, row).
            z: Depth at that pixel.
            intrinsics: (3, 3) matrix.
            extrinsics: (4, 4) matrix.

        Returns:
            np.array: [x, y, z] in world frame.

        """
        fx = intrinsics[0, 0]
        fy = intrinsics[1, 1]
        cx = intrinsics[0, 2]
        cy = intrinsics[1, 2]

        x_cam = (u - cx) * z / fx
        y_cam = (v - cy) * z / fy

        print(f"u: {u}, v: {v}, z: {z}")

        p_cam = np.array([x_cam, y_cam, z, 1.0])
        p_world = extrinsics @ p_cam
        return p_world[:3]

    ### 3. Category: Geometry & Math
    # These functions help manipulate 3D data once it has been extracted from the camera.

    # **Why Reusable:** The `transform_points` function is particularly useful because it handles both lists of points `(N, 3)` and organized point clouds `(H, W, 3)` via reshaping, making it a "do-it-all" spatial transformer.

    def transform_points(self, points: np.ndarray, transform_matrix: np.ndarray) -> np.ndarray:
        """
        Apply a 4x4 homogeneous transform to a set of 3D points.

        Args:
            points: (N, 3) or (H, W, 3) array of points.
            transform_matrix: (4, 4) homogeneous transformation matrix.

        Returns:
            np.array: Transformed points with same shape as input.

        """
        original_shape = points.shape
        # Flatten to (N, 3)
        points_reshaped = points.reshape(-1, 3)

        # Convert to homogeneous (N, 4)
        ones = np.ones((points_reshaped.shape[0], 1))
        points_hom = np.hstack((points_reshaped, ones))

        # Apply transform: (4,4) @ (4,N) -> (4,N) -> Transpose back to (N,4)
        points_transformed = (transform_matrix @ points_hom.T).T

        # Return to (N, 3) and original shape
        return points_transformed[:, :3].reshape(original_shape)

    def interpolate_segment(
        self, p1: np.ndarray, p2: np.ndarray, step: float = 0.03
    ) -> list[np.ndarray]:
        """
        Generate waypoints along a line segment between two 3D points.

        Args:
            p1: Start point (3,).
            p2: End point (3,).
            step: Distance between waypoints in meters.

        Returns:
            list[np.ndarray]: List of points including p1 and p2.

        """
        dist = np.linalg.norm(p2 - p1)
        if dist < 1e-6:
            return [p1]

        num_points = int(np.ceil(dist / step))
        # Using linspace to ensure we hit the start and end exactly
        return [p1 + (p2 - p1) * t for t in np.linspace(0, 1, num_points + 1)]

    def normalize_vector(self, v: np.ndarray) -> np.ndarray:
        """
        Normalize a vector to unit length.

        Args:
            v: (3,) vector.

        Returns:
            np.array: (3,) unit vector.
        """
        norm = np.linalg.norm(v)
        if norm < 1e-6:
            return v
        return v / norm

    # ### 4. Category: Grasp Heuristics
    # A reusable heuristic for filtering grasps generated by learned models (like Contact-GraspNet).

    def select_top_down_grasp(
        self,
        grasps: np.ndarray,
        scores: np.ndarray,
        cam_to_world: np.ndarray,
        vertical_threshold: float = 0.8,
    ) -> tuple:
        """
        Selects the best grasp that aligns the gripper vertically (Top-Down).

        Args:
            grasps: (N, 4, 4) Grasp poses in camera frame.
            scores: (N,) Grasp scores.
            cam_to_world: (4, 4) Extrinsics matrix.
            vertical_threshold: Dot product threshold (1.0 is perfectly vertical).

        Returns:
            tuple: (best_grasp_world_matrix, best_score) or (None, -inf)
        """
        best_grasp = None
        best_score = -np.float64("inf")

        # World Z axis (vertical)
        world_z = np.array([0, 0, 1])

        for i, g_camera in enumerate(grasps):
            # Transform grasp to world frame
            g_world = cam_to_world @ g_camera

            # Extract rotation
            R = g_world[:3, :3]

            # Assuming Gripper Z or Y is the approach vector depending on gripper definition.
            # For Franka/Robotiq, the approach vector is usually the Z-axis of the end effector.
            gripper_approach = R[:, 2]

            # Check alignment with negative World Z (pointing down)
            # Dot product should be close to -1 for top-down
            alignment = -np.dot(gripper_approach, world_z)

            if alignment > vertical_threshold:
                if scores[i] > best_score:
                    best_score = scores[i]
                    best_grasp = g_world

        return best_grasp, best_score
