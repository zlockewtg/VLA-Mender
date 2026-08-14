from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any

os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"


@dataclass
class LiberoHandle:
    env: Any
    suite_name: str
    task_id: int
    task_language: str
    init_states: Any
    task: Any | None = None
    asset_provenance: dict[str, Any] | None = None

    def reset(self, seed: int | None = None) -> tuple[Any, dict[str, Any]]:
        self.env.seed(seed)
        obs = self.env.reset()
        if self.init_states is not None:
            obs = self.env.set_init_state(self.init_states[0])
        return obs, {}

    def step(self, action: list[float]) -> tuple[Any, float, bool, dict[str, Any]]:
        obs, reward, done, info = self.env.step(action)
        return obs, float(reward), bool(done), info


def _extract_language_from_bddl(bddl_path: str) -> str | None:
    try:
        with open(bddl_path, "r") as f:
            content = f.read()
        match = re.search(r"\(:language\s+(.*?)\)", content, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
    except Exception as e:
        print(f"Warning: Could not extract language from {bddl_path}: {e}")
    return None


def load_libero_task(
    suite_name: str,
    task_id: int,
    cam_w: int = 128,
    cam_h: int = 128,
    controller: str = "OSC_POSE",
    horizon: int = 1000,
    control_freq: int = 20,
    camera_depths: bool = True,
    init_states_root: str | None = None,
) -> LiberoHandle:
    """Load a LIBERO task using OffScreenRenderEnv.

    Reference: https://github.com/Lifelong-Robot-Learning/LIBERO
    """
    # Prefer vendored third_party/LIBERO if present, then fall back to installed package
    import os
    import sys

    here = os.path.dirname(os.path.abspath(__file__))
    vendor_root = os.path.normpath(os.path.join(here, "..", "..", "third_party", "LIBERO-PRO"))
    if os.path.isdir(vendor_root) and vendor_root not in sys.path:
        sys.path.append(vendor_root)
    try:
        from libero import benchmark  # type: ignore[import-not-found]
        from libero.envs import OffScreenRenderEnv  # type: ignore[import-not-found]
        from libero.utils import get_libero_path  # type: ignore[import-not-found]
    except Exception as e:  # pragma: no cover - optional dependency
        raise ModuleNotFoundError(
            "LIBERO not available; add submodule or run `uv sync --extra libero`."
        ) from e
    import os

    # setting help=True will print the available benchmarks
    benchmark_dict = benchmark.get_benchmark_dict(help=False)
    task_suite = benchmark_dict[suite_name]()
    task = task_suite.get_task(task_id)

    bddl_file_path = os.path.join(
        get_libero_path("bddl_files"), task.problem_folder, task.bddl_file
    )

    if not os.path.exists(bddl_file_path):
        # Fallback: try to locate BDDL files relative to this file
        # This handles cases where get_libero_path returns incorrect relative paths
        here = os.path.dirname(os.path.abspath(__file__))
        # Path: capx/integrations/libero/../../third_party/LIBERO-PRO/libero/libero/bddl_files
        fallback_bddl_root = os.path.abspath(
            os.path.join(here, "..", "..", "third_party", "LIBERO-PRO", "libero", "libero", "bddl_files")
        )
        fallback_path = os.path.join(fallback_bddl_root, task.problem_folder, task.bddl_file)

        if os.path.exists(fallback_path):
            print(f"Found BDDL file at fallback path: {fallback_path}")
            bddl_file_path = fallback_path
        else:
            print(f"Error: BDDL file not found at {bddl_file_path} OR {fallback_path}")

    env_args = {
        "bddl_file_name": bddl_file_path,
        "camera_heights": cam_h,
        "camera_widths": cam_w,
        "controller": controller,
        "horizon": horizon,
        "control_freq": control_freq,
        "camera_depths": camera_depths,
    }
    env = OffScreenRenderEnv(**env_args)
    env.seed(0)

    # Try to extract language from BDDL file directly
    task_language = _extract_language_from_bddl(bddl_file_path)
    if not task_language:
        task_language = task.language

    official_init_states_path = os.path.join(
        get_libero_path("init_states"), task.problem_folder, task.init_states_file
    )
    override_init_states_path = None
    if init_states_root:
        override_init_states_path = os.path.join(
            os.path.abspath(os.path.expanduser(init_states_root)),
            task.problem_folder,
            task.init_states_file,
        )

    if override_init_states_path and os.path.exists(override_init_states_path):
        import torch

        init_states = torch.load(override_init_states_path, weights_only=False)
        init_states_path = override_init_states_path
        init_states_kind = "configured_override"
        print(
            f"Loaded {len(init_states)} override init states for task {task_id} "
            f"in suite {suite_name}: {override_init_states_path}"
        )
    else:
        # Libero's get_task_init_states uses get_libero_path("init_states") internally.
        # Fall back to an explicit vendored path if the configured LIBERO path is stale.
        try:
            init_states = task_suite.get_task_init_states(task_id)
            init_states_path = official_init_states_path
            init_states_kind = "official_libero_assets"
            print(f"Loaded init states for task {task_id} in suite {suite_name}")
        except (FileNotFoundError, OSError):
            print(f"Warning: Could not load init states for task {task_id} in suite {suite_name}")
            init_states_path = official_init_states_path
            if not os.path.exists(init_states_path):
                here = os.path.dirname(os.path.abspath(__file__))
                fallback_init_root = os.path.abspath(
                    os.path.join(
                        here,
                        "..",
                        "..",
                        "third_party",
                        "LIBERO-PRO",
                        "libero",
                        "libero",
                        "init_files",
                    )
                )
                fallback_init_path = os.path.join(
                    fallback_init_root, task.problem_folder, task.init_states_file
                )

                if os.path.exists(fallback_init_path):
                    print(f"Found init states file at fallback path: {fallback_init_path}")
                    init_states_path = fallback_init_path
                else:
                    print(
                        f"Error: Init states file not found at {official_init_states_path} "
                        f"OR {fallback_init_path}"
                    )
                    raise

            import torch

            init_states = torch.load(init_states_path, weights_only=False)
            init_states_kind = "official_libero_assets_fallback"

    if init_states is None or len(init_states) == 0:
        env.close()
        raise RuntimeError(
            f"no init states available for suite={suite_name} task_id={task_id}"
        )

    if override_init_states_path and not os.path.exists(override_init_states_path):
        print(
            f"No override init-state file for suite={suite_name} task_id={task_id}; "
            "using official LIBERO assets"
        )

    provenance = {
        "kind": init_states_kind,
        "bddl": os.path.abspath(bddl_file_path),
        "init_states": os.path.abspath(init_states_path),
        "init_state_count": len(init_states),
    }

    handle = LiberoHandle(
        env=env,
        suite_name=suite_name,
        task_id=task_id,
        task_language=task_language,
        init_states=init_states,
        task=task,
        asset_provenance=provenance,
    )
    return handle
