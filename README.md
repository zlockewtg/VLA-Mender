# VLA-Mender

VLA-Mender follows the CaP-X dependency layout: source repositories live in
`third_party/` as Git submodules, and `uv` installs those checked-out sources
editable through `[tool.uv.sources]`. The parent repository pins the exact
source commits, so a fresh checkout is reproducible.

## Install

Clone with submodules (or initialize them after cloning), then run the bootstrap
script:

```bash
git clone --recurse-submodules <VLA-Mender-url>
cd VLA-Mender
bash scripts/bootstrap_environment.sh
```

For an existing checkout, the script runs `git submodule sync --recursive` and
`git submodule update --init --recursive` before `uv sync`. It creates
`.venv-libero` with Python 3.12, sets the CMake compatibility policy required by
`egl-probe`, installs the local sources editable, and runs the environment
verification. Use `VLA_MENDER_VENV=/path/to/venv` or `--venv PATH` to select an
existing environment. Add `--no-server-smoke` when only installation and import
checks are wanted. The bootstrap also provisions both tool checkpoints at their
fixed repository-relative paths before running verification.

The equivalent manual commands are:

```bash
git submodule update --init --recursive
uv venv .venv-libero --python 3.12
source .venv-libero/bin/activate
export CMAKE_POLICY_VERSION_MINIMUM=3.5
export UV_LINK_MODE=copy
uv sync --active --locked --extra libero
python scripts/install_tool_checkpoints.py
python scripts/verify_environment.py --libero
```

## Checkpoints and assets

Tool services always load checkpoints from these repository-relative paths:

```text
third_party/sam3/checkpoints/sam3.pt
third_party/contact_graspnet_pytorch/checkpoints/contact_graspnet/checkpoints/model.pt
```

`scripts/bootstrap_environment.sh` runs `scripts/install_tool_checkpoints.py`
after `uv sync`. The installer keeps valid files already present at those
locations, otherwise it copies SAM3 from a local Hugging Face cache or downloads
`facebook/sam3`, and verifies the Contact-GraspNet checkpoint installed by its
submodule. Runtime service startup does not search external caches or consume
checkpoint-path environment variables.

Check all expected assets with:

```bash
python scripts/check_runtime_assets.py
```

The source pins and runtime-asset policy are recorded in
[`third_party/PROVENANCE.md`](third_party/PROVENANCE.md).

## Tool services

Start SAM3, Contact-GraspNet, and PyRoKi in the background using the same
launcher pattern as CaP-X:

```bash
python scripts/verify_tools.py --server-smoke
# or: uv run --no-sync --active vla-mender/tools/launch_servers.py --profile default
```

The smoke check waits for `/health` on ports 8114, 8115, and 8116, then leaves
healthy services running under `.runtime/tool_servers`. Logs and process state
are kept there; re-running the check reuses healthy owned services. Stop them
with:

```bash
python scripts/verify_tools.py --stop-servers
```

If either repository-local checkpoint is removed, rerun:

```bash
python scripts/install_tool_checkpoints.py
```

## uv path troubleshooting

If Bash reports a missing path such as `/opt/venv/openvla/bin/uv` after switching
environments, clear the shell's cached command location. `uv` is a system
executable and does not need to be installed inside the virtual environment:

```bash
hash -r
source .venv-libero2/bin/activate
command -v uv                    # should resolve to /usr/local/bin/uv
uv run --no-sync --active vla-mender/tools/launch_servers.py --profile default
```

The explicit fallback is also safe:

```bash
/usr/local/bin/uv run --no-sync --active vla-mender/tools/launch_servers.py --profile default
```

## Generic LIBERO pre-repair

The generic infrastructure lives in `vla-mender/src/workflow` and `vla-mender/src/run/pre_repair`. It ends at validated
reset states and `repair_jobs.json`; it deliberately contains no Task 9/Task 0
configuration, repair policy, training, evolution, or repair execution.

The compact implementation layout is:

```text
vla-mender/src/workflow/
  parameters.py                 # experiment contract and resolved YAML
  pipeline.py                   # rollout -> agent handoff -> reset jobs
  libero_runtime.py             # shared LIBERO/MuJoCo/controller bridge
  rollout/action_noise.py       # deterministic OSC rollout perturbations
  rollout/state_provider.py     # official/randomized/manifest state core
  rollout/runner.py             # shared state/trial seeds and batch execution
  rollout/evaluator.py          # shared single-episode semantics
  rollout/rollout.py            # pre-repair workers and JSON/video adapter
  failure_diagnosis/failure_diagnosis.py
                                 # evidence, diagnosis, windows, replay, reset bank
vla-mender/src/run/pre_repair/
  experiment.example.yaml        # task/run contract template
  generate_prompt.py             # config -> rendered task prompt
  prompt.md                      # reusable pre-repair prompt template
```

Copy `vla-mender/src/run/pre_repair/experiment.example.yaml` and set only task-specific
values. The core outcome-changing settings are randomized/official state count,
20-Hz control frequency, max rollout steps, policy seed, GPUs/workers,
source/target control spaces, reset-frame count, window stride, and reset
dynamics (`preserve_full_state` or `quiescent_osc`).

Standalone checkpoint evaluation lives in
`vla-mender/scripts/eval`. Its CLI worker and the pre-repair workflow both
reuse `workflow.rollout.state_provider`, `workflow.rollout.runner`, and
`workflow.rollout.evaluator`; only their artifact adapters differ. This keeps
state validation, seeds, stabilization, action dispatch, binary-gripper, and
native-success semantics aligned.

The next stage—retained repair selection to LeRobot dataset and OpenPI
post-training—is available under `vla-mender/src/run/dataset`. One strict YAML
resolves source lineage, builder parameters, trainable action-chunk filtering,
the pinned OpenPI checkout/environment, and training hyperparameters. See
[`vla-mender/src/run/dataset/README.md`](vla-mender/src/run/dataset/README.md).

```bash
# Render the complete pre-repair prompt from YAML only.
PYTHONPATH=vla-mender/src python -m run.pre_repair.generate_prompt \
  --settings outputs/my_experiment/experiment.yaml
# The workflow CLI provides the same operation and writes a run manifest.
PYTHONPATH=vla-mender/src python -m workflow.pipeline prompt \
  --settings path/to/experiment.yaml --output outputs/my_experiment
# Run the VLA rollout and preserve all success/failure trajectories.
PYTHONPATH=vla-mender/src python -m workflow.pipeline rollout \
  --settings path/to/experiment.yaml --output outputs/my_experiment
PYTHONPATH=vla-mender/src python -m workflow.pipeline diagnose \
  --settings path/to/experiment.yaml --output outputs/my_experiment
# Give failure_diagnosis/prompt.md and agent_input.json to the agent.
PYTHONPATH=vla-mender/src python -m workflow.pipeline materialize \
  --settings path/to/experiment.yaml --output outputs/my_experiment \
  --diagnosis outputs/my_experiment/failure_diagnosis/diagnosis.json
```

The agent must return one diagnosis entry per failed episode containing phase,
causal frame, and an inclusive recoverable window. The infra then applies
configured stride/count inside that window, replays the source action prefix,
switches to the target controller, applies the selected reset dynamics, and
fails closed on any public-state divergence.

## OpenPI runtime backend and isolated environment

Rollout workers now use the `openpi` backend in
`vla-mender/src/workflow/openpi_backend.py`. OpenPI is imported lazily, so
prompt rendering and diagnosis do not require loading the model. The vendored
checkout is pinned by [`third_party/openpi.commit`](third_party/openpi.commit)
(current pin: `15a9616a00943ada6c20a0f158e3adb39df2ccac`). A changed checkout is
rejected before simulator/model initialization.

Build the dedicated environment using the commands from the pinned OpenPI
README (including the PyTorch `transformers` source patch):

```bash
VLA_MENDER_OPENPI_ENV=/absolute/path/to/.venv-openpi \
  ./scripts/bootstrap_openpi_env.sh
VLA_MENDER_OPENPI_ENV=/absolute/path/to/.venv-openpi \
  ./scripts/check_openpi_runtime.py \
  --checkpoint /absolute/path/to/openpi-checkpoint
```

Do not merge this environment with `.venv-libero2`: OpenPI's official Python
3.11 / Torch / JAX pins are intentionally isolated from the LIBERO control
runtime. The backend preflight requires `model.safetensors` and
`assets/physical-intelligence/libero/norm_stats.json` in the checkpoint (or an
explicit `backend.openpi_norm_stats` path).

The same resolved YAML is passed to both `rollout` and `diagnose`. Rollout
artifacts carry the `vla-mender.libero.openpi` trajectory protocol v2, the
settings fingerprint, and the backend commit/environment manifest. Diagnosis
refuses a mismatched fingerprint or protocol and exports only public
state/action observations and video references.
