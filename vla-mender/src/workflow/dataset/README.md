# Prefix + repair dataset construction

This package builds LeRobot v2 datasets by joining a VLA prefix with a
successful repair suffix. It is the generalized successor of the Goal-0
`build_prefix_plus_repair45_dataset.py` workflow.

The builder is fail-closed: an output is published only after source lineage,
compiled simulator identity, exact reset state, visual continuity, schema, and
training masks pass validation. Incomplete builds stay in a hidden staging
directory and are removed on failure.

## Package layout

```text
workflow/dataset/
  config.py             strict YAML configuration
  manifest.py           task-agnostic episode input contract
  continuity.py         model/state hashes and optical-flow admission
  media.py              embedded PNG and native-video adapters
  demo_videos.py        deterministic complete-trajectory demo rendering
  builder.py            atomic LeRobot dataset builder
  cli.py                generic build CLI
  libero_manifest.py    adapter for VLA-Mender LIBERO repair batches
  libero_observation_manifest.py  adapter for observation-only repair batches
  research_manifest.py  adapter for retained standalone-repair evidence
  run.py                high-level YAML preparation and atomic build orchestration
  visualize_splices.py  annotated full and boundary-clip splice videos
```

The core builder does not know a LIBERO suite, task number, worker layout,
camera name, state width, action width, or selection strategy. Those decisions
are expressed by two files:

1. A YAML build configuration describes the output schema mapping and policies.
2. A JSON episode manifest provides already-selected, ordered source episodes.

Selection is intentionally outside the builder. A Base-FM selector, human
curation, or another ranking method can all produce the same episode contract.

## Installation and invocation

From the repository root:

```bash
PYTHONPATH=vla-mender/src python -m workflow.dataset.cli \
  --config path/to/build.yaml --validate-config-only

PYTHONPATH=vla-mender/src python -m workflow.dataset.cli \
  --config path/to/build.yaml
```

For the end-to-end standalone-repair path, including retained-manifest
materialization and a resolved post-training config, use the higher-level run
entrypoint documented in `run/dataset/README.md`:

```bash
PYTHONPATH=vla-mender/src python -m run.dataset.build \
  --settings path/to/dataset.yaml
```

To inspect all spliced episodes after a build:

```bash
PYTHONPATH=vla-mender/src python -m workflow.dataset.visualize_splices \
  --dataset /path/to/built-dataset
```

This writes side-by-side agent/wrist full videos, five-second boundary clips,
and a hashed index under `meta/visualization/splice_videos/`.

The output path must not exist. This prevents a partially changed rerun from
silently overwriting an immutable training artifact.

Every build renders three complete annotated trajectory demos by default under
`meta/visualization/trajectory_demos/`. Episodes are selected deterministically
and evenly over final episode order; datasets with fewer than three episodes
render all of them. Set `demo_videos.enabled: false` or change
`demo_videos.count` in the build YAML when needed.

## YAML contract

See [`examples/prefix_repair.example.yaml`](examples/prefix_repair.example.yaml).
Important settings are:

- `reference_dataset`: supplies the exact Arrow schema, `info.json`, and task
  catalog expected by the training loader.
- `episodes_manifest`: selected episodes in final output order.
- `cameras`: maps output image columns to prefix columns, native dimensions,
  and repair-only image transforms.
- `action`: state/action widths and optional gripper binarization.
- `continuity.signature_fields`: compiled simulator properties which must match.
- `pre_guard_frames`, `post_guard_frames`, and `action_horizon`: training-mask
  policy around the controller transition.
- `allow_splice_crossing_action_chunks`: defaults to `false`; set it to `true`
  only for native continuous trajectories whose prefix and repair boundary is
  intentionally trainable as one action-chunk segment.

Relative paths in YAML resolve relative to the YAML file. Relative paths in the
episode manifest resolve relative to the manifest.

## Episode manifest contract

```json
{
  "schema_version": 1,
  "episodes": [
    {
      "source_episode_id": 47,
      "restart_frame": 53,
      "task_index": 19,
      "task": "open the middle drawer of the cabinet",
      "prefix": {
        "parquet": "/data/selected-prefix/episode_000000.parquet",
        "images": {
          "image": {
            "column": "image",
            "continuity_video": "/data/original-vla/image/episode_000009.mp4"
          }
        }
      },
      "repair": {
        "parquet": "/data/repair/episode_000000.parquet",
        "images": {
          "image": {"video": "/data/repair/agentview/episode_000000.mp4"}
        }
      },
      "continuity": {
        "reset_descriptor": "/private/reset_descriptor.json",
        "attempt_manifest": "/repair/attempt_manifest.json",
        "result": "/repair/repair_result.json"
      },
      "metadata": {"failure_mode_id": "FM-01"}
    }
  ]
}
```

Each image source uses exactly one payload source:

- `column`: embedded PNG payload from the segment parquet; or
- `video`: native RGB video decoded and embedded as PNG.

`continuity_video` is optional and deliberately separate. It supplies the
authoritative same-state image without changing the output payload. This is
needed when a selected prefix must remain byte-for-byte inherited from an
earlier dataset while continuity must be checked against the original rollout.

A standalone repair fragment uses explicit `repair_only` mode and omits both
`restart_frame` and `prefix`:

```json
{
  "source_episode_id": "47:window_midpoint",
  "mode": "repair_only",
  "task_index": 19,
  "task": "open the middle drawer of the cabinet",
  "repair": {
    "parquet": "/data/repair/episode_000000.parquet",
    "images": {"image": {"video": "/data/repair/agentview/episode_000000.mp4"}}
  },
  "continuity": {
    "reset_descriptor": "/private/reset_descriptor.json",
    "attempt_manifest": "/repair/attempt_manifest.json",
    "result": "/repair/repair_result.json"
  }
}
```

Repair-only episodes still require exact simulator admission evidence and a
successful terminal transition. They do not run prefix-state or splice-flow
checks because no prefix or controller transition is present.

## Admission checks

For every episode the default policy requires:

1. Repair batch metadata reports a strict simulator signature check.
2. Reset and repair signatures match on dimensions, names, geom layout, and
   `model_numeric_sha256`.
3. The reset state and repair recorder row 0 have identical canonical float64
   hashes and maximum absolute error no larger than `1e-12`.
4. Prefix state at `restart_frame` matches repair state row 0.
5. Every configured camera compares the same logical state: original prefix
   `restart_frame` versus transformed repair row 0. Median and p90 optical flow
   must stay below configured thresholds.
6. Repair action-valid rows form a contiguous prefix and the last valid action
   is a successful terminal transition when `require_terminal_success` is true.

The compiled model hash includes randomized static values such as
`model.body_pos`. These values are not represented in MuJoCo's flattened
qpos/qvel state; checking only a 79-D state vector cannot detect a moved fixture
or background object.

## Training semantics

An output episode is:

```text
prefix rows [0, restart_frame) + repair valid-action rows [0, N)
```

or, in explicit `repair_only` mode:

```text
repair valid-action rows [0, N)
```

The restart observation is not duplicated. Prefix and repair are distinct
continuous segments. Rows in the final prefix guard, initial repair guard, and
terminal row are non-trainable. A valid action-chunk start must remain entirely
inside one trainable segment, so chunks cannot cross the splice.
Repair-only rows form one continuous segment, receive no intervention guard,
and only mask the terminal row. Prefix-plus-repair rows normally remain two
continuous segments. With zero guards and
`allow_splice_crossing_action_chunks: true`, they instead form one native
episode segment and action chunks may cross the splice.

The full per-row decision is recorded in
`meta/trainable_index_manifest.json`; consumers should use this artifact rather
than recreating masking rules from episode lengths.

## Rebuild the current Goal-0 repair30 input manifest

The LIBERO adapter joins the frozen Base-FM selection order, the repaired batch,
failure windows, original rollout videos, and inherited prefix dataset:

```bash
REPO=/mnt/public/tgy/VLA-Mender
EVAL_ROOT=/mnt/public/tgy/data/libero_eval/libero_goal_task000_randomized50_20260810T052115Z

PYTHONPATH="$REPO/vla-mender/src" python -m workflow.dataset.libero_manifest \
  --eval-root "$EVAL_ROOT" \
  --selection-manifest /mnt/public/tgy/datasets/pi0_libero_goal_task0_randomized50_vla_prefix_plus_osc_repair_basefm_lowest30_guardpre20_post5_boundarymasked_20hz_20260811_native256_policyv18_v1/meta/build_manifest.json \
  --repair-root "$EVAL_ROOT/failure_diagnosis/repair_observation_only/batch_numeric_strict79_repair30_v4" \
  --prefix-dataset /mnt/public/tgy/datasets/pi0_libero_goal_task0_randomized50_vla_prefix_handle_alignment_v4_osc_repair30_basefm_lowest30_native256_fliprepair_guardpre20_post5_boundarymasked_prefixruntime_strict79_20hz_20260812_v2 \
  --prefix-camera 'image=videos/chunk-000/image/episode_{episode_index:06d}.mp4' \
  --prefix-camera 'wrist_image=videos/chunk-000/wrist_image/episode_{episode_index:06d}.mp4' \
  --repair-camera 'image=videos/chunk-000/observation.images.agentview/episode_000000.mp4' \
  --repair-camera 'wrist_image=videos/chunk-000/observation.images.wrist/episode_000000.mp4' \
  --output /tmp/goal0-repair30-episodes.json
```

Copy the example YAML, set `episodes_manifest` to that JSON, choose a new output
path, then invoke `workflow.dataset.cli`.

## Trust boundary

Simulator state and reset descriptors are coordinator-private artifacts. They
are used only for dataset admission and provenance; they must not be exposed to
the repair-policy synthesis actor. The builder never infers a successful repair
from images alone and never relaxes missing evidence into a warning. For legacy
data exploration only, `continuity.require_simulator_evidence: false` disables
private-state admission, but such an output should not be treated as strictly
aligned training data.
