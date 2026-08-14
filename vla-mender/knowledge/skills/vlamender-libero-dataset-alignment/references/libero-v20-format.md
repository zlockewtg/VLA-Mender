# Local LIBERO v2.0 dataset contract

Authoritative reference: `/mnt/public/tgy/datasets/libero`.

## Dataset-level requirements

- `codebase_version`: `v2.0`
- `robot_type`: `panda`
- `fps`: `10`
- `chunks_size`: `1000`
- `data_path`: `data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet`
- Images are embedded PNG bytes. A conforming output has no required external video files and sets
  `total_videos` to `0`.
- Copy the complete reference `tasks.jsonl` when retaining global task indices. LIBERO-Spatial local
  task 9 maps to global task index 39 and task text
  `pick up the black bowl on the wooden cabinet and place it on the plate`.
- `episodes.jsonl` contains one object per episode with keys in the order `episode_index`, `tasks`,
  `length`.
- `stats.json` contains `mean`, `std`, `max`, and `min` for every training column. Image statistics
  are normalized to `[0, 1]` and have shape `[3, 1, 1]`.

## Exact Parquet column order and Arrow types

| Column | Arrow type | Requirement |
|---|---|---|
| `image` | `struct<bytes: binary, path: string>` | PNG bytes; path `frame_NNNNNN.png` |
| `wrist_image` | `struct<bytes: binary, path: string>` | PNG bytes; path `frame_NNNNNN.png` |
| `state` | `fixed_size_list<float32>[8]` | Preserve source values exactly |
| `actions` | `fixed_size_list<float32>[7]` | Preserve dimensions 0–5; binarize dimension 6 |
| `timestamp` | `float32` | `frame_index / 10` within float32 tolerance |
| `frame_index` | `int64` | Start at zero and be contiguous in each episode |
| `episode_index` | `int64` | Equal the containing file's episode index |
| `index` | `int64` | Globally contiguous across the standalone dataset |
| `task_index` | `int64` | Global reference task index |

Copy the reference Arrow schema metadata as well as its field types. The required Hugging Face
metadata declares the two image structs as `Image`, `state/actions` as fixed-length `Sequence`, and
the remaining fields as scalar `Value` features.

Write Parquet with Snappy compression and row groups of at most 100 rows to match the reference.
Source-only fields such as `done`, `is_success`, and `intervene_flag` do not belong in the aligned
training schema; record their removal in provenance instead.

## Gripper action conversion

Apply the sign-preserving rule:

```text
source_gripper >= 0  -> +1.0
source_gripper < 0   -> -1.0
```

Do not clip or quantize action dimensions 0–5. Record the source gripper minimum, maximum, counts by
sign, zero policy, and output unique values in `meta/conversion_manifest.json`. The immutable source
dataset remains the recovery path for continuous gripper values.

## Required validation

For every episode:

1. Parquet row count equals `episodes.jsonl.length`.
2. Arrow field order, types, and schema metadata equal the reference.
3. Both image structs decode to `256x256x3`.
4. Stored PNG pixels exactly equal frames decoded from the source MP4 files.
5. State, timestamps, frame indices, episode indices, and global indices equal the source.
6. Motion actions 0–5 equal the source; gripper actions equal the sign rule.
7. `task_index` equals the selected global task.
8. No frame is missing or duplicated in either camera.

Dataset-level totals in `info.json` must equal the Parquet and episode metadata totals.
