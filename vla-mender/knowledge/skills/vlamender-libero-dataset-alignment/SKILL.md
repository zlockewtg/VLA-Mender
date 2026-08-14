---
name: vlamender-libero-dataset-alignment
description: Convert and validate VLA-Mender or LeRobot video-backed rollout datasets against the local LIBERO v2.0 embedded-image Parquet contract. Use when unifying MP4-backed image loading, Arrow schemas, task indices, metadata, or continuous gripper actions before training with /mnt/public/tgy/datasets/libero.
---

# VLA-Mender LIBERO Dataset Alignment

Read [references/libero-v20-format.md](references/libero-v20-format.md) before changing a dataset.
Treat the reference dataset's Arrow schema and Hugging Face schema metadata as authoritative.

## Convert

Keep the source immutable and choose a new output directory. Run:

```bash
python scripts/align_libero_dataset.py convert \
  --source /path/to/video_backed_source \
  --reference /mnt/public/tgy/datasets/libero \
  --output /path/to/new_embedded_png_dataset \
  --task-index 39
```

The converter must:

1. Decode every `image` and `wrist_image` MP4 frame in episode order.
2. Store each decoded frame losslessly as PNG bytes in the Parquet image structs.
3. Cast `state` and `actions` to the reference fixed-size list types.
4. Drop source-only columns from the training schema.
5. Preserve the first six action dimensions and binarize gripper action by sign.
6. Use the reference task catalog and a caller-supplied global task index.
7. Recompute `stats.json`; never copy source or reference statistics.
8. Write through a staging directory and leave the source untouched.

For LIBERO-Spatial local task 9, use global task index 39.

## Validate

Run source-equivalence validation after conversion:

```bash
python scripts/align_libero_dataset.py validate \
  --dataset /path/to/new_embedded_png_dataset \
  --reference /mnt/public/tgy/datasets/libero \
  --source /path/to/video_backed_source \
  --task-index 39 \
  --report /path/to/new_embedded_png_dataset/meta/validation_report.json
```

Require exact Arrow schema equality, exact numeric preservation outside the gripper dimension,
lossless equality between decoded source video frames and stored PNG frames, contiguous row indices,
10 Hz timestamps, valid task metadata, and gripper values restricted to `{-1, +1}`.

Do not call a conversion aligned when it only changes `info.json`, leaves images in MP4, retains
variable-length Arrow lists, or silently clips the six motion-action dimensions.
