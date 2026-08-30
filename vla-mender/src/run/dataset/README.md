# Dataset build and post-training

这个入口把已完成的 pre-repair、repair 质量筛选、LeRobot 数据集构建和 OpenPI
后训练串成同一个 YAML 契约。通用实现位于 `workflow.dataset` 与
`workflow.training`，每个实验只需要复制和修改 `dataset.example.yaml`。

当前 repair 产物可直接使用：

```bash
cd /mnt/public/tgy/VLA-Mender
export PYTHONPATH=/mnt/public/tgy/VLA-Mender/vla-mender/src
export VLA_MENDER_PYTHON=/mnt/public/tgy/VLA-Mender/.venv-libero2/bin/python
export DATASET_SETTINGS=/mnt/public/tgy/VLA-Mender/outputs/stove_pan_scene3_repair_v2_contact_path_review/dataset.yaml

# 只校验 YAML，不写文件。
$VLA_MENDER_PYTHON -m run.dataset.build \
  --settings "$DATASET_SETTINGS" --validate-config-only

# 生成 retained episode manifest、task catalog 以及 resolved 构建/训练配置。
$VLA_MENDER_PYTHON -m run.dataset.build \
  --settings "$DATASET_SETTINGS" --prepare-only

# 完整构建并执行 simulator hash、splice state、双相机光流和输出 schema 校验。
$VLA_MENDER_PYTHON -m run.dataset.build --settings "$DATASET_SETTINGS"

# 数据集通过后，只做 OpenPI 环境、commit、checkpoint、norm stats、batch 和
# trainable action-chunk 的训练 preflight。
$VLA_MENDER_PYTHON -m run.dataset.train \
  --settings "$DATASET_SETTINGS" --dry-run

# preflight 无误后启动 torchrun。GPU 和超参数全部来自 YAML。
$VLA_MENDER_PYTHON -m run.dataset.train --settings "$DATASET_SETTINGS"
```

数据集发布后若只想做后训练超参数对照，不需要重建数据。复制
`dataset_quality40_v1/training.resolved.yaml` 到一个新的 YAML，保留其中的 `dataset`
和 `trainable_index_manifest`，修改 `experiment_name`、学习率或 step 数，然后把该文件
直接传给 `run.dataset.train`。每个对照必须使用新的 `experiment_name`。

训练配置默认使用 `sampling_mode: transition_aware`。若要复现 OpenPI native LeRobot
采样，设置 `sampling_mode: native`；此时 `trainable_index_manifest` 可省略，所有 dataset
row 都是样本起点，action chunk 可以跨 VLA/repair 拼接边界，短 episode 尾部使用
LeRobot 标准 padding。native 模式不需要重新构建 parquet。

当前 builder 有意与 `/mnt/public/tgy/datasets/libero` 保持 LeRobot v2 schema。固定版本的
LeRobot loader 会打印 v2.0 向 v2.1 迁移提示，但仍是向后兼容读取；不要对已发布数据集
做原地转换。需要升级格式时应产生一个新的、重新验证的 dataset 版本。

`source.adapter: research_quality_selection` 读取 retained-only manifest；每个 episode
拼接 `[0, reset_frame)` 的 VLA prefix 与选中 repair trajectory。它会验证 repair job、
handoff reset、result identity 和公开证据路径，不能把未保留或未成功样本混入数据集。

构建输出遵循 LeRobot v2 schema，并额外写入
`meta/trainable_index_manifest.json`。pre/post intervention guard、terminal row，以及
会跨越 VLA/repair 连续动力学边界的 action chunk 都不会进入训练采样。训练入口在每次
启动时重新计算允许的全局索引，因此不会只信任 manifest 中的计数。

每次新 dataset build 默认还会按最终 episode 顺序均匀选取 3 条完整轨迹（首条、
中间、末条），将所有相机横向拼接并标注 VLA/repair phase 与 splice frame，写到
`meta/visualization/trajectory_demos/`。不足 3 条时保存全部。可通过
`build.demo_videos.enabled` 和 `build.demo_videos.count` 调整；demo 生成属于原子构建
门禁，失败时不会发布 dataset。

重要产物：

- `<run.output_dir>/episodes_manifest.json`：从质量筛选物化的 40 条 episode 源清单。
- `<run.output_dir>/dataset.resolved.yaml`：绝对路径化的底层 builder 配置。
- `<run.output_dir>/training.resolved.yaml`：注入最终 dataset 与 trainable manifest 的训练配置。
- `<dataset>/meta/validation_report.json`：训练启动的硬门禁。
- `<dataset>/meta/visualization/trajectory_demos/`：默认 3 条完整拼接轨迹 demo 和哈希清单。
- `<checkpoint_base_dir>/<run_name>/<experiment_name>/`：OpenPI checkpoint。

已生成的 run artifact 采用不可变写入；如果改变输入或超参数，应使用新的
`run.output_dir`、dataset output 和 `experiment_name`，不要覆盖旧实验。`resume: true`
只允许从已有数字 step checkpoint 继续；新训练默认拒绝复用已存在的 checkpoint 目录。
