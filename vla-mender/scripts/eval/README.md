# 通用 LIBERO checkpoint evaluation

`run_eval.sh` 是 standalone eval 的统一入口。它按顺序执行 YAML 中的任务，单个
任务内部再把连续、均衡的 initial-state 分片并行分配给 GPU worker。

核心评测实现位于 `src/workflow/rollout/`：`state_provider.py` 统一状态加载、manifest
校验和随机场景生成，`runner.py` 统一 states × trials、seed 和 batch 执行，
`evaluator.py` 负责单 episode 的 reset、推理与 action dispatch。standalone eval 和
pre-repair rollout 都调用这些 API，因此 scene/policy seed、action chunk、action
clip、binary gripper 和 LIBERO native `done` 语义一致；pre-repair 的配置与目录接口
没有变化。

本目录只保留命令行适配层：`eval.py` 负责 YAML/CLI、GPU 调度、契约和跨任务汇总；
`_runtime.py` 是内部 GPU 子进程入口，只负责把 worker 参数转换为共享核心请求，并
适配 LeRobot/video、resume 和随机场景生成命令。它作为进程边界保留，以隔离 CUDA、
MuJoCo、模型内存和 worker 日志。旧 shard merger 已由 campaign 汇总取代；reset
materializer 和 observation-only repair runner 位于 `scripts/repair/`。

## 快速开始

```bash
cd /mnt/public/tgy/VLA-Mender

vla-mender/scripts/eval/run_eval.sh default \
  --task libero_goal:0 \
  --task libero_spatial:9 \
  --state-provider official \
  --gpus 0,1,2,3 \
  --num-envs 4 \
  --checkpoint /path/to/checkpoint \
  --output /mnt/public/tgy/data/my_eval
```

`--task suite:id` 可以重复；只要 CLI 出现一个 `--task`，就会整体替换 YAML 的
`tasks`。单任务也可以写成 `--suite libero_goal --task-id 0`。最终优先级是
`CLI > tasks[] 任务级覆盖 > YAML 全局值`，YAML 中的相对路径均相对于 YAML 文件
所在目录解析。

先检查完整的命令、分片、GPU 映射和契约，同时不创建目录、不生成场景、不加载模型：

```bash
vla-mender/scripts/eval/run_eval.sh default \
  --task libero_goal:0 --state-provider official \
  --checkpoint /path/to/checkpoint --dry-run
```

## Initial-state 来源

每个 profile 必须显式选择 `initial_states.provider`：

- `official`：使用 LIBERO suite 自带的 state，支持 `count` 或显式 `indices`。
- `randomized_bddl`：按 BDDL 原生 sampler 生成并验证确定性场景；只支持 `count`。
  缓存键包含任务、数量、seed、控制频率和全部验证参数。缓存不完整、契约变化或
  hash 校验失败会直接报错，不会覆盖已有缓存。
- `manifest`：读取外部 schema-v1 manifest。worker 会校验 suite/task、任务文本、
  控制频率、数组 shape、整数组 hash、逐 state hash、连续索引和 validation report。

`count` 与 `indices` 互斥。CLI 可分别用 `--state-count`、
`--initial-state-indices 0,2,4`、`--state-manifest` 覆盖。

## Resume、overwrite 与产物

resume 必须指定相同的 `--output`：

```bash
vla-mender/scripts/eval/run_eval.sh default \
  --output /mnt/public/tgy/data/my_eval --resume
```

保存的 campaign/task/worker 契约必须一致；允许修改 `--gpus` 的物理 GPU ID，但
worker 数和分片不能改变。worker 只接受 results 与 LeRobot metadata 一致、且已完成
episode 构成预期前缀的安全断点。`videos_only` 不支持 resume，`--resume` 与
`--overwrite` 互斥。

输出形状对单任务和多任务完全一致：

```text
<run>/
├── eval_contract.json
├── summary.json
└── tasks/<task-key>/
    ├── task_contract.json
    ├── summary.json
    ├── scene_generation.log
    └── workers/worker_XX/
        ├── eval.log
        └── dataset/
```

任务失败后会继续执行后续任务。顶层汇总会检查 task key、initial-state index 和
trial index 的覆盖与重复项；任一任务/worker 失败时命令最终返回非零。


## 实验结果索引

`/mnt/public/tgy/data/experiment_results.json` 汇总 `data` 下文件名包含
result、summary、report 或 metric 的全部 JSON 结果，并为每个 checkpoint 关联训练
数据集、训练参数和配置来源。逐 episode 大数组不会复制到索引中；索引保留原文件
绝对路径和 SHA-256，可按需回查完整结果。`experiments` 按实验目录的总文件大小降序
排列；同一实验的 state、GPU 或 worker shard 合并计数，不作为独立实验。

新增实验后可重新生成：

```bash
cd /mnt/public/tgy/capx-aspire/aspire/vla_mender/openpi
.venv/bin/python \
  /mnt/public/tgy/VLA-Mender/vla-mender/scripts/eval/build_experiment_results_index.py
```

训练元数据按本地 W&B 历史快照、不可变训练契约、当前 OpenPI 配置注册表、路径参数
提示的顺序解析；每个 checkpoint 的 `training_metadata_match_method` 会注明实际来源。
