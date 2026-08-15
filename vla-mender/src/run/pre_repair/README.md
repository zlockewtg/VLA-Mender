# Pre-repair run

本目录放置“单次实验”的配置和 prompt 入口；通用实现位于
`vla-mender/src/workflow`。Task 9、Task 0 或其他 LIBERO 任务都应通过复制
`experiment.example.yaml`、修改任务相关字段来构造自己的实验，不需要修改
`workflow` 源码。

## 1. 配置文件的含义

配置文件是一次实验的唯一契约。加载时会解析相对路径、校验取值，并生成
`settings_fingerprint`；实验过程中不要静默修改这些字段。

| 配置段 | 字段 | 含义 |
| --- | --- | --- |
| `task` | `suite` | LIBERO suite 名称，例如 `libero_goal`。 |
|  | `task_id` | suite 内的任务编号，从 0 开始。 |
|  | `checkpoint` | VLA/OpenPI checkpoint 目录；相对路径相对于 YAML 文件所在目录解析。 |
|  | `policy_config` | OpenPI 的 policy/training config 名称，例如 `pi0_libero`。 |
|  | `task_description` | 传给 policy 的任务指令；为 `null` 时使用 rollout 环境/任务提供的指令。 |
| `initial_states` | `provider` | 初始状态来源：`official`、`randomized_bddl` 或 `state_manifest`。 |
|  | `count` | 要 rollout 的随机场景/初始状态数量。 |
|  | `seed_start` | `randomized_bddl` 采样的起始场景 seed，随后按序尝试可复现的 seed。 |
|  | `state_manifest` | `provider=state_manifest` 时使用的 manifest；路径相对于 YAML 解析。 |
| `rollout` | `control_frequency_hz` | 仿真控制频率，例如实验契约中的 20 Hz。 |
|  | `max_steps` | 每个 episode 的最大 policy step 数；超时视为任务失败。 |
|  | `policy_seed` | rollout 的基础随机种子；episode index 会参与派生每条轨迹的 seed。 |
|  | `gpus` | 使用的 GPU 编号列表，例如 `[0, 1, 2, 3]`。 |
|  | `workers_per_gpu` | 每张 GPU 启动的 rollout worker 数。 |
|  | `action_chunk` | 一次 policy 推理后实际消费的连续 action 数。 |
|  | `inference_steps` | policy 每次推理的采样步数。 |
|  | `num_steps_wait` | reset 后、policy 执行前的稳定步数；official eval 使用 10。 |
|  | `binary_gripper` | 是否对实际下发的夹爪动作应用 `{-1,+1}` 滞回离散化。 |
|  | `gripper_hysteresis_threshold` | 二值夹爪的开/合阈值；official eval 使用 0.2。 |
| `controller` | `source_control_space` | rollout/replay 原始 action 空间：`osc` 或 `joint`。 |
|  | `target_control_space` | reset state 发布给后续 repair 的目标控制空间：`osc` 或 `joint`。 |
| `reset` | `frames_per_failure` | 每个失败 episode 最终保留的 reset candidate 数。 |
|  | `frame_stride` | 在已诊断的 recoverable window 内取帧的步长。 |
|  | `dynamics` | reset 动力学：`preserve_full_state` 保留完整状态，`quiescent_osc` 使用 Task 9 风格的 OSC 清零/静止处理。后者要求 `target_control_space=osc`。 |
| `backend` | `libero_root` | LIBERO 资源根目录，目录内应包含 `bddl_files/`、`init_files/` 和 `assets/`。设置后 runtime 在内存中覆盖 LIBERO 的路径解析，不再依赖额外的 `LIBERO_CONFIG_PATH/config.yaml`。 |

示例：

```yaml
task:
  suite: libero_goal
  task_id: 0
  checkpoint: /absolute/path/to/checkpoint
  policy_config: pi0_libero
  task_description: null

initial_states:
  provider: randomized_bddl
  count: 50
  seed_start: 100000
  state_manifest: null

rollout:
  control_frequency_hz: 20
  max_steps: 300
  policy_seed: 7
  gpus: [0, 1, 2, 3]
  workers_per_gpu: 1
  action_chunk: 5
  inference_steps: 5
  num_steps_wait: 10
  binary_gripper: true
  gripper_hysteresis_threshold: 0.2

controller:
  source_control_space: osc
  target_control_space: osc

reset:
  frames_per_failure: 3
  frame_stride: 5
  dynamics: preserve_full_state

backend:
  name: openpi
  openpi_environment: /opt/venv/openpi
  openpi_source: /absolute/path/to/third_party/openpi
  libero_root: /absolute/path/to/LIBERO-PRO/libero/libero
```

没有写入 YAML 的非核心实现细节继续使用 workflow 的默认行为；不要为了调
整日志、文件格式或内部容错而扩展实验参数。

## 2. 从 config 生成 prompt

模板文件是 [`prompt.md`](./prompt.md)。它包含完整的 pre-repair agent 契约，
包括 rollout、成功/失败轨迹保存、公开证据、失败阶段和窗口诊断、失败模式
聚类以及可选 reset bank 验证。模板中的 `{{...}}` 占位符只能由配置渲染，
不要在任务 prompt 中手工改写实验数字。

推荐把 YAML 放在对应 run 的根目录。核心参数全部写入 YAML 后，生成 prompt 时只需
提供这个 YAML 路径：

```bash
cd /mnt/public/tgy/VLA-Mender
export PYTHONPATH=vla-mender/src

mkdir -p /path/to/outputs/<run>
cp vla-mender/src/run/pre_repair/experiment.example.yaml \
  /path/to/outputs/<run>/experiment.yaml
# 编辑 experiment.yaml，填写任务、checkpoint、状态来源和 rollout/backend 参数。

python -m run.pre_repair.generate_prompt \
  --settings /path/to/outputs/<run>/experiment.yaml
```

例如，重新生成仓库中 `outputs/my_run` 的 prompt：

```bash
cd /mnt/public/tgy/VLA-Mender
unset LIBERO_CONFIG_PATH
PYTHONPATH=/mnt/public/tgy/VLA-Mender/vla-mender/src \
  /opt/venv/openpi/bin/python -m run.pre_repair.generate_prompt \
  --settings /mnt/public/tgy/VLA-Mender/outputs/my_run/experiment.yaml
```

该命令只会在同目录写入或覆盖 `prompt.generated.md`，不会生成 prompt
manifest。LIBERO 路径直接取自 YAML 的 `backend.libero_root`，不会创建
`libero_config/config.yaml`。

脚本执行顺序为：

1. `load_settings` 读取 YAML，并将 checkpoint/state manifest 的相对路径解析
   为绝对路径。
2. 校验 suite、状态数量、频率、GPU、控制空间、reset 参数及其组合约束。
3. 根据解析后的配置计算 fingerprint，作为实验身份的一部分。
4. 将 YAML 所在目录作为 run root，用解析后的字段替换 `prompt.md` 占位符；生成的
   命令会直接包含实际 YAML 绝对路径和正确的 run root。
5. 在同一目录写出 `<prompt 名称>.manifest.json`，记录 fingerprint、完整
   resolved settings 和 prompt 路径。

不指定 `--output` 时，默认在 YAML 同目录写出：

```text
<run>/
├── experiment.yaml
└── prompt.generated.md
```

因此不要直接修改或使用源码目录中的 `experiment.example.yaml` 运行正式实验，应先将
它复制到独立 run 目录。`--output` 仍可覆盖 prompt 文件位置，但 prompt 内的
`OUTPUT_DIR` 始终由 YAML 所在目录确定。

也可以使用统一 pipeline 的 `prompt` 子命令；它会由 `--output` 明确指定 run root，
并额外保存 resolved config：

```bash
python -m workflow.pipeline prompt \
  --settings /path/to/experiment.yaml \
  --output /path/to/outputs/<run>
```

结果包括：

```text
<run>/
├── experiment.resolved.yaml
├── prompt_manifest.json
└── failure_diagnosis/prompt.md
```

## 3. Pre-repair 实验流程

以下流程只覆盖 repair 之前的准备阶段，不执行 repair policy、不训练模型。

### 3.1 固化配置并生成任务 prompt

先准备任务 YAML，然后运行上面的 `prompt` 命令。后续所有阶段都使用同一份
YAML/fingerprint；如果已有同一 run 的 rollout，不要用新配置覆盖它。

### 3.2 VLA rollout

```bash
python -m workflow.pipeline rollout \
  --settings /path/to/experiment.yaml \
  --output /path/to/outputs/<run>
```

该阶段生成初始状态并执行 policy。每个 episode 都要保留，无论成功还是失败：

```text
<run>/rollout/
├── initial_states.npy
├── initial_state_manifest.json
├── summary.json
├── successful_episodes.json
├── failed_episodes.json
├── episodes/episode_<index>.json
└── videos/
    ├── episode_<index>_wide.mp4
    └── episode_<index>_wrist.mp4
```

episode JSON 保存 public state、实际执行 action、reward、逐步 success 标记、
场景 seed 和控制空间。成功轨迹不是可有可无的日志：它们会作为失败诊断时的
行为参考；失败轨迹则是后续窗口分析的唯一对象。

### 3.3 生成诊断输入

```bash
python -m workflow.pipeline diagnose \
  --settings /path/to/experiment.yaml \
  --output /path/to/outputs/<run>
```

该阶段从 rollout 生成：

```text
<run>/failure_diagnosis/
├── agent_input.json
└── prompt.md
```

`agent_input.json` 为 agent 可见的公开证据，包含所有失败 episode 和成功
episode 的参考信息、稀疏 state/action timeline 及 wide/wrist 视频路径；不应
暴露私有 simulator state、BDDL/XML、隐藏 predicate 或 reset payload。

### 3.4 Agent 诊断失败阶段、失败窗口和失败类别

agent 对每个失败 episode 按以下顺序处理，顺序不能颠倒：

1. 先判断失败所处的语义阶段，例如 approach、alignment、grasp/contact、
   manipulation、transport、release、recovery 或 timeout。
2. 在该失败轨迹中定位 `first_causal_frame_index`：行为首次离开可恢复路径
   的帧，而不是最明显的最终失败帧。
3. 定义包含 causal frame 的、仍可恢复的闭区间
   `[recoverable_window_start_frame_index, recoverable_window_stop_frame_index]`。
4. 在所有 episode 完成窗口判断后，按重复的因果机制聚类为少量任务内失败
   模式，分配本 run 内稳定的 `FM-01`、`FM-02` 等 ID。
5. 输出 `failure_diagnosis/diagnosis.json`。它必须覆盖每个任务失败且仅覆盖
   任务失败；每条记录必须有 phase、mode、category、causal frame、窗口、
   evidence 和 `[0,1]` 范围内的 confidence。

窗口只负责定义可恢复范围，agent 不直接挑选 reset frame。必须满足：

```text
0 <= window_start <= causal_frame <= window_stop < num_frames
```

### 3.5 生成并验证 reset bank（可选）

确认 `diagnosis.json` 通过 schema 和 episode/mode 覆盖校验后，运行：

```bash
python -m workflow.pipeline materialize \
  --settings /path/to/experiment.yaml \
  --output /path/to/outputs/<run> \
  --diagnosis /path/to/outputs/<run>/failure_diagnosis/diagnosis.json
```

协调器随后在每个诊断窗口内按升序执行：

```text
range(window_start, window_stop + 1, frame_stride)
```

取前 `frames_per_failure` 个 candidate。若窗口按 stride 后不足指定数量，流程
直接失败，不会偷偷放宽窗口或 stride。每个 candidate 会进行 action-prefix
replay，检查 public state 误差，应用 `reset.dynamics`，然后发布：

```text
<run>/failure_diagnosis/
├── reset_candidates.json
├── replay_verification.json
├── public_reset_bank.json
├── repair_jobs.json
└── private_reset_states/
```

只有 replay verification 通过的 candidate 才能进入 reset bank；该阶段仍然不
执行 repair。

## 4. 最小可执行顺序

```bash
cd /mnt/public/tgy/VLA-Mender
export PYTHONPATH=vla-mender/src

python -m workflow.pipeline prompt \
  --settings /path/to/experiment.yaml \
  --output /path/to/outputs/<run>

python -m workflow.pipeline rollout \
  --settings /path/to/experiment.yaml \
  --output /path/to/outputs/<run>

python -m workflow.pipeline diagnose \
  --settings /path/to/experiment.yaml \
  --output /path/to/outputs/<run>

# agent 写入 diagnosis.json 后再执行：
python -m workflow.pipeline materialize \
  --settings /path/to/experiment.yaml \
  --output /path/to/outputs/<run> \
  --diagnosis /path/to/outputs/<run>/failure_diagnosis/diagnosis.json
```
