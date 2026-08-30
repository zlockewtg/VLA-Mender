# Pre-repair run

本目录放置单任务实验和顺序多任务 campaign 的配置及 prompt 入口；通用实现位于
`vla-mender/src/workflow`。单任务复制 `experiment.example.yaml`，多个 LIBERO
任务按顺序执行时复制 `campaign.example.yaml`，不需要修改 `workflow` 源码。

## 1. 配置文件的含义

配置文件是一次实验的唯一契约。加载时会解析相对路径、校验取值，并生成
`settings_fingerprint`；实验过程中不要静默修改这些字段。

| 配置段 | 字段 | 含义 |
| --- | --- | --- |
| 顶层 | `tasks` | 可选的有序任务列表；存在时进入顺序 campaign 模式。 |
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
| `reset` | `candidate_selection` | reset 介入点选择：`per_episode_stage_entry_only` 对每条轨迹独立抽帧并定位行为阶段入口；`failed_stage_entry_only` 取预先声明的宽粒度阶段入口；`pre_causal_only` 取异常首帧之前的最后正常状态；`window_start_only` 取失败子阶段起点；v5 `pre_window_and_endpoints` 取窗口外预防点及两个端点；历史 v4 `window_endpoints` 取两个端点。 |
|  | `prevention_steps` | 预防点相对 `window_start` 提前的 policy step 数；v5 固定为 10，不能截断或平移。 |
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
  candidate_selection: pre_window_and_endpoints
  prevention_steps: 10
  dynamics: preserve_full_state

backend:
  name: openpi
  openpi_environment: /opt/venv/openpi
  openpi_source: /absolute/path/to/third_party/openpi
  libero_root: /absolute/path/to/LIBERO-PRO/libero/libero
```

多任务配置保留上面的公共参数，只把有序任务写入 `tasks:`。此时 `task:` 中的
checkpoint、policy config 和 description 是共享默认值，各任务项可以覆盖这些字段，
也可以覆盖自己的 `initial_states`、`rollout`、`controller`、`reset` 或 `backend`：

```yaml
task:
  checkpoint: /absolute/path/to/checkpoint
  policy_config: pi0_libero
  task_description: null

tasks:
  - key: object-0
    suite: libero_object
    task_id: 0
  - key: spatial-9
    suite: libero_spatial
    task_id: 9
    rollout:
      max_steps: 280
  - key: goal-0
    suite: libero_goal
    task_id: 0
```

列表顺序就是执行顺序。每个 `key` 必须唯一，并用于构造隔离的任务目录；省略时
自动生成 `<suite>-task<三位 task_id>`。campaign 采用 fail-fast 语义：只有当前
任务的 `repair_handoff/manifest.json` 完整且 replay 验证通过，才进入下一个任务；
基础设施失败不会被跳过或计作 policy failure。

没有写入 YAML 的非核心实现细节继续使用 workflow 的默认行为；不要为了调
整日志、文件格式或内部容错而扩展实验参数。

## 2. 从 config 生成 prompt

模板文件是 [`prompt.md`](./prompt.md)。它包含完整的 pre-repair agent 契约，
包括 rollout、成功/失败轨迹保存、公开证据、失败阶段和窗口诊断、失败模式
聚类以及必需的 reset bank 验证和 repair handoff。模板中的 `{{...}}` 占位符只能由配置渲染，
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

多任务时把 `--settings` 指向 `campaign.yaml` 即可，命令不变。它会额外生成：

```text
<campaign>/
├── campaign.yaml
├── campaign_manifest.json
├── prompt.generated.md
└── tasks/
    ├── 000_<task-key>/
    │   ├── experiment.resolved.yaml
    │   └── failure_diagnosis/prompt.md
    └── 001_<task-key>/
        ├── experiment.resolved.yaml
        └── failure_diagnosis/prompt.md
```

总 prompt 是顺序协调器；各子目录里的 prompt 仍是完整、独立的单任务契约。

例如，重新生成仓库中 `outputs/my_run` 的 prompt：

```bash
cd /mnt/public/tgy/VLA-Mender
unset LIBERO_CONFIG_PATH
PYTHONPATH=/mnt/public/tgy/VLA-Mender/vla-mender/src \
  /opt/venv/openpi/bin/python -m run.pre_repair.generate_prompt \
  --settings /mnt/public/tgy/VLA-Mender/outputs/my_run/experiment.yaml
```

单任务时，该命令只会在同目录写入或覆盖 `prompt.generated.md`，不会生成 prompt
manifest。多任务时则写入前述 `campaign_manifest.json` 和各任务的 resolved
contract。LIBERO 路径直接取自 YAML 的 `backend.libero_root`，不会创建
`libero_config/config.yaml`。

脚本执行顺序为：

1. `load_experiment_plan` 读取 YAML，并将每个任务的 checkpoint/state manifest
   相对路径解析为绝对路径。
2. 校验 suite、状态数量、频率、GPU、控制空间、reset 参数及其组合约束。
3. 每个任务根据自己的 resolved config 计算独立 fingerprint；campaign 另计算有序
   任务列表 fingerprint。
4. 单任务将 YAML 所在目录作为 run root，用解析后的字段替换 `prompt.md` 占位符。
5. 多任务为每项写出独立 resolved settings 和任务 prompt，再写总协调 prompt 与
   `campaign_manifest.json`。

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

如果输入包含顶层 `tasks:`，同一命令改为生成 `campaign_prompt.md`、
`campaign_manifest.json` 和上述 `tasks/` 隔离目录。`rollout`、`diagnose`、
`materialize` 仍是单任务原子阶段，campaign prompt 会按顺序把它们作用于每个任务
自己的 `experiment.resolved.yaml` 和 run root。

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
2. 在该失败轨迹中定位 `first_causal_frame_index`：失败子任务阶段内，轨迹首次
   明确出现错误或相对同阶段成功轨迹产生偏移的帧；它是失败窗口末帧。不能等待
   更晚、更清晰的碰撞、回弹或 timeout 才截断窗口。
3. 将 `recoverable_window_start_frame_index` 定义为**同一个失败子任务阶段内**的
   最早可纠正帧：该阶段的前置条件已经完成、该阶段已经开始，但错误偏移尚未发生。
   不得为了扩大窗口而退回上一阶段。例如 pick 成功而 transport 失败时，首帧必须
   是 pick 完成后的最早可纠正 transport 帧，不能选 pick/grasp 阶段；placement
   失败时同理，首帧必须已经进入 placement 阶段。
4. 定义闭区间 `[recoverable_window_start_frame_index,
   recoverable_window_stop_frame_index]`，并强制 `window_stop == causal_frame`。
5. 在所有 episode 完成窗口判断后，按重复的因果机制聚类为少量任务内失败
   模式，分配本 run 内稳定的 `FM-01`、`FM-02` 等 ID。
6. 输出 `failure_diagnosis/diagnosis.json`。它必须覆盖每个任务失败且仅覆盖
   任务失败；每条记录必须有 phase、mode、category、causal frame、窗口、
   evidence 和 `[0,1]` 范围内的 confidence。

窗口的语义阶段决定边界，而不是固定帧数。必须满足：

```text
0 <= window_start < window_stop == causal_frame < num_frames
```

### 3.5 生成并验证 reset bank（必需）

确认 `diagnosis.json` 通过 schema 和 episode/mode 覆盖校验后，运行：

```bash
python -m workflow.pipeline materialize \
  --settings /path/to/experiment.yaml \
  --output /path/to/outputs/<run> \
  --diagnosis /path/to/outputs/<run>/failure_diagnosis/diagnosis.json
```

`diagnosis.json` 只是 materialize 前的工作文件，不能作为 pre-repair 的终点。
候选点由 `reset.candidate_selection` 决定。`per_episode_stage_entry_only` 禁止为阶段
指定固定帧或复用其他任务的阶段词表。它先从任务指令和多条成功轨迹归纳可观察
阶段图，再逐轨迹检查所有相机邻帧和 public state/action change-point。每个入口需
记录前置阶段、多相机证据、可观察转折、状态/动作转折、持续性、成功参考，以及
为何前一帧过早、后一候选过晚。运动方向只是证据之一；接触关系、夹爪模式、
当前操作实体和任务相关对象状态同样可以定义阶段边界。
`failed_stage_entry_only` 要求每条失败
记录 `intervention_stage` 和 `intervention_stage_start_frame_index`，并且只装填该
宽粒度行为阶段的入口；例如 pick 内发生抓取错误时回到 pick 入口，持物移动或
篮筐接近发生错误时回到 transport 入口。`pre_causal_only` 要求每条失败显式引用
成功轨迹，按可观察阶段对齐确定第一帧异常，并且只装填
`pre_causal = first_causal_frame_index - 1`，即异常出现前的最后正常状态。
`window_start_only` 为每条失败只选择
失败子阶段内最早可纠正的 `window_start`；不装填 `pre_window`、`window_stop` 或
内部帧。`pre_window_and_endpoints` 则选择窗口外的
`pre_window = window_start - prevention_steps`、`window_start` 和
`window_stop`。窗口本身始终严格遵守同一失败语义阶段的诊断要求。所有被选中的
candidate 都会进行 action-prefix replay，检查 public state 误差，应用
`reset.dynamics`，然后原子发布：

```text
<run>/repair_handoff/
├── manifest.json
├── private_reset_states/
└── agent_views/
```

`manifest.json` 是 repair 唯一读取的 pre-repair JSON，将 validated diagnosis、
candidate selection、repair jobs、public reset metadata 和 replay verification
合并在一起。只有所有 candidate 均通过 replay verification、附件存在且哈希匹配时，
`complete=true` 和 `all_replays_verified=true` 才表示 pre-repair 完成；该阶段仍然不
执行 repair。完整字段契约和完成检查表以生成的 prompt 为准。

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
