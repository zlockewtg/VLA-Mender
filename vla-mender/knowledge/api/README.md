本目录中的 API 与配套工具代码引用并移植自 [Cap-X](https://github.com/capgym/cap-x.git)。

# Code-policy API

这里保存 VLA-Mender code policy 使用的机器人控制、感知和运动规划 API。代码来源于
Cap-X 的 `capx/integrations`；其直接依赖的通用工具从 `capx/utils` 移入了本目录的
`utils/`。迁移时，原来的 `capx.integrations.*` 和 `capx.utils.*` 内部引用已分别改为
`knowledge.api.*` 和 `knowledge.api.utils.*`。

## 核心接口

- Repair agent 不依赖静态 API manifest。任务开始时应直接读取
  `workflow/research/libero_backend.py` 中的 `_api_for`，再检查所选 API 类的
  `functions()` 映射、签名、docstring 和实际函数实现。只把这些结果列入临时
  Markdown 清单；不写入 campaign/candidate artifact，也不把 privileged 实现当作
  可调用接口。
- `base_api.py`：定义 `ApiBase`、API 注册表以及 `register_api()`、`get_api()` 和
  `list_apis()`。每个 API 通过 `functions()` 决定向 code policy 暴露哪些函数，
  `combined_doc()` 则生成可加入模型提示词的函数签名和说明。
- `gym_action.py`：提供通用 Gym/Gymnasium 环境的观察和低维动作执行接口。
- `__init__.py`：按名称注册可用 API；缺少可选机器人或仿真依赖时，对应 API 会被跳过。

## Franka API

- `franka/control.py`：完整的 Franka 感知、抓取、逆运动学和位姿控制接口。
- `franka/control_privileged.py`：允许读取仿真器真值状态的 privileged 控制接口。
- `franka/control_reduced.py`：面向视觉观察的精简接口，组合 Molmo、SAM、GraspNet、
  PyRoki 与机械臂控制。
- `franka/control_reduced_exampleless.py`：去掉文档示例的 reduced 版本，用于控制
  code-policy prompt 中的先验信息量。
- `franka/control_reduced_skill_library.py`：在 reduced API 上增加常用几何、抓取和
  操作技能。
- `franka/libero.py`：面向 LIBERO 任务的完整 Franka API。
- `franka/libero_reduced.py`：面向 LIBERO 视觉观察的精简感知、抓取与关节/位姿控制接口。
- `franka/libero_reduced_skill_library.py`：在 LIBERO reduced API 上增加机器人状态估计、
  抓取状态判断以及可复用操作技能。
- `franka/libero_osc_reduced_skill_library.py`：使用原生 operational-space control
  （OSC）的 LIBERO 技能库，提供 `osc_step()`、控制器规格查询和 OSC 位姿运动。
- `franka/handover*.py`：双臂物体交接 API，包含完整、privileged、reduced 和
  exampleless 变体。
- `franka/two_arm_lift*.py`：双臂抬起任务 API，包含视觉版和 privileged 版。
- `franka/spill_wipe*.py`：液体擦拭任务 API，包含视觉版和 privileged 版。
- `franka/nut_assembly*.py`：螺母/销装配任务 API，包含视觉版和 privileged 版。
- `franka/common.py`：上述 Franka API 共用的坐标变换、TCP offset、夹爪控制和 IK
  收敛辅助函数。

`reduced` 表示只向策略提供受限、以公开观察为基础的函数；`privileged` 表示接口可能读取
仿真器真值，只适合 oracle、调试或明确允许 privileged state 的实验。

## 视觉 API

- `vision/graspnet.py`：初始化 Contact-GraspNet，并从深度图或点云生成抓取候选。
- `vision/molmo.py`：通过 Molmo 根据文本目标返回图像位置提示。
- `vision/owlvit.py`：通过 OWL-ViT 执行开放词汇目标检测。
- `vision/sam2.py`：提供 SAM2 文本/检测结果和点提示分割客户端。
- `vision/sam3.py`：提供 SAM3 文本与点提示分割，以及分割结果可视化。

这些模块中的部分函数是模型服务客户端；使用前需要启动相应服务并正确配置服务地址。

## 运动规划 API

- `motion/pyroki.py`：PyRoki IK 和轨迹优化服务客户端。
- `motion/pyroki_context.py`：保存 PyRoki 机器人模型、碰撞几何和规划上下文。
- `motion/pyroki_snippets/`：IK、带基座 IK、碰撞约束、静止位姿代价、多目标 IK、
  manipulability 和 trajectory optimization 的本地求解片段。
- `motion/curobo.py`：cuRobo 服务客户端。
- `motion/curobo_api.py`：cuRobo 抓取轨迹、携物规划和轨迹执行封装。

## 其他机器人与配置

- `r1pro/control.py`：R1Pro 的视觉、抓取、运动规划、底盘和双臂控制 API。
- `r1pro/utils.py`：R1Pro 使用的坐标、点云和相机辅助函数。
- `robosuite/controllers/config/robots/`：Panda joint controller 配置。
- `libero/`：LIBERO 集成的包入口。

## 配套工具

- `utils/camera_utils.py`：从嵌套 observation 中提取 RGB 图像。
- `utils/depth_utils.py`：像素反投影、深度图转点云、彩色点云和深度可视化。
- `utils/serve_utils.py`：视觉/规划服务的 HTTP POST、超时和重试处理。
- `utils/visualization_utils.py`：分割 mask、定向包围盒、图像点和三维轴可视化。
- `utils/video_utils.py`：视频编码、写入以及图像 resize/crop。
- `utils/execution_logger.py`：记录 code-policy 工具调用、文本、图像和执行历史。

## 使用与依赖边界

若将 `vla-mender/` 加入 Python 搜索路径，可以从 `knowledge.api` 导入注册表或具体实现，
例如：

```python
from knowledge.api.base_api import get_api, list_apis
from knowledge.api.franka.libero_osc_reduced_skill_library import (
    FrankaLiberoApiReducedOscSkillLibrary,
)
```

本地 Franka/LIBERO API 使用 `knowledge.api.env_protocol.BaseEnv` 结构协议，不再因
类型接口而 import Cap-X。具体 repair runtime 由 `workflow.research` 提供本地 LIBERO
adapter，并管理视觉/运动规划服务。API 仍依赖相应的 LIBERO/robosuite、视觉模型和
运动规划 Python 依赖；`r1pro` 等与 LIBERO repair 无关的可选后端保留各自的运行时
依赖。
