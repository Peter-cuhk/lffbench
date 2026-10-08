# LFF-Bench agent harness（`lffbench/agent/`）

把 VLM 智能体（默认 GPT-6 Astra，经 OpenAI Responses API）接到 LFF-Bench 任务上：工具接口、观测与历史打包、
记忆模式、反馈粒度、协议控制（cross / within）、日志与指标。对应设计文档 `docs/02-benchmark设计.md` §4、§5。

> 本文档里"实测"指在本机跑出来的结果；"假设 / 估计"是没验证过的，单独标出。

## 1. 文件

| 文件 | 内容 |
|---|---|
| `tools.py` | 工具 JSON schema（`tool_specs`）和执行器 `ToolExecutor`（底层全部调 `Skills`） |
| `camera.py` | 相机内外参、深度反投影 `pixel_to_world`、`world_to_pixel`、图像方向说明（robosuite `camera_utils`） |
| `context.py` | 后端无关的对话结构、尝试记录 `AttemptRecord`、记忆选择 `select_history`、全部提示词文本 `PromptBuilder`、关键帧选择、旧图裁剪 |
| `harness.py` | 协议控制器 `AgentHarness` + `HarnessConfig` |
| `backends/openai_responses.py` | `OpenAIResponsesBackend`（GPT-6 Astra） |
| `backends/scripted.py` | `ScriptedBackend`：不调模型，把任务自带的 scripted 策略经工具接口跑一遍 |
| `backends/chat_completions.py` | `ChatCompletionsBackend`：OpenAI 兼容 Chat Completions（本地 Qwen3-VL 服务、vLLM 等） |
| `runlog.py` / `replay.py` | 运行目录写入；从运行目录重建尝试记录（给 mismatched 当 donor） |
| `metrics.py` / `cost.py` | succ@k、ΔICL、重复动作率；token 计价 |
| `../../scripts/run_agent.py` | 命令行入口 |
| `../../scripts/agent_compare.py` | 两个运行目录配对算 ΔICL（bootstrap 区间） |
| `../../scripts/agent_cost.py` | 从运行目录（真实或 dry run）估算 GPT-6 成本 / 延迟 |
| `../../tests/test_agent_*.py` | 单元测试 + 仿真端到端测试 |

## 2. 快速开始

```bash
cd <repo> && source run_env.sh && export OMP_NUM_THREADS=1
# 管线自检（不调模型；同时记录"如果换成 GPT-6，每次请求多大"）
python scripts/run_agent.py --task l5_slide_to_target --backend scripted --memory full --feedback F2 --n 3 --k 5 --dry-run-payload
# GPT-6（需要 key；见 §6）
export OPENAI_API_KEY=...            # 必需
export OPENAI_BASE_URL=...           # 可选：走网关 / 代理时
python scripts/run_agent.py --task l5_slide_to_target --backend openai --effort medium --memory full --feedback F2 --n 20 --k 5 --seed0 5000
python scripts/run_agent.py --task l5_slide_to_target --backend openai --effort medium --memory none --feedback F2 --n 20 --k 5 --seed0 5000
python scripts/agent_compare.py --with runs/agent/l5_slide_to_target/<full 运行> --iid runs/agent/l5_slide_to_target/<none 运行>
# 测试
python -m pytest -q tests/test_agent_payload.py tests/test_agent_protocol.py   # 无仿真，约 1–2 分钟（CPFS 上 import 慢）
python -m pytest -q tests/test_agent_sim.py                                    # 仿真，含 n=3 端到端
```

常用开关：`--instruction direct|indirect`、`--privileged-locate`（消融：给 `locate`）、`--image-res 512`、
`--keyframes 4`、`--keyframe-res`、`--max-steps 30`（每次尝试的工具调用上限）、`--no-auto-obs`（只在 `get_observation` 时给图）、
`--grasp-report bool|name|none`、`--early-stop auto|yes|no`、`--donor-run DIR`（mismatched 的 donor）、`--seeds 5000,5001`。
种子约定（我定的）：1000–1999 开发（validate.py 也用 1000+），5000+ 留作正式测试。

## 3. 工具接口

坐标一律**世界系、米**（按本次任务要求；设计文档 §4 草稿写的是"机器人基座系"，两者只差一个平移，这里统一用世界系，与 Skills 和各任务代码一致）：桌面 z = 0.90，机器人基座在 x = −0.66, y = 0，+x 远离机器人，+y 机器人左侧，+z 向上。
位置指夹爪抓取点（两指尖中点，`gripper0_grip_site`）。夹爪始终竖直向下；`yaw` 单位是**度**（LLM 处理度比弧度少出错），
yaw 0 时两指沿世界 y 轴开合（实测：手指 body 在 y = ±0.02），yaw 90 沿 x 轴。yaw 按 180° 取模：`Skills.move_to`（共享代码，
2026-10-06 协调者改）会取与当前 yaw 最近、且 |yaw| ≤ 90°+0.4 rad（约 113°）的等价角，避免手腕绕 180° 把手臂卷进关节限位；
`move_to` 报告里的 `yaw_used_deg` 是实际用的角度。
方向词：harness 生成的文字里一律以机器人为参照，并在 system prompt 里定义——forward = +x、backward = −x、left = +y（机器人左手）、
right = −y、up = +z，"不是图像里的左右"；同时说明 agentview 图里机器人的左边出现在图像右侧。任务自己的 F2 文本由任务作者统一。

| 工具 | 参数 | 底层 | 说明 |
|---|---|---|---|
| `move_to` | x, y, z, yaw, speed∣null | `Skills.move_to` | speed 为 null 时按控制器最快（约 0.6 m/s）；yaw 按 180° 取模；超出工作空间的目标被裁剪并在报告里注明 |
| `open_gripper` / `close_gripper` | – | `Skills.set_gripper` | |
| `push` | x, y, dir_x, dir_y, distance, speed, z∣null, runup∣null | `Skills.push` | 语义与 `Skills.push` 完全一致：闭爪，到 (x,y) 后方 runup（默认 0.06）处、高度 z（默认 0.92），沿方向扫到 (x,y) 前方 distance 处，再抬 12 cm；朝向由 `Skills.push` 自动定 |
| `get_observation` | – | 渲染 | 返回 agentview + 腕部图像和本体状态 |
| `pixel_to_world` | camera(agentview∣wrist), u, v | 深度反投影 | u=列（从左数）、v=行（从上数），坐标就是模型看到的那张图（默认 512×512）；返回世界点、离桌面高度、深度 |
| `done` | reason | – | 结束本次尝试（within 协议：结束一个分段，任务被检查） |
| `locate`（仅 `--privileged-locate`） | object_name（枚举） | 特权 | 物体原点的真实世界坐标 + 碰撞体 AABB；只用于"去掉感知难度"的消融 |

所有工具 schema 都是 OpenAI strict 模式（所有字段 required、可选字段可为 null、禁止额外字段）。

**动作工具的执行报告**（JSON 文本，作为 function_call_output 返回）：
```json
{"ok": true, "tool": "move_to", "reached": true, "yaw_used_deg": 0.0, "eef_pos": [-0.418, 0.019, 0.920], "eef_yaw_deg": 0.5,
 "gripper": "closed", "gripper_width_m": 0.001, "holding_object": false, "touched_object": false, "sim_time_s": 13.7}
```
- `reached`：控制器是否到达（2 cm 内，`Skills` 的判据）；`push` 给 `sweep_reached`。
- `holding_object` / `touched_object`：默认只给布尔（`--grasp-report bool`）——真实机器人能感知"夹住了东西 / 碰到了东西"，
  但不知道是哪个物体；物体名会泄露识别类任务的答案。`--grasp-report name` 时给物体名（调试 / 消融）。
  接触在**每个 MuJoCo 子步**检查（包装 `env.sim.step`，一个控制步 = 25 子步）：实测快推时手指与方块的接触不到一个控制步，只在控制步边界查会漏掉（l5 上推了 4.4 cm 却一次接触都没记到）。不碰 `Skills.recorder`，任务自己的注入钩子不受影响。
- 任务若定义 `drain_events()`（L6 有），其返回的事件文本会作为 `events` 加进当次报告（L6 作者的约定）。
- 参数错误、未知工具、JSON 解析失败都返回 `{"ok": false, "error": ...}`，占一步预算，不中断尝试。

**隐藏 bias（L1）怎么处理**：所有命令位置都经 `Skills`，bias 照常加上。本体状态按"机器人自己以为的位置"报告：
`eef_pos = 真实末端 − bias`。也就是说机器人认为自己准确到了指令点，只有相机能看出偏差——这就是标定误差的本义；
如果报告真实末端，agent 对比指令和报告就能直接读出 bias，L1 就失效了。`pixel_to_world` 和 `locate` 给真实世界坐标。
（测试 `test_tool_reports_and_hidden_bias` 实测：bias (3, −2, 0) cm 时报告与指令差 < 1 cm，真实末端 = 指令 + bias。）
L1 任务作者在 `l1_bias.py` 的 `TrackedSkills.reported_eef_pos()` 里独立采用了同一约定，两边一致。

**`pixel_to_world` 精度（实测，l5 场景方块顶面中心，投影→反投影）**：agentview 512px 误差约 1.5 mm，256px 约 3.4 mm；
wrist 均 < 0.1 mm。图像方向（由外参算出并写进 system prompt）：agentview 图右 = +y、图下 = +x（机器人在图上方）；
wrist 在 yaw 0 时图右 = −y、图下 = −x。

## 4. 观测、消息格式与历史打包

请求由三部分组成，按缓存友好的顺序排列：
1. **instructions（system）**：机器人 / 坐标 / 相机 / 工具约定、协议（k 次尝试、重置到同一状态）、反馈粒度说明、
   "新尝试开始前先用 1–3 句话评估前几次尝试"（这句话就是"自报失败原因"的来源，记录为 `diagnosis`）。
2. **历史项（user，非 live，不裁剪）**：`TASK: <指令>` + 前几次尝试的打包（按记忆模式选择）。
3. **当前尝试（live）**：尝试编号 + 当前 agentview / wrist 图 + 本体状态；之后每一步：模型的 function_call →
   function_call_output（执行报告）→ 新观测（两张图 + 本体状态）。

每次尝试的打包（≤4 张关键帧 + 动作摘要 + 反馈）。下面是 scripted dry run 第 2 次尝试第 1 次调用的 `input`，原样摘自
`runs/agent/l5_slide_to_target/20261006-060757_scripted-adaptive_full_F2_e2e/requests/request_007.json`（图像以占位符代替，
`[BREAKPOINT]` 表示该内容块带 `prompt_cache_breakpoint`）：
```
== user（历史项，live=False）
TASK: Push the red cube once, straight away from the robot, so that it slides and comes to rest inside the green target zone.

=== Your earlier attempts ===

--- Attempt 1 ---
Actions executed:
1. close_gripper() -> eef (-0.208, 0.000, 1.173); width 0.002 m
2. move_to(x=-0.419, y=0.021, z=1.02, yaw=0, speed=null) -> NOT reached; eef (-0.394, 0.023, 1.032)
3. move_to(x=-0.419, y=0.021, z=0.92, yaw=0, speed=null) -> reached; eef (-0.418, 0.019, 0.920)
4. move_to(x=-0.204, y=0.021, z=0.92, yaw=0, speed=0.3) -> reached; eef (-0.202, 0.021, 0.921); touched an object
5. move_to(x=-0.204, y=0.021, z=1, yaw=0, speed=0.3) -> reached; eef (-0.205, 0.020, 0.997)
6. done(reason="scripted adaptive policy finished with params {'speed': 0.3}")
Outcome feedback: Attempt failed. The cube stopped 5.2 cm past the centre of the target zone.
Key images (start of attempt, after step 1 (close_gripper), after step 4 (move_to), end of attempt, after objects settled):
[start of attempt] <image>
[after step 1 (close_gripper)] <image>
[after step 4 (move_to)] <image>
[end of attempt, after objects settled] <image>
=== End of earlier attempts ===  [BREAKPOINT]
== user（当前尝试，live=True）
This is attempt 2 of at most 5. The scene has been reset.

Current observation:
agentview camera: <image>
wrist camera: <image>
Proprioception: {"eef_pos": [-0.2085, -0.0, 1.1733], "eef_yaw_deg": 0.0, "gripper": "open", "gripper_width_m": 0.0411, "holding_object": false}  [BREAKPOINT]
```
之后每一步依次追加：模型输出项（reasoning（加密）/ message / function_call）→ `function_call_output`（执行报告 JSON）→
user 项 `Observation after step N (<tool>):` + 两张图 + 本体状态。完整请求（含 instructions 和 tools）见同目录 `requests/*.json`。
- 关键帧（agentview）：尝试开始、最多 2 张中间帧（优先 push / 开合爪 / 碰到物体的步）、结束帧（`task.outcome()` 让物体静止之后渲染）。
- 反馈：`task.feedback(inst, outcome, level)`。F0 不给反馈行；F1 只有成功 / 失败；F2 加结构化测量（任务的 `detail`）。
- 历史头不带计数，所以第 j+1 次尝试的历史项以第 j 次的为前缀（只多一块），前缀可被缓存命中（单测 `test_history_prefix_is_append_only`）。
- live 图像超过 `max_live_images`（默认 16）时，把最旧的裁到预算的一半（滞回，减少前缀变化）；历史关键帧从不裁剪。
- 一张图就是一个 JPEG（quality 90）；发给模型的字节和 `images/` 下存的文件完全相同。

**记忆模式**（`--memory`）：

| 模式 | 下一次尝试看到什么 | 用途 |
|---|---|---|
| `none` | 什么都没有；提示词里也不提"多次尝试"，每次请求与第 1 次完全相同 | 独立重来，测 pass@k^iid |
| `full` | 之前全部尝试 | 主条件 |
| `last` | 只有最近一次 | 记忆长度消融 |
| `mismatched` | 另一个实例（不同布局和隐变量）的前 j−1 次尝试，措辞与 full 完全相同 | 错配对照：用的是具体信息还是"泛泛练习" |

mismatched 的 donor：`--donor-run DIR` 复用一个已跑完的 full 运行（按种子循环错位配对，保证 donor ≠ 自己；不用再付一次钱）；
不给时自动用同一后端、memory=full 在 `seed + 10000` 上先跑一遍 donor（存到运行目录下 `donors/`）。

**早停与 F0**：默认（`--early-stop auto`）F1/F2 在首次成功时结束 run；F0 不早停、跑满 k 次——否则"还有下一次尝试"本身就泄露了"上次失败"。

**协议**：`cross` 每次尝试前 `reset_instance`（同布局、同隐变量）。`within` 只在开始重置一次；`done` 结束一个分段：
（若任务定义 `go_home()` 先调用它，L6 有）→ 保持 20 个控制步 → `outcome()`；未成功且未到终止条件（`outcome["terminal"]` 或
`outcome["overshoot"]`，或任务的 `is_terminal(out)`）就带着反馈从当前状态继续。within 下各记忆模式的语义同上（none = 每段都清空上下文）。

## 5. 后端

- **`OpenAIResponsesBackend`**（`--backend openai`）：`client.responses.create(model="gpt-6-astra", instructions, input, tools,
  tool_choice="auto", parallel_tool_calls=False, store=False, include=["reasoning.encrypted_content"], reasoning={"effort": ...})`。
  - 无状态：每次请求重发完整（裁剪后的）上下文；上一轮返回的 reasoning 项（加密内容）原样回放，模型在一次尝试内保留思路。
  - effort 校验：gpt-6-astra 只接受 low / medium / high / xhigh / max（不支持 none）。
  - 缓存：自动把 `prompt_cache_key` 设为 `lffbench:<task>:<seed>`；默认在"历史项末尾"和"最新输入末尾"打显式
    `prompt_cache_breakpoint`（openai SDK 3.24 的字段，gpt-5.6 以后支持）。网关不认这个字段时用 `--cache-breakpoints none`。
    `--prompt-cache-options '{"ttl":"30m"}'`、`--prompt-cache-retention 24h`、`--service-tier`、`--extra-body` 透传。
  - 用量：记录 input / cached / cache_write / output / reasoning tokens 和每次调用的墙钟延迟。
- **`ScriptedBackend`**（`--backend scripted --scripted-policy adaptive|naive|oracle|blind`）：任务自己的 `execute(inst, params)`
  在工作线程里跑，`task.sk` 换成代理，每个 `Skills` 原语变成一次 schema 合法的工具调用交给 harness 在主线程执行
  （队列交接，同一时刻只有一个线程碰 MuJoCo）。参数来自任务的 `default/adapt/blind/oracle_params`，只看记忆模式允许的历史、
  只看反馈粒度允许的字段（F2 结构化测量 / F1 成功标志 / F0 无；看不够就退回 blind 规则）。它读实例布局（特权），只用来测管线。
  支持实现了 `execute` 的 cross 任务；within 任务和只重写了 `run_scripted` 的任务不支持（会报错说明）。
- **`ChatCompletionsBackend`**（`--backend chat --chat-base-url http://127.0.0.1:8765/v1 --chat-model ...`）：OpenAI 兼容
  Chat Completions + function calling，用于本地开源 VLM（见 §8）或任何 vLLM / SGLang 端点。live 图像上限降到 4。

## 6. 怎么接 GPT-6（需要用户提供）

1. `OPENAI_API_KEY`（必需），可选 `OPENAI_BASE_URL`（网关 / 代理）。本机没有 key，代码从未发过真实请求。
2. 账号对 `gpt-6-astra` 的访问权限和 Tier（Tier 1 限额 500 RPM / 500K TPM；单个 run 串行，每次调用 ≤ 几十 K token，Tier 1 够用）。
3. Python 包：`setup/install.sh` 建的 venv 里直接装了 `openai==3.24.0`。（原开发机的 libero-cpu venv 没有 `openai`，
   它装在一个 `pip install --target` 目录里，由环境变量 `LFF_OPENAI_SDK_DIR` 指定；后端在 `import openai` 失败时把该目录
   **追加**到 `sys.path` 末尾，venv 里已有的包始终优先。）
4. 第一次真跑建议：`--n 2 --k 3 --effort low --dump-requests 5`，看 `requests/*.json`（图像已省略）和 `events.jsonl` 里的
   `llm_call.usage`，确认网关接受 `prompt_cache_breakpoint` / `include` 等字段后再放大。
5. 并行：一个进程一个 MuJoCo 环境、串行跑实例；要并行就按 `--seeds` 切分开多个进程（每个进程 `OMP_NUM_THREADS=1`）。
6. 代理（实测）：本机环境变量里有 `HTTPS_PROXY=http://127.0.0.1:1080` 和 `ALL_PROXY=socks5h://127.0.0.1:1080`。openai SDK 3.x
   底层的 httpx2 只要看到 socks 代理变量就要 `socksio`，否则建客户端时直接 ImportError；我已把 `socksio==1.0.0` 装进同一目录，
   现在在这些变量存在时能正常建客户端（实测）。真实请求会走 `HTTPS_PROXY`；能否经这个代理到达 OpenAI / 你的网关没测（没有 key）。

## 7. 日志与指标

运行目录 `runs/agent/<task>/<时间>_<backend>_<memory>_<feedback>[_tag]/`：
- `config.json`：命令行、harness 配置、后端配置、任务信息。
- `events.jsonl`：`run_start`（实例参数，去掉 `_` 开头的私有键）、`llm_call`（延迟、usage、成本估计、模型文本、reasoning 摘要、
  工具调用、请求大小；dry run 时加 `dry_run_request`）、`tool_call`（参数、报告、仿真墙钟、仿真步数）、`observation`（图像路径、本体）、
  `attempt_end`（outcome 全字段、给模型看的反馈、F2 反馈、结束原因、done 理由、自我诊断、显示了哪些历史、关键帧路径、动作摘要、
  该尝试的 LLM 延迟和仿真时间）、`run_end`（run 级汇总）、`llm_error`。
- `images/seed<s>/a<attempt>_s<step>_<n>_<cam>.jpg`：模型看到的每一张图。
- `summary.json`：每个 run 的 `first_success`、`success_by_attempt`、残差（outcome 里所有 `*_err`）、token、成本、延迟；以及汇总：
  `succ_at`（j = 1..k，按 run 计）+ Wilson 95% 区间、`attempt1_success`、首次成功尝试数均值 / 中位数、
  `pass_at_k_iid_analytic = 1 − (1 − p1)^k`、`repeated_action_rate`（失败后下一次动作序列与上次相同（1 cm 取整）的比例，
  "同错复犯"的通用近似）、每 run 调用数 / 延迟 / token / 成本、缓存命中率。
- ΔICL：正式算法是同种子配对的 `succ@k(full) − succ@k(none)`（`scripts/agent_compare.py`，配对 bootstrap 区间、不一致对计数）；
  `delta_icl_vs_analytic_iid` 只是用本组 attempt-1 成功率算的便宜参照。

## 8. 成本 / 延迟估算方法

- 真实运行：成本 = Σ调用 [(input − cached − cache_write)·$10 + cache_write·$12.5 + cached·$1 + output·$50] / 1e6
  （gpt-6-astra 价格，>272K 输入时输入 ×2、输出 ×1.5；见 `cost.py`）；延迟直接用记录的每次调用墙钟。`scripts/agent_cost.py <运行目录>`。
- 跑之前：用 scripted 后端加 `--dry-run-payload` 跑同一配置，记录每次调用如果发给 GPT-6 的请求大小，再
  `scripts/agent_cost.py <dry run 目录> --out-tokens 1500 --latency 20 --calls-scale 2` 外推。假设（未验证）：图像按 32px patch
  每 patch 1 token 计（512² ≈ 256 token）；文本 4 字符 ≈ 1 token；同一尝试内上一调用的输入全部命中缓存、每次尝试第一调用全不命中（偏保守）；
  scripted 的调用次数是 LLM 的下限（LLM 还会看图、反投影、检查），用 `--calls-scale` 放大；每次调用输出 token 和延迟按参数给
  （RoboICL 实测 GPT-6 Astra 每次调用 30–50 s，那是 50–90K 输入、high/xhigh 时的数）。
- 本机 dry run 实测数字见 §10。

## 9. 已知限制

- **l5_slide_to_target 的绿色目标区在图像里看不见（实测，任务侧问题，未改任务文件）**：`apply_instance` 把 zone 的碰撞关掉了，
  但它是 free joint，`settle` 期间直接穿过桌面掉下去（reset 后 z ≈ 0.81），agentview 和腕部图里都没有绿色区域。
  scripted 测试不受影响（它只看 F2 数字），但 VLM 智能体只能靠 F2 的文字找目标——F0/F1 条件下这个任务对 VLM 不可解。
  需要任务作者把 zone 固定住（例如不加 free joint、或每步把它放回 / 设 gravcomp / 设为 mocap）。
- l5 的 scripted 执行里第一个 `move_to`（到推杆起点上方 x≈−0.42, z≈1.02）每次都跑满 250 步（12.5 s 仿真）仍差约 2.5 cm 报 `NOT reached`，
  下一步下降时又到位了；这是 `Skills.move_to` 在靠近基座处的行为，不是 harness 的问题，但 VLM 会在报告里看到 `NOT reached`。
- 渲染：每步两张 512² 图（OSMesa CPU）约 0.5 s；本机负载高时更慢。仿真时间对比 LLM 延迟可以忽略。
- 一个 run 内串行调用模型；没有实现 async tool calling / mid-turn steering / WebSocket（Ultrafast 推荐）。
- 没有实现"每次尝试后单独问一次模型失败原因"（会多一次调用）；自报原因来自每次新尝试开头的那 1–3 句（`diagnosis`）。
- `ScriptedBackend` 不支持 within 协议任务（L2 / L3-within / L5-B / L6 的 scripted 参照写在各自的 `run_scripted` 里，
  不是 `execute(params)`）；within 协议逻辑用假任务做了单测。
- 工具里没有 `wait`；scripted 代理遇到 `Skills.hold` 直接执行并计数（`backend_meta.untooled_hold_steps`）。
- 图像 token 的计法、GPT-6 每次调用的延迟都是假设，第一次真跑后用 `usage` 和 `latency_s` 校准。
- 本地开源 VLM 后端：见 §11。

**任务作者可选的钩子**（harness 会自动使用，没有就跳过）：`agent_instruction(inst, variant)`、`agent_object_names`、
`on_agent_tool(inst, name, args, report)`（每次工具调用后，可用于注入）、`drain_events()`（事件文本进报告）、
`go_home()`（每次尝试结束、测 outcome 之前调用）、`is_terminal(out)` 或 `outcome["terminal"]`（within 不可恢复失败）、
`feedback(..., indirect=bool)`（按指令写法改写反馈）。

## 10. 测试与实测结果（2026-10-06，本机实测，8 核负载 30–40）

单元测试（`source run_env.sh` 后 `python -m pytest -q tests/test_agent_*.py`）：
- `test_agent_payload.py` 6 项通过：请求结构、strict 工具、图像 data URL 能解码成 512² JPEG、缓存断点位置、effort 校验、
  无 key 报错；**真实 openai SDK 3.24 + 进程内 mock HTTP 传输**的两轮往返（URL `…/v1/responses`、`Bearer` 头、body、usage
  解析含 cache_write、reasoning 加密项回放、function_call / function_call_output 的 call_id 配对）；chat 后端消息格式。
- `test_agent_protocol.py` 9 项通过（假任务、无仿真）：full / none / last / mismatched、F0 / F1 / F2、F0 不早停、within 不重置、
  历史前缀只追加、关键帧选择、配对 ΔICL。
- `test_agent_sim.py` 5 项通过（单独 143 s；协调者改了 `Skills.move_to` 的 yaw 取模之后，三个文件 20 项一起重跑，305 s，全部通过）：像素↔世界往返、bias 隐藏、工具报错 / 裁剪 / locate 门控 / 接触检测、
  **ScriptedBackend 在 l5_slide_to_target 上 n=3、k=5 端到端**、mismatched 自动 donor。

端到端运行（留在 `runs/agent/l5_slide_to_target/` 下，可以直接看图和 events）：

| 运行 | succ@1..5 | 首次成功（seed 1000/1001/1002） | 每 run 调用数 | 每 run 仿真 + 渲染墙钟 |
|---|---|---|---|---|
| `20261006-060757_scripted-adaptive_full_F2_e2e` | 0.33 / 0.67 / 1 / 1 / 1 | 3 / 2 / 1 | 12 | 15.5 s |
| `20261006-060913_scripted-adaptive_none_F2_e2e` | 0.33 / 0.33 / 0.33 / 0.33 / 0.33 | – / – / 1 | 22 | 16.2 s |

- 经工具接口跑的 scripted adaptive 与任务自己的参照 rollout（`results/l5_slide_to_target/validate.json`）**逐次一致**：
  8 次尝试的推速完全相同，停止误差 along_err 相同到 0.1 mm（如 seed 1000：+5.22 / −1.14 / −0.49 cm）。说明工具接口足以表达
  这个任务的参照策略，bias / 物理没有被 harness 改变。
- memory=none 时 scripted 策略每次都用默认推速，结果完全重复（`repeated_action_rate` = 1.0），正好是"独立重来"的退化情形；
  `agent_compare.py` 给出 ΔICL = 1.0 − 0.33 = 0.67（n=3，bootstrap 区间 [0, 1]，只是管线演示）。
- dry run 请求大小（`agent_cost.py`）：每次调用约 4.7K 输入 token（估计：文本 /4 + 每张 512² 图 256 token），同一尝试内
  约 78% 可命中缓存；第 2 次尝试首个请求 6 张图（4 关键帧 + 2 实时），到尝试末尾约 16 张。
  估计成本（gpt-6-astra 价格，**输出 token 和延迟是假设**）：每次调用输出 1500 token、20 s 时，每 run 约 $1.1、API 时间约 4 分钟；
  按 LLM 调用数是 scripted 的 2 倍算，约 $2.1 / run、8 分钟；输出 4000 token、40 s 时约 $5.1 / run、16 分钟。
  成本主要在输出（reasoning）token，不在图像。一组条件 20 个实例 ≈ $40–100；ΔICL 至少要 full + none 两组。

另外用 scripted 后端在 **l1_bias_place**（别人刚写的 L1 任务，cross 协议）上跑了 n=2、k=5 的兼容性检查（运行目录在 scratch，未保留）：
两个种子都在第 2 次尝试成功（第 1 次 F2 反馈"夹爪在奶酪中心右侧 3.9 cm、后方 1.4 cm"，adaptive 据此补偿 offset），
0 个错误——harness 不需要为 L1 做任何改动；L1 作者的 `TrackedSkills` 记录、bias 都经工具调用照常生效。

## 11. 本地开源 VLM 后端（Qwen3-VL-8B-Instruct，PPU）

能跑（实测）。由子代理在约 1 小时内做通：`scripts/qwen_vl_server.py` + `scripts/qwen_vl_server.sh`（OpenAI 兼容
`/v1/chat/completions`，支持工具调用 / 多图 / 多轮），用系统 python3.12（torch 2.6 PPU 版）+ 单独目录
`/mnt/cpfs/workspace/env/qwen3vl-ppu-overlay`（只放 transformers 4.57.6 / tokenizers 0.22.1 / huggingface_hub 0.36.0，
只对服务进程加 PYTHONPATH，不改任何已有环境）。PPU 上 SDPA 自带 GQA 在解码步会报错，服务里改成先展开 K/V 头再算（`--sdpa-native-gqa` 可恢复）。

```bash
nohup bash scripts/qwen_vl_server.sh --port 8765 --warmup > logs/qwen_vl_server.log 2>&1 &   # 等日志出现 "serving ..."
python scripts/run_agent.py --task l5_slide_to_target --backend chat --chat-base-url http://127.0.0.1:8765/v1 \
    --chat-model Qwen3-VL-8B-Instruct --memory full --feedback F2 --n 2 --k 3 --max-steps 15 --keyframes 3
kill <服务 PID>   # 用完停掉，显存约 18 GB
```
- 加载：页缓存热时权重 18 s（本次实测）；冷读 CPFS 约 6.5 分钟（子代理实测 385 s）。
- 端到端实测（`runs/agent/l5_slide_to_target/20261006-061100_chat_full_F2_qwen_smoke`，n=2、k=3、每次尝试 15 步上限）：
  81 次模型调用、0 次后端错误、0 次格式错误的工具调用；每次调用延迟均值 5.1 s（p50 4.9 s，p95 9.5 s），
  输入均值 6.3K token（最大 10.3K），输出均值 113 token；用到了 move_to / push / close_gripper / get_observation / pixel_to_world / done，
  新尝试开头都写了对上一次的评估（`diagnosis`）。成功 0/2：模型反复 `move_to(0, 0, 0.9)`、没有先定位方块，
  加上 l5 的目标区本来就看不见（§9）。这只说明**管线对真实 VLM 是通的**，不说明 8B 模型的能力；之后我在 system prompt 里加了
  "物体位置不会直接给出，用 pixel_to_world 从图像定位"一句（这次运行之后加的，未再测）。
- 用途：在花 GPT-6 的钱之前，先用本地模型检查提示词、工具描述、日志和指标。
