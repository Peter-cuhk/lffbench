# L6-B `l6_count_into_box`：往不透明投放箱里放恰好 N 个方块，中途一块滑落

> 2026-10-07。"实测" = 本机仿真跑出来的；"推测" = 没验证。

**任务 → 隐藏信息 → 所需能力 → 调整后的动作**
- 任务：把桌上一堆一模一样的橙色方块，恰好放 N 个进一个不透明的投放箱（N = 4 或 5，写在指令里）。
- 隐藏信息：第 p+1 次搬运时方块从夹爪里滑落、掉回桌上（p ∈ {1..N−1}）。滑落之后箱子外观不变、看不见里面，桌上剩下的方块数和 p 无关——"已经放进去几个"只存在于 agent 自己的历史里。
- 所需能力：Save（每放进一个就记一笔）、Retrieve（滑落后取回计数）、Utilize（滑落那次不算，只再放 N−p 个）；Perceive（看出方块是从口里掉进去了还是滑回了桌上）。
- 调整后的动作：滑落之后再放 N−p 个（把滑落的那块重新拿起来放进去，再放剩下的），而不是"从头再放 N 个"或"随便猜个数"。

| 项 | 内容 |
|---|---|
| 类别 / 等级 | procedural_memory / L6 |
| 能力 | Perceive, Save, Retrieve, Utilize |
| 协议 | within：一个 episode 不重置，最多 5 次 done（"尝试"/segment），粒度同 l6_swap_interrupt |
| 代码 | `lffbench/tasks/l6_count.py`，`CountIntoBox`；验证 `python scripts/validate.py --task l6_count_into_box --n 20 --k 5` |
| 场景 | T = N + e + p 个 4 cm 橙色方块（同一外观，11 个对象预先声明，用不到的停在桌子外的地上），随机散放（中心间距 ≥ 8.5 cm；朝向 ±45°，且这个朝向对应的抓取轴上手掌 ±11 cm 内不碰邻块和箱子，见 §7-6）。投放箱：深棕色静态 fixture，外形 21.6 × 23.6 × 10.3 cm，放在桌子左侧或右侧（y = ±0.22–0.25 m），箱盖靠机器人一侧开 7 × 7 cm 方口；口下是低摩擦斜坡（μ = 0.03），方块落进去后滑到盖子下面 |
| 指令（直呼其名） | "Put exactly four of the orange cubes into the dark box, through the opening in its lid. Cubes cannot be taken back out of the box."（N 按实例填 four / five；`agent_instruction` 钩子） |
| 指令（间接描述） | "Drop exactly four of the small orange blocks into the closed brown container through the square hole in its top. Blocks cannot be taken back out of the container." |
| 成功 | done 时箱内恰好 N 个方块，夹爪里没有方块 |
| 终止失败 | 箱内 > N（放多了取不出来）；`outcome["terminal"] = True`，harness 停止该 run |
| 失败注入 | 第 p+1 次搬运途中：某块方块被夹着、高于桌面 ≥ 5 cm、离开它被拿起的位置 ≥ 触发距离（5–9 cm），且此时箱内恰有 p 个——夹爪张开，方块出现在它被拿起的位置旁 0.5–2 cm（随机朝向，保证仍可抓）。按状态触发（每个控制步检查），不依赖脚本，接 VLM agent 不用改代码 |
| 隐变量 | p ∈ {1..N−1}（均匀）；e ∈ {0, 1}（与 p 独立）；N ∈ {4, 5}；布局、箱子位置/侧、触发距离、落点偏移和朝向 |
| 先验为什么失败 | 滑落后的画面：桌上 R = N + e 块，箱子和开局一样。只看当前画面的 agent 不知道里面已有几个：按"箱子看起来是空的"从头放 N 个 → 必然放多（终止）；随便猜一个数 → 期望只有 1/N 对 |
| 要从失败里带走 | 自己已经成功放进去几个（滑落那次不算）→ 还差 N − p 个 |

## 1. 尝试粒度、事件和箱子
- 尝试（segment）= 从当前状态开始操作直到 agent 宣布 done；脚本策略里 done 前把机械臂收回初始位姿（`go_home`）。done 后测 outcome、给 F0/F1/F2；没成功也没终止就从当前状态继续，最多 5 次。
- 尝试内部的即时事件（`task.drain_events()`，harness 附在当时那次工具调用的报告里）：每有一块方块从口里掉进箱子 → "A cube dropped into the box."；滑落 → "A cube slipped out of the gripper and fell back onto the table."。前者相当于"听到/看到方块掉进去"，让计数只依赖记忆、不依赖从图像里判断方块是否进了口。
- 箱内隐藏：方块滑过开口远端 2 cm 后，被挪到盖子下面的存放位：先填远端两排的两侧位置（8 个），再填正对开口的中间通道（2 个），再在远端一排上叠第二层（5 个）。这样新掉进去的方块总能沿中间通道滑过开口，不会沿滑道排成一列、退回到开口下方被看见。挪动发生在任何相机都看不到的地方。
- 看不见里面（实测，seed 1001 和 1004，最终版存放顺序；测试脚本在开发用 scratchpad 里，没进仓库）：箱内有 1、3、5、6 个方块时，腕部相机在开口上方 36 个位置（高 3/6/10 cm、前后 ±7 cm、左右 ±5 cm）+ agentview 各拍两张，一张保持原样、一张把箱内方块移走（其余一切相同）。差异像素只在几何边缘呈 1 像素细线，且差异像素数与箱内方块数无关（seed 1001 四种个数下都是 424 / 276 像素），是渲染阴影图随场景包围盒变化的噪声；开口里没有任何方块的像素。

## 2. 反馈（F2 = 结论 + 当前可观测状态 + 本次尝试里的滑落事件；不报箱内个数、不报坐标）
validate 实测原文（seed 1000–1019）：
- 成功（adaptive，seed 1003，N = 4，p = 1）："Attempt succeeded. During this attempt: A cube slipped out of the gripper and fell back onto the table. The box holds exactly 4 cubes. Now 2 cubes are on the table."
- 放少（naive，seed 1004，N = 5，p = 1，猜"已有 3 个"只补 2 个）："Attempt failed. During this attempt: A cube slipped out of the gripper and fell back onto the table. The box holds fewer than 5 cubes. Now 3 cubes are on the table."
- 放多（blind，seed 1003，从头再放 4 个）："Attempt failed. During this attempt: A cube slipped out of the gripper and fell back onto the table. The box holds more than 4 cubes. Cubes cannot be taken out of the box, so the task can no longer be completed. Now 1 cube is on the table."
- 方块落在箱盖上时末尾加 "… and 1 on the box lid."（代码格式，本次验证没有出现）。
- 尝试内部即时事件（CLI 自测原文，`open_gripper` 的报告）：`"events": ["A cube dropped into the box."]`。

刻意不给的：箱内现在有几个。如果 F2 说"箱里有 3 个"，就是裁判替 agent 计数，题目不再考记忆（"放少了 → 补齐"一步就成）。"fewer than N" 只给方向：由于放多是终止的，非终止失败本来就意味着放少了。

## 3. 脚本参照策略（"感知"用仿真状态；每次拿离开口最近的方块，抓取方向选手掌更空的那条轴）
| 策略 | 有什么记忆 | 滑落后 / 之后的尝试 |
|---|---|---|
| oracle | 特权：箱内真实个数 | 放到箱内恰好 N 个 |
| adaptive | 自己数成功掉进去的次数（滑落那次不算） | 放到自己的计数 = N；之后若 F2 说 fewer than N，每段再补 1 个 |
| naive（1 次尝试） | 滑落时丢失计数（只剩当前画面） | 画面看不出进度：在 0..N−1 里均匀猜"已有几个"，补足到 N |
| blind（k 次尝试） | 无 | "不知道做到哪了，从头来"：再放 N 个（必然放多 → 终止）；若没终止，之后每段再放 N 个 |
| skip（附加对照） | 数的是搬运次数（把滑落那次也算上） | 第 1 段放了 N−1 个；第 2 段读到 F2 "fewer than N" 后补 1 个 |
| inc1（附加对照） | 无状态 | 滑落后每段只放 1 个就 done，利用 harness"成功即停" |

## 4. headroom 验证（实测，n = 20，seed 1000–1019，k = 5，`results/l6_count_into_box/validate.json`）
| 策略 | @1 | @2 | @3 | @5 | 验收线 |
|---|---|---|---|---|---|
| oracle | 100% | | | 100% | @1 ≥ 95% ✓ |
| naive | 30% | | | 30% | @1 ≤ 30% ✓（压线；期望 22.5%，见下） |
| adaptive | 100% | 100% | 100% | 100% | @5 ≥ 90% ✓ |
| blind | 0% | 0% | 0% | 0% | 明显低于 adaptive ✓ |

同一次运行的其他实测：
- 80 个 run 全部触发了滑落，滑落时箱内个数都等于采样的 p；482 次搬运 = 402 次掉进箱子 + 80 次滑落，0 次没夹住、0 次没投进口。所以 oracle / adaptive 的 100% 不靠运气，naive / blind 的失败全部来自数错。
- 这 20 个实例：N = 5 有 12 个、N = 4 有 8 个；p = 1/2/3/4 分别 10/6/3/1 个（p = 1 偏多）；开局方块数 T = 5–9。
- naive 的 6 次成功都是恰好猜中 p（1000、1002、1006、1007、1010、1014）。均匀猜的期望是 1/N：这 20 个实例上 22%，2000 个实例上 22.5%；20 个种子抽到 6 次是偏高的一次抽样（naive 的猜测用独立随机流 `seed + 7919`，开发中没有挑种子）。14 次失败里 6 次放多（终止）、8 次放少。
- blind 20/20 放多（多 1–4 个）并终止。
- 整次验证 701 s（同时跑着另一个进程）。

缩略图：`init_<seed>.png`（开局）、`naive_end_<seed>.png`（naive 结束）、`interrupt_<seed>.png`（滑落后 4 个控制步的 agentview | 腕部：夹爪空着、方块回到桌上）。

## 5. 对照（实测，`results/l6_count_into_box/controls.json`）
| 对照 | @1 | @2 | @3 | @4 | @5 | 说明 |
|---|---|---|---|---|---|---|
| skip（数搬运次数，把滑落那次也算上） | 0% | 100% | 100% | 100% | 100% | 第 1 段都少放 1 个；第 2 段读到 "fewer than N"，结合自己的记忆补 1 个。说明 F2 的方向信息 + 记忆足以纠正"跳步"，但 @1 全错 |
| inc1（无状态，滑落后每段只放 1 个） | 10% | 25% | 80% | 100% | 100% | 在第 N − p 段成功。它不需要任何记忆，只利用了 harness"成功就停"：见 §8 |
| 关掉滑落（`task.inject = False`），前 10 个种子 | adaptive 100%、naive 100% | | | | | 不打断时脚本策略都能做对，失败全部来自滑落后丢失计数 |

演示视频：`videos/l6_count_into_box_adaptive_s1000.mp4`（N = 4、p = 2：放进 2 个后第 3 块滑落，adaptive 不把这次算进去，重新拿起它放进去，再放 1 个，成功）；`videos/l6_count_into_box_blind_s1000.mp4`（同一实例，滑落后从头再放 4 个 → 箱里 6 个，终止）。

## 6. 无记忆策略最多能拿多少（解析，假设执行完美）
假设执行完美，只看滑落后再放几个（`controls.json`）：

| 无记忆策略 | @1 | 说明 |
|---|---|---|
| 从头放 N 个（= blind） | 0% | p ≥ 1，必然放多 |
| 在 0..N−1 里均匀猜已有几个（= naive） | 22.5%（2000 个实例；这 20 个种子 22% 期望，实测 30%） | |
| 固定补 k 个（知道 N，挑最好的 k） | 30%（2000 个实例）；这 20 个种子上"补 3 个"碰巧对 55% | 最好的单次猜测；p 均匀，所以上限约 1/(N−1) |
| 每段只补 1 个（inc1） | @1 10%，@(N−p) 必成 | 依赖"成功即停"，见 §8 |

- 结论：只看当前画面、只做一次时，最好的无记忆策略约 30%；有记忆（adaptive）100%。但多段 + 成功即停时，"每段补一个"这种无状态策略最终必成（§8 第 1 条）。

## 7. 设计决定和理由
1. **R = N + e 与 p 独立。** 桌上剩余块数不泄露进度；只有"开局一共几块"（第一帧）+ 现在几块 才能推出 p——这也是记忆（回看第一帧），算合法的另一条记忆路径。
2. **放多终止、放少可补。** 对应真实投放箱/投币箱：进去就拿不出来。这样"从头再来"（重复步骤，RMBench 里观察到的失败）有确定的代价。
3. **F2 不给箱内个数**（§2）。
4. **滑落 = 传送**：夹爪张开 + 方块出现在它被拿起的地方旁边（不是物理下落），和 l6_swap 一样，落点可控（仍可抓、不压别的块）。
5. **箱内存放位**（§1）：实测纯物理滑道下方块沿滑道排成一列，第 3 个起就退回到开口正下方，能被看见；改为滑过开口后挪到盖下存放位。
6. **抓取可行性写进采样**（开发中实测的两类失败）：(a) Panda 手掌沿开合轴约 ±11 cm 宽、底面比 4 cm 方块顶面低，邻块在这个范围内会被手掌撞开；(b) 抓取方向转到接近 ±90° 时，在部分位置腕关节（joint 6）顶到 3.75 rad 上限、OSC 卡住，之后的动作全都到不了（早期版本 seed 1012 / 1013 / 1020 连续抓取失败、一个 run 拖到十几分钟）。现在：方块朝向在 ±45° 内采样，并要求这条"小转角"抓取轴在手掌范围（±11 cm）内离邻块中心 ≥ 6 cm、离箱子 ≥ 3.5 cm；方块中心间距 ≥ 8.5 cm；滑落落点做同样检查。脚本策略只用这条小转角轴抓取，搬运时把夹爪转回 yaw 0（方块转多少都能进 7 cm 的口）。
7. **箱子放在侧面**、离方块 ≥ 12 cm：箱高 10 cm，放在桌子中间会挡住 agentview 里它后面的方块；手掌靠近箱子时也会撞到箱盖。

## 8. 已知问题 / 效度威胁
实测到的：
- **within 协议的"成功即停"会把进度泄露给无状态策略。** inc1 对照：滑落后每段只放 1 个就宣布完成，第 N − p 段一定成功（@4 100%）。原因是放少不终止、harness/robot_server 在第一次成功时结束 run——这个停止信号本身就告诉了 agent "够了"。memory=none 时新 agent 不会这么做（它不知道箱里已有东西，自然会放 N 个），所以这不影响 blind 的含义，但报 ΔICL 时要知道：比较的是"自然的无记忆行为"，不是所有无记忆策略的上限。若要堵死，可以加一个"done 即封箱"的变体（放少也终止，只剩 1 段），当前没做。
- **每段命令预算不够。** CLI 自测：搬一块（张夹爪、到上方、下降、夹、抬起、到口上方、下降、松手、抬起）用 9 条命令；N = 5、再加滑落重做 = 6 次搬运 ≈ 40–55 条。`robot_server` 默认每段 40 条、`HarnessConfig.max_steps` 默认 30，第 1 段很可能被截断（截断 = 自动 done，within 下还能继续，但 succ@1 被压低；memory=none 时还会在截断处换一个没有记忆的新 agent，等于额外丢一次进度）。建议这个任务每段预算设到 80。没改共享脚本。
- `robot_server.py` 不看 `outcome["terminal"]`（只有 harness 看），CLI 会话里放多之后 agent 还能继续操作，结果不变。
- 滑落和箱内存放都是传送。滑落：方块从夹爪里"消失"、出现在桌上；存放：在看不见的地方发生，画面上没有痕迹（§1 的可见性测试）。存放位按"先两侧、后中间通道"排，10 个以内新掉进去的方块都能滑过开口；开发早期版本（中间通道先填）在箱里 ≥ 7 个时出现过方块卡在开口下方（只在已经放多、任务已失败的 run 里）。
- 方块抓取方向只在 ±45° 内（见 §7-6）。agent 如果选接近 90° 的夹爪转角，在部分位置会把腕关节顶到限位、之后的动作到不了位（实测于脚本策略）；这是 Panda + OSC 的共性问题，不是本任务特有，但这个任务搬运次数多，碰到的机会更多。

推测、未验证的：
- 全历史在上下文里的强 VLM 数 4–5 个"掉进去"事件应该不难；难点更可能是滑落那一次：报告里先看到夹爪动作成功、再看到滑落事件，需要意识到这次不算。要让它真正考 Save，可以用 `announce_drops = False`（不发"掉进去"事件，只能从图像判断方块是否进了口）和/或 harness 层的"有限历史 + 自己的笔记"条件（同 l6_swap_interrupt §8）。这两个变体都没验证。
- VLM 也可能"保守地"每次补一个然后 done（相当于 inc1），这在 memory=full + F2 下是合理的学习行为，succ@k 曲线上会表现为 @1 低、后面逐步上升。

## 复现
```bash
cd /mnt/cpfs/workspace/agentic-robotics/lffbench && source run_env.sh && export OMP_NUM_THREADS=1
python scripts/validate.py --task l6_count_into_box --n 20 --k 5 > logs/validate_l6_count.log 2>&1
python scripts/make_video.py --task l6_count_into_box --kind adaptive --seed 1000
python scripts/l6_controls.py l6_count_into_box 20   # 对照 -> results/l6_count_into_box/controls.json
```
