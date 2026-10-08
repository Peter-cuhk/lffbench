# L5 `l5_tablecloth_pull`：把木板从方块下面抽出来，让方块落进绿带（替代 l5_tip_threshold）

> 状态（2026-10-07）：已实现并通过 headroom 验证（n = 30）。代码 `lffbench/tasks/l5_tablecloth_pull.py`。
> 2026-10-08：加"只许抽一次"规则（指令写明 + 判定强制），堵掉 cla2 里"先慢拖预定位再抽"的漏洞，见下方「单次抽出规则」。改前备份 `*.bak-20261008-onepull`。
> 这是 l5_tip_threshold 的替代任务：高物体倾倒版本在这台机器人上做不成，证据见 `task_cards/l5_tip_threshold.md`。

| 项 | 内容 |
|---|---|
| 类别 / 等级 | dynamics_adaptation / L5（"搞清楚这个物体怎么动"） |
| 能力 | Perceive（方块落点相对绿带的位置）、Reason（"抽得越快，方块被带走得越少"——反直觉的方向）、Utilize（把速度往对的方向调对的量） |
| 协议 | cross（每次尝试重置到同一实例、同一隐变量），k = 5 |
| 指令（direct） | Pull the wooden board out from under the red cube so that the cube ends up standing on the table completely inside the green band. Touch only the board and its black handle, never the cube. Only one pull is allowed per attempt: grasp the black handle, pull the board out with ONE move command, then release it. Any other move that drags, pushes or nudges the board, before or after the pull, counts as another pull. |
| 指令（indirect） | Pull the long brown plank out from under the small red block … green stripe … Touch only the plank and its black handle, never the block. Only one pull is allowed per attempt: … pull the plank out with ONE move command … |
| 动作 | 只用 move_to / 开合爪：张爪 → 到黑色把手上方（yaw 0，两指沿 y 夹住 4 cm 宽的把手）→ 下到把手中高 → 合爪 → 朝机器人方向（−x）以某个速度拉 25 cm → 张爪 |

## 任务 → 隐藏信息 → 所需能力 → 调整后的动作
- **任务**：桌上一块 30 × 8 × 1 cm 的木板（长边朝前），近端有黑色把手；红色 4 cm 方块立在木板远端（离远端边缘 5 cm）。
  桌面上画着一条横向绿带（6 cm 宽，比方块离机器人近 3.3–7.5 cm，木板两侧能看见）。要把木板抽走，让方块立在桌面上、完全落在绿带内（中心偏差 ≤ 1 cm）。不许碰方块（每个控制步检测机器人与方块的接触）。
- **隐藏信息**：方块底面的摩擦系数 μ（0.08–0.15 对数均匀；方块–木板、方块–桌面都用它），外观对所有 μ 一样。
- **物理**：慢拉时方块随木板一起走（"搭车"）；快拉时木板从下面滑出，滑动摩擦把方块往机器人方向带一段 d 后落到桌上。
  d 随拉速**单调减小**（拉得越快带得越少），随 μ 增大；μ 越大，"搭车"的临界速度也越高。
- **为什么先验会失败**：经典的"抽桌布"先验是能拉多快拉多快（0.6 m/s，控制器上限）。绿带总是放在这个速度带出的位移再往机器人方向约 1.5 cm 处，所以按先验做方块会停在绿带远端（离机器人远的一侧）之外。
- **要从失败里学到**：方块停在绿带的哪一侧、差多少 → 速度该往哪个方向调（停在远侧 = 带得不够 = 要**拉慢**；搭车或停在近侧 = 带得太多 = 要拉快），再按差值调多少。
- **调整后的动作**：把拉速调到与 μ、绿带距离匹配的值（oracle 实测 0.31–0.64 m/s，多数 0.39–0.52；30 个实例里 26 个要比 0.6 m/s 慢）。

## F2 反馈示例（实测文本，自测会话 seed 1000）
- "Attempt failed. The board came out from under the cube and the cube is standing on the table, but not completely inside the green band: its centre is 2.1 cm forward (+x, away from the robot) of the band's centre line. It moved 2.9 cm toward the robot."（0.6 m/s 猛拉）
- "Attempt failed. … its centre is 1.1 cm forward (+x, away from the robot) of the band's centre line. It moved 3.9 cm toward the robot."（0.48 m/s）
- "Attempt succeeded. The board came out from under the cube and the cube is standing inside the green band; its centre is 0.1 cm forward (+x, away from the robot) of the band's centre line. It moved 4.9 cm toward the robot."（0.43 m/s）
- 其他分支（脚本实测）："The board was pulled 25 cm, but the cube rode along on it: it is still on the board and moved 23.9 cm toward the robot."（0.2 m/s）；"The robot touched the cube, which is not allowed."；"The board was not pulled; the cube is still standing on it."；方块翻倒（toppled）。
- 只给相对测量（相对绿带中线偏多少、往机器人哪个方向、总共被带了多少），不给绝对坐标，不给 μ。

## 物理标定（实测，2026-10-07，脚本 scratchpad `l5dyn/cloth.py`、`cloth_cal.py`；木板在 x = 0.05、L_F = 5 cm）
方块被带向机器人的位移 d（cm；R = 搭车）：

| μ \ 拉速 m/s | 0.25 | 0.30 | 0.35 | 0.40 | 0.45 | 0.50 | 0.55 | 0.60 | 0.70 | 0.80 |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.08 | R | 5.4 | 4.1 | 2.9 | 2.7 | 2.3 | 2.4 | 1.8 | 1.9 | 1.9 |
| 0.10 | R | 11.0 | 5.8 | 4.4 | 3.9 | 3.3 | 2.2 | 2.3 | 2.5 | 2.2 |
| 0.12 | R | R | 8.5 | 7.0 | 5.3 | 4.4 | 3.6 | 3.1 | 2.7 | 3.1 |
| 0.15 | R | R | R | 14.3 | 7.6 | 6.0 | 5.5 | 4.3 | 3.9 | 3.7 |

- 0.6 m/s 以上控制器饱和，d 不再变小。同参数重复运行结果一致（确定性）。
- 绿带距离按 μ 取样：d* ∈ [0.33 μ + 0.65 cm, 0.5 μ]（米制写法见代码 `d_range`），即放在"猛拉位移"之外约 1.5 cm、"刚过搭车速度的位移"之内。
- 木板位置在 x 0.03–0.08、y ±0.08 随机；不同位置同一速度的 d 有 ≲ 1 cm 的差（例如 μ 0.08 时 0.6 m/s 的 d 在 1.9–2.6 cm），这就是 naive 有 4/30 碰巧落进去的原因。
- agent 接口和脚本的抓取姿态稍有不同，同一速度的 d 差约 0.5 cm（seed 1000：脚本 0.48 m/s → 4.4 cm，自测会话 → 3.9 cm）。oracle 是按脚本执行用真实 μ 仿真二分得到的。

## 单次抽出规则（2026-10-08，实测）
- **为什么**：cla2（Sonnet 5.5，seed 3000）带历史 FS、独立 S S S，4 个成功全靠先慢拖木板让方块搭车预定位，最后一抽只补一点：
  带历史第 2 次先拖 1.4 cm 再抽；N0 拖 2.7 + 1.3 cm、又推回 0.9 cm 再抽；N1 夹着拖 4 下、又张爪敲把手 7 下；N2 慢拖 4.1 cm 后横向猛抽。
  GPT（full1，seed 2000）3 次尝试也都是先拖 4–5 cm 再抽。摩擦 μ 完全不起作用。
- **规则**（指令写明）：抓住把手，用**一条**移动命令把木板抽出，然后松开；抽之前或之后用其他移动拖、推、蹭木板都算又抽了一次。
- **判定**（`ClothSkills.pull_check`）：每条 move_to 记录木板两端的 3-D 位移路程，只在机器人"驱动"木板时累计（接触木板，或木板这一步比上一步更快；木板自己滑只会减速），所以敲击/松手后的惯性滑行不算到下一条命令上。开合爪命令不算抽。
  - 驱动木板 ≥ `PULL_EPS` = 0.3 cm 的移动命令算一次抽；多于 1 次 → 失败。
  - 抽开始时方块离初始位置 > `PRE_TOL` = 1 cm → 失败（堵"很多条 < 0.3 cm 的小推"和合爪挪板）。
  - 其余移动命令合计驱动木板 > `STRAY_MAX` = 0.5 cm → 失败。
  - 违规时 `failure = "several_pulls"`，成功与否只看规则 + 原判定；F2 文本把违规句放在原来的落点描述前，例如："The board was moved by 2 separate move commands (4.1, 18.1 cm); only one pull is allowed. The board came out from under the cube and …"。F1 只有 "Attempt failed."（与 slide / L7 一致）。
- **阈值依据**（实测）：靠近把手的移动 0 cm；agent 合爪夹把手 0.06–0.20 cm（脚本 0.46 cm）；抽完松爪木板还会滑 0.3–1.0 cm（是开爪命令，不计）；漏洞的预拖每条 0.9–4.1 cm，张爪敲击每下 1.1–5.4 cm。
- **回放验证**（把日志里的原始命令在新代码上重放，`python scripts/replay_la.py <run_dir> <attempt> [--seg] [--fb]`；旧代码重放与日志完全一致）：

| run | 原判定 | 新判定 | 抽的次数（cm） |
|---|---|---|---|
| cla2 带历史 第 1 次（单次抽，没给速度 = 控制器最快） | F missed_band | F missed_band（反馈文本不变） | 1（19.3） |
| cla2 带历史 第 2 次 | S | F several_pulls | 2（1.4, 19.3） |
| cla2 独立 N0 | S | F several_pulls | 4（2.7, 1.3, 0.9, 23.4） |
| cla2 独立 N1 | S | F several_pulls | 11 |
| cla2 独立 N2（拖后横抽） | S | F several_pulls | 2（4.1, 18.1） |
| full1 GPT player 第 1–3 次 | F missed_band | F several_pulls | 2 / 2 / 3（每次先拖 4.2–4.7 cm） |

- **脚本参照策略不受影响**：加规则后重跑 `validate.py --n 30 --k 5`，oracle / naive / adaptive / blind 各 @k 曲线与 10-07 完全相同（下表）；228 次脚本尝试全部恰好 1 次抽、规则未触发（抽前方块位移最大 0.5 cm，其余移动 0 cm）。旧结果备份 `results/l5_tablecloth_pull/validate.json.bak-20261008-onepull`。
- **对抗用例**（seed 3000，μ 0.105，d* 4.5 cm，会话 scratchpad 脚本，未入库）：单次抽 0.6 / 0.45 m/s → 1 次抽、规则不触发；6 × 2.5 mm 慢推再猛抽（落点 4.6 cm，原本会成功）→ several_pulls；单次横抽 → 1 次抽、missed_band（方块几乎不动）；抽完用合爪夹爪把木板往前顶 2 cm → several_pulls。

## headroom 验证（n = 30，k = 5，脚本参照策略，实测）
`python scripts/validate.py --task l5_tablecloth_pull --n 30 --k 5` → `results/l5_tablecloth_pull/validate.json`

| 策略 | @1 | @2 | @3 | @4 | @5 |
|---|---|---|---|---|---|
| oracle（知道 μ，仿真二分拉速） | 100% | | | | 100% |
| naive（0.6 m/s 猛拉） | 13% | | | | 13% |
| adaptive（按有符号落点误差：单侧有界时乘/除 1.25，双侧有界时割线/二分） | 13% | 83% | 93% | 97% | 100% |
| blind（0.6 ± 0.06 m/s 随机重试） | 13% | 37% | 47% | 57% | 60% |

- naive 的 4 次成功（seed 1001/1004/1013/1021）都是该位置下 0.6 m/s 的位移恰好落在 d* ± 1 cm 内（见上条位置差异）。
- blind 偏高：0.54–0.6 m/s 的随机抖动常常正好比猛拉多带 1 cm。按设计文档 §8，用户已确认盲重试偏高不是问题（看的是带历史 vs 独立重来）；adaptive 与 blind 差 40 个百分点。

## agent 接口自测（实测，2026-10-07，端口 9881，rgb 模式）
`start_robot_session.sh l5_tablecloth_pull 1000 full F2 9881 direct 3 selftest rgb 1`：README 只有通用说明和任务指令，没有 μ 或绿带距离；
`status` / `open_gripper` / `move_to` / `close_gripper` / `move_to(speed)` / `done` 都正常，三次尝试的反馈见上面 F2 示例（0.6 → 0.48 → 0.43 m/s，第 3 次成功）。
小问题：拉动过程中 move_to 报告的 `holding_object` 常为 false（robosuite 的抓取判据），但木板确实被拉走了，不影响任务判定。

## 演示视频
`videos/l5_tablecloth_pull_adaptive_s1003.mp4`（seed 1003，μ 0.090，d* 3.8 cm：0.6 m/s 只带 2.2 cm → 0.48 m/s 带 2.8 cm → 0.384 m/s 带 3.6 cm，成功）。

## 已知问题 / 威胁（未验证的标【推测】）
- 反直觉方向是本任务的核心难点：第一次失败后，直觉型 agent 可能会"拉得更猛"，那只会让方块更远离绿带。
- ~~两步策略："先慢拉让方块搭车到某处，再猛拉抽走"也是合法解法~~（10-07 的判断，错了）：cla2 里 Sonnet 4/4 的成功、GPT 3/3 的尝试都是这么做的，慢拖量正好按 F2 报的差值补，摩擦不再起作用。10-08 起由单次抽出规则禁止。
- 容差 ±1 cm 偏紧；agent 抓取姿态与脚本的差异会造成约 0.5 cm 的系统差。
- 隐变量是摩擦（与 l5_slide_to_target 同为摩擦），但交互方式和速度–位移关系的方向都不同（推一下 vs 抽走支撑）。
- 不许碰方块是规则约束（每步接触检测），不是物理约束；方块在机器人够得着的范围内。
