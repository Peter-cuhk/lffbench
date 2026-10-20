# L5-A `l5_slide_to_target`：推一下，让方块停在目标区

代码：`lffbench/tasks/l5_slide.py`　结果：`results/l5_slide_to_target/validate.json`（10-19 容差修改后重跑，n = 20、k = 5；同日容差修改前的版本备份为 `validate.json.bak-20261019-tol`，10-08 的 n = 30 版本备份为 `validate.json.bak-20261019-relax`）

> **2026-10-19 肘关节上限修改后重跑 validate（实测）**：共享层 `lffbench/envs.py` 把第 4 关节上限收紧到 −0.55（任务侧会话改）。重跑 `validate.py --n 20 --k 5`，80 个 run 的全部记录和改前逐项相同。
>
> **2026-10-19 第二次改（容差，用户定）：远边（推的方向那一侧）放宽，方块可以压在远边的绿线上。判定：方块中心最多越过框中心 2.0 cm（原来 1.0 cm）；近边和左右不变（1.0 cm，完全在框内）。指令措辞同步改了，画出来的方框没改。**
> 起因：自测里方块伸出远边 0.8 cm，agentview 里看着像在框里，违反 `DESIGN_INSIGHTS.md` §2.1 / R5（失败要在 agentview 里看得出；陷阱和可见性冲突时优先保可见性）。实测、阈值和取舍见下面"容差修改（2026-10-19）"。改前备份：`lffbench/tasks/l5_slide.py.bak-20261019-tol`、本卡 `task_cards/l5_slide_to_target.md.bak-20261019-tol`、`results/l5_slide_to_target/validate.json.bak-20261019-tol`、`results/l5_slide_to_target/arms/validate.json.bak-20261019-tol`（及同目录 3 张 `seed3000_*_naive_end.png.bak-20261019-tol`）。**任务代码又改了，容差修改前跑的 run 不能和之后的混算。** 这次也没改共享代码（robot_server、envs、skills、task_base、scripts）、`sets/*`、`DESIGN_INSIGHTS.md`、`RUN_WITH_YOUR_API.md`，要同步的内容见"已知问题"。

> **2026-10-19 改（用户定）：一次尝试里可以击打多次（推过头可以把方块夹回去再打）；成功判定改成"最后一段必须是离开夹爪后的自由滑行，且 ≥ 3 cm"；击打命令等方块停稳后才返回。**
> 10-08 起的"每次尝试只能击打一次（一条 move 命令推方块，首次接触后夹爪最多再前进 4 cm）"是为堵准静态顶推漏洞加的（exp1 / cla1 / la2 里模型用 0.05 m/s 把方块一路顶进框，摩擦完全不起作用），但它把尝试内的试错硬挪到了尝试间（违反 `DESIGN_INSIGHTS.md` §1：尽量在一次尝试里做成，真机上 reset 要人复原场景）。现在指令不再限制次数，漏洞改由新判定挡住；做法照 `l7_key_turn`（10-16，commit 609c33f）。下面"10-19 前"的各节保留原样。
> 共享代码（robot_server、envs、skills、task_base、scripts）、`sets/*`、`DESIGN_INSIGHTS.md` 都没改。**任务代码改了，10-19 前跑的 l5_slide_to_target run 不能和新 run 混算**（`api_batch.py contract` 会因代码哈希不同报出来）。

## 新规则和判定（2026-10-19）
**指令**（README 的 Task 段，`instruction` = `instruction_indirect`；容差修改后的版本，见下面"容差修改"一节）：
> Push the red cube straight away from the robot so that it slides and comes to rest in the green square marked on the table. On the side of the square nearest the robot and on its left and right, no part of the cube may lie on or beyond the green border line. On the far side, away from the robot, the cube may lie on the border line, but no part of it may stick out beyond that line. The cube must slide the last part of the way on its own: after it leaves the gripper it has to slide at least 3 cm, with nothing touching it, before it comes to rest in the square.

容差修改前（10-19 上午）是："Push the red cube straight away from the robot so that it slides and comes to rest completely inside the green square marked on the table. The cube must slide the last part of the way on its own: …"（后半句相同）。相对 10-08 版去掉了 "once" 和 "Strike it with a single move command: after the gripper first touches the cube it may move at most 4 cm further forward"；原来的 "the cube must slide the rest of the way on its own" 改写成新判定的可检验形式（判定要能从指令推出来，§2.1）。不说"可以推几次"（与 key_turn 一致）。

**成功** = finish 时（outcome 先 settle 2 s，方块还在动就再等最多 3 s）同时满足：
1. 方块静止（线速度 < 3 mm/s、角速度 < 0.05 rad/s），平放在桌面上（中心高度在 TABLE_Z + 2 cm ± 5 mm 内；夹着悬空不算），在框内：沿推方向方块中心在框中心前后 **−1.0 cm（近边，完全在框内）到 +2.0 cm（远边，可以压在远边绿线上，`TOL_PAST`）** 之间，横向在 ±1.0 cm 内（容差修改前沿向是 ±1.0 cm）；
2. 最后一段运动是自由滑行：从机器人最后一次"移动了方块的接触"结束时方块的位置（release）到最终位置，水平距离 ≥ **D_MIN = 3 cm**；且结束时机器人没有正在接触或夹住方块（结束时还在接触、并且这次接触移动过方块，自由滑行记 0）。

**实现**（`CubeWatch`，挂在 `env.sim.step` 上，每个 2 ms 物理子步查一次；和 `agent/tools.ContactTracker` 同样的包法，只读状态）：
- 接触 = 机器人任何 geom（`robot0_*`、`gripper0_*`）与方块的接触；夹住必然有接触。快速击打的接触只有 3–16 个子步（6–32 ms，实测），按 50 ms 控制步查会漏。
- 相邻接触间隔 < 25 子步（50 ms）合并成一段；一段结束 50 ms 后，方块离这段开始时的位置 ≥ 1 mm（3-D）或仍在运动，就算"移动了方块"，release = 这段最后一个接触子步时方块的位置。碰了静止的方块但没让它动（< 1 mm）的接触忽略（对抗用例 B3：击打成功后夹爪压在方块顶上再抬走，仍判成功）。
- outcome 新字段：`free_slide_cm`、`release_xy`、`touching_at_end`、`at_rest`、`n_moves`（移动了方块的接触段数）、`n_strikes`、`strikes`（每次击打的 start / release / rest、`pushed_cm`、`free_slide_cm`、沿向和横向误差、result）、`n_missed_strikes`、`trap_fired`、`recovered_in_attempt`、`irrecoverable`。去掉了 `n_pushes`、`max_advance`。
- failure：`position`（不在框内）、`slide_too_short`（在框内，但自由滑行 < 3 cm 或结束时还接触着）、`not_at_rest`、`not_touched`。原来的 `several_pushes`、`carried` 不再出现。
- F2（只在消融里用）新增两句："After it last left the gripper it slid only X cm on its own; it has to slide at least 3 cm." / "The gripper was still touching (or holding) the cube when the attempt ended."
- 容差修改后，F2 第一句括号里的状态改成 "(in the square)" / "(it sticks out beyond the far green line)" / "(it sticks out past the edge of the square nearest the robot)" / "(not in the square)"（横向出框时后面照旧加 "It also drifted X cm to the left / right"），原来是 "(completely inside the square)" / "(not completely inside the square)"。击打记录（`strikes[].result`、`missed`、`n_missed_strikes`、`trap_fired`）用同一个判定。

**错误事件**：一次"击打" = 一段移动了方块的接触 + 之后 ≥ 1 cm 的自由滑行，在方块停稳时记录；停在框外（result = past / short / beside / off）就是一次错误事件。`n_missed_strikes` = 次数，`trap_fired` = 至少一次，`recovered_in_attempt` = 陷阱触发过且本次尝试最终成功（字段名与 key_turn 一致）。滑行中又被机器人碰到记 `interrupted`，不算错误；夹回、放下、慢推（自由滑行 < 1 cm）不算击打。论文四分类：第 1 次尝试成功且无 `trap_fired` 是 CS，有 `trap_fired` 而成功是 IR。
**不可恢复**：方块中心低于桌面 5 cm（掉下桌），或离指令可达范围（`agent/tools.py` 的 WS_LO / WS_HI）边界不到 3 cm，记 `irrecoverable = True`，不自动结束尝试（不设 `terminal`）。

**观测 bug 修复**：`SlideSkills.move_to` / `set_gripper` 在动作结束后，如果方块还在动且没被夹住，就让手臂原地保持，直到方块连续两个控制步静止（上限 100 步 = 5 s），然后才返回、拍照。等待时间计入仿真时间预算。同一串 la 命令（seed 3000，22 cm 助跑），击打命令返回时方块的位置（实测）：

| 代码 | 推速 | 返回时方块已走 | 返回时方块速度 | 最终停点 |
|---|---|---|---|---|
| 10-19 前 | 0.30 m/s | 3.34 cm | 0.27 m/s | 11.73 cm |
| 10-19 前 | 0.45 m/s | 12.74 cm | 0.19 m/s | 16.84 cm |
| 10-19 | 0.30 m/s | 11.73 cm | 0 | 11.73 cm |
| 10-19 | 0.45 m/s | 16.84 cm | 0 | 16.84 cm |

击打命令多花 0.1–0.8 s 仿真时间。接口自测里击打回复的图像与之后单独 `observe` 的图像逐像素相同（见下）。

## 容差修改（2026-10-19，实测）
**怎么量的**：主表 5 个实例（3000、3001、3003、3004、3007）和 validate 的 20 个（1000–1019），把方块瞬移到框附近（沿向 −4 … +4 cm、步长 2 mm，边界附近 1 mm；横向 ±1.0 … 3.0 cm），用和服务器一样的相机和渲染出图（`TASKS[...](cam_res=512)`、`envs.render`、agentview 512 × 512、存成 JPEG q90）。用 MuJoCo 分割图量两样东西：绿线在方块所在的那几列还露出几个像素；方块的红色轮廓越过绿线**外沿**几个像素（下面叫"越线"）。同时在 JPEG 图上按颜色数越线的红色像素，两者一致。腕部相机量两个位置：脚本击打后的位置（夹爪在方块起点上方、比推的高度高 8 cm；自测里 agent 击打后也停在这附近）和框中心正上方 25 cm。脚本、图和逐实例的数在 `results/l5_slide_to_target/visibility_20261019/`（`visibility_summary.json`）。

**agentview 的几何**：相机在桌子远端、朝机器人看（俯角约 42°），方块是朝相机推的，所以"远边"离相机近。沿向 1 cm ≈ 4 px，横向 1 cm ≈ 6 px；整个框只有约 35 × 24 px，绿线 3 px 宽（3007 是 2 px）。

| agentview，25 个实例的范围（实测） | 远边（中心越过框中心 +） | 近边（中心不到框中心 −） | 左右 |
|---|---|---|---|
| 方块所在列的绿线被完全挡住 | +1.5 … +1.7 cm | −4.0 … +2.8 cm **始终挡住**（主表 5 个实例；方块自己的身子挡住它后面的近边线，方块在框中心时也一样） | ±1.0 … 1.6 cm |
| 越线 ≥ 1 px | +1.6 … +1.8 | 没有越线可看 | |
| 越线 ≥ 2 px | +1.8 … **+2.0** | 没有越线可看 | ±1.4 … 2.0 |
| 越线 ≥ 3 px（一条线宽） | +2.0 … +2.2 | | |
| JPEG 上越线的红色 ≥ 2 px | +1.7 … +2.0 | | ±1.4 … 2.1 |

- **远边**：方块越过内沿 0–0.6 cm 时只是把 3 px 的线压细，压满整条线（约 +1.6 cm）以后每往外 2.5 mm 才多 1 px。自测那一下（+1.79 cm）是"线全压住、越线 1 px"；full5N Sonnet noharness0 s3003 第 1 次（+1.64 cm，越线 0 px）agent 自己写的是 "came to rest within the green square outline (front edge about 1-2 mm inside the outer green boundary per wrist camera)"，判了 F。图：`visibility_20261019/selftest_attempt1_final.png`、`full5N_s3003_attempt1_final.png`、`agentview_far_s3000.png`、`agentview_full_s3000.png`。
- **近边**：近边线在方块后面，被方块挡住，画面上没有"越线"可看。唯一的线索是方块前面露出的浅绿色有多少行：方块在框中心 3–4 px，在近边界（−1.0 cm）7–8 px，−1.7 cm（3003 / 3004 的 naive 停点）10–11 px，−2.0 cm 11–13 px，−3.0 cm 15–17 px（框深 23 px）。要从 agentview 看出近边出框，只能从"方块前面的绿色超过框深的一半"去推算【推断：模型多半看不出】。图：`agentview_near_s3004.png`。
- **腕部相机**和 agentview 正好互补。击打后的位置（从后上方看，约 17–22 px/cm）：近边线在方块前面，−1.2 cm 起被压细（10 px 剩 7 px）、−1.6 cm 起全挡住、−1.6 … −1.8 cm 越线 ≥ 2 px，看得清；远边反而失真：方块顶面往前投，中心越过框中心 0.2 cm 就把远边线全挡住（在框里也像出框）。框中心正上方 25 cm：远边 +1.8 cm 越线 ≥ 2 px；近边在 −0.6 cm（还在框里）就被挡住。图：`wrist_post_strike_s3000.png`。

**"看得出"的阈值 = 越线 ≥ 2 px**（红色轮廓压满整条绿线，并且在绿线外沿以外还有 ≥ 2 px）。理由：(1) 上面两个实例是 0–1 px，看着都像在框里 / 停在线上；(2) JPEG q90 的色度隔一个像素采样（4:2:0），1 px 宽的颜色边会被抹掉，JPEG 上按颜色数，越线的红色 ≥ 2 px 出现的位置和分割图一致（+1.7 … +2.0）；(3) 更严的 3 px（一条线宽）要放到 +2.2 cm，会让 3004 twin 的 naive（+2.15 cm）判成功，对照臂就不成立了。

**选的方案：只放宽看不清的远边**。判定 = 中心 ≤ +2.0 cm（`TOL_PAST = 0.020`），近边和左右仍是 1.0 cm；方框不重画；指令改成"近边和左右不能压线，远边可以压在线上、但不能伸出线外"（原文见上面"新规则和判定"）。
- 指令给的边界是"方块前面到远边绿线的外沿"（中心 +1.6 cm），判定再多放 4 mm（< 2 px，画面上分不出"压在线上"和"伸出 4 mm"）。所以判定不比指令严（没有冤判）；多出来的 4 mm 只是看不出的那一段，不能用来绕过摩擦（准静态顶推、夹着放仍然因自由滑行 < 3 cm 失败，那部分判定没动）。凡是判失败的远边终态，agentview 里越线都 ≥ 2 px（25 个实例，方块不转；转了的方块角伸得更出，只会更明显）。
- **没选"把目标沿向画长、判定仍是完全在框内"**：画出来的边界和判定的边界重合时，边界外面总有一段看不出的带子（压线、越线 < 2 px，约 1 cm）。把远边线往外挪 1 cm，这段带子就落在 +2.0 … +3.0 cm，正好是 3004 twin（+2.15）、3003 twin（+2.74）、3001（+2.95）naive 的停点，它们的失败会变得看不出；现在的做法下这几个停点越线约 2–6 px（3004 twin 2–3 px）。这样也不用动 `square_frame_xml`（`l3_wide` 也用它）。
- **近边没改（可见性和陷阱冲突，留给用户定）**：要让近边出框在 agentview 里看得出，得放到方块的前面也过了近边线（中心 −5 cm 左右，近边线才露在方块前面），陷阱就没了。而只要放宽 0.2 cm（到 −1.2 cm）validate 1013 的 naive（−1.18）就判成功，到 −1.45 cm validate 里 5 个 naive（−1.18 … −1.43）都成功，到 −1.66 / −1.73 cm 主表 3004 / 3003 的 naive 也成功。近边出框只有击打后的腕部图看得出。
- **左右没改**（用户只同意放宽沿向）：越线 ≥ 2 px 要到 ±1.4 … 2.0 cm，所以 1.0–2.0 cm 的横向出框在 agentview 里也看不清。旧日志 93 个 attempt 里横向误差 > 1 cm 的有 7 个，沿向都差 ≥ 6 cm，横向从来不是决定因素；脚本的横向误差 ≤ 0.3 cm。
- 代码改动：`TOL_PAST`、`SlideToTarget.tol_past`；`CubeWatch(env, target, tol_short, tol_past, tol_across)` 和 `judge()` 改成不对称；`_adaptive_attempt` 的停止条件和 F2 文字跟着改；指令里不能有分号和括号（指令也写进 BDDL 的 `:language`，`;` 在那里是注释，第一次试的时候建环境报 "Missing close parenthesis"）。`INNER_HALF`、`LINE_W`、`square_frame_xml`、`CUBE_HALF`、`V_MIN`、`V_MAX`、`Z_PUSH` 都没变，`l5_iterative`、`l3_wide` 能正常 import。`l5_iterative_push`（已停用）继承了 `tol_past = 2.0 cm`，只影响它的 `CubeWatch` 击打记录，它自己的 outcome 不用 `CubeWatch`，不变。

**改后的结果（实测）**：
- validate（n = 20、k = 5，种子 1000–1019）：oracle 100 / 100 不变；naive @1 25% → **30%**（1005 停在 +1.90 cm，原来判冲过头，现在在框内）；blind @1 / @5 25 / 65% → 30 / 70%；adaptive 第 1 次尝试仍是 20/20，**CS 5 → 6、IR 15 → 14**（1005 第一击就进）。IR 的修复次数从 1 次 9 个、2 次 5 个、3 次 1 个变成 1 次 11 个、2 次 3 个（1002、1004、1006、1009 各少修一次：其中一击停在 +1.06 … +1.90 cm，原来算冲过头，现在算进框），IR 的命令数平均 18.7（14–34）→ 16.1（14–24），仿真时间平均 60.9 s → 52.2 s（最多 97.8 → 90.0 s）。
- 主表 5 个实例：naive 第 1 次仍全部失败（停点 +6.94、+2.95、−1.73、−1.66、+4.05 cm，k = 5 内都没成功），oracle 第 1 次全部成功（停点不变）。3007 的 adaptive 第二击停在 +1.66 cm，现在第一次修复就成功（14 条命令、47.2 s；原来修 2 次、24 条、66.9 s），其余 4 个和原来一样。主表 naive 离远边界最近的是 3001（+2.95 cm，越线约 6 px）。
- 对照臂（`results/l5_slide_to_target/arms/validate.json` 重跑，旧的是 10-12 用单击规则跑的，10-19 放开击打次数后没重跑过）：5 个实例 7 项检查全部符合预期，所有停点和 10-12 的逐个相同（差 < 0.01 cm），`arm_specs` 不用改。离新边界最近的是 3004 twin 的 naive（+2.15 cm，比 2.0 多 1.5 mm，仍判失败）；validate 1005 的 naive（+1.90）离边界 1 mm。
- 旧日志按新判定重放：见下面"旧日志重放"一节末尾。

## D_MIN = 3 cm 的依据（实测）
- **目标距离**：代码里是 `uniform(5, 12) cm`（再拒绝最高推速够不着的组合），不是本卡旧表写的 6–10 cm。主表 5 个实例 7.9–9.0 cm；validate 种子 1000–1019 最短 5.3 cm（1009）。
- **击打中被夹爪推着走的距离**（release 前的 `pushed_cm`）：0.1–1.7 cm，推速越高越长（22 cm 助跑、停在接触点后 1.5 cm 的脚本击打；agent 自己的击打重放出来是 0.7–1.5 cm）。所以从初始位置一次打进框，自由滑行 = 停点距离 − (0.1–1.7) cm。
- **25 个实例（validate 1000–1019 + 主表 5 个）**：用二分推速让方块停在框的近边（沿向 −0.9 cm）、中心、远边（+0.9 cm），近边处的自由滑行最小 4.05 cm（1009，d = 5.3 cm），中心最小 4.98 cm；主表实例近边最小 6.49 cm。
- **采样下限的最坏情况**（合成实例 d = 5.0 cm）：μ = 0.10 时近边 3.33 cm、沿向 −0.5 cm 处 3.81 cm、中心 4.01 cm；μ = 0.085 近边 3.50；μ = 0.07 近边 3.65；μ = 0.035 近边 3.60。d = 5.5 / 6.0 cm、μ = 0.10 时近边 3.84 / 4.41 cm。
- **结论**：3 cm 是"从初始位置打出、停在框内任何位置的击打都满足"的最大整数值（最坏 3.33 cm）。4 cm 在这 25 个真实实例上也不冤判，但在 d ≈ 5–5.5 cm、μ 大的实例上，框的近端 0.5 cm 左右从初始位置打不进（自由滑行 3.3–3.9 cm），agent 只能先把方块往后挪；这类实例约占采样空间的几个百分点【推断，按 d、μ 的分布估算】。主表实例对 3 / 4 cm 没有区别。
- 容差修改（远边放到 +2.0 cm）只会让停在远边的击打自由滑行更长，最坏的仍是近边，这一节的结论不变【推断：自由滑行 = 停点距离 − pushed_cm】。
- **3 cm 足够挡漏洞**：准静态顶推自由滑行 0–1.2 cm，夹着放 0 cm，放在框前 2–2.5 cm 再轻碰 0–0.14 cm（对抗用例 A1–A3），都远低于 3 cm。
- **代价：短距离击打对 μ 的要求更松**。固定一个击打（按中位 μ = 0.059 调到落在中心），在 9 个对数等距的 μ 上重复：滑行 3.5 cm（4 cm 助跑）2/9 成功（慢速轻碰变成顶推，自由滑行不够）；4.5 cm 4/9；6 cm 3/9；8 cm 2/9（≈ naive 的 25%）。所以 agent 把方块放在框前 4.5 cm 左右再打，所需的 μ 精度比从 8 cm 外打低一些，但仍要从失败的击打里估出速度和距离的关系；这样做是规则允许的（对抗用例 A3 的 3.5 cm 一档，用真 μ 调速后 6 个种子里 4 个成功）。μ 扫描里 4.5 cm 那一档若 D_MIN = 4 cm 是 3/9。要更严就改成 4 cm（用户拍板）。

## headroom 验证（2026-10-19 容差修改后，n = 20，k = 5，seeds 1000–1019，实测）
| 策略 | @1 | @2 | @3 | @5 | 容差修改前（10-19 上午）@1 / @2 / @3 / @5 | 10-19 前（同 20 个种子，单击规则）@1 / @2 / @3 / @5 |
|---|---|---|---|---|---|---|
| oracle（知道 μ，模拟二分推速，一击） | 100% | | | 100% | 100 / 100 / 100 / 100 | 100 / 100 / 100 / 100 |
| naive（0.30 m/s 一击就 finish，不看结果） | 30% | | | 30% | 25 / 25 / 25 / 25 | 25 / 25 / 25 / 25 |
| adaptive（**尝试内**按停点修正，见下） | 100% | 100% | 100% | 100% | 100 / 100 / 100 / 100 | 25 / 65 / 90 / 100（跨尝试割线） |
| blind（0.30 ± 0.05 随机，每次一击） | 30% | 45% | 55% | 70% | 25 / 45 / 55 / 65 | 25 / 45 / 55 / 65 |

- naive 第 1 次交互 14/20 失败（`trap_fired`，冲过头或没到；容差修改前 15/20，1005 停在 +1.90 cm 现在算进框），陷阱有效；oracle 20/20。
- 容差修改后 adaptive 第 1 次尝试：6 个 CS（1001、1003、1005、1007、1012、1016）、14 个 IR；IR 修 1 次 11 个、2 次 3 个（1002、1011、1014）；IR 命令数平均 16.1（14–24），仿真时间平均 52.2 s（最多 90.0 s，1011）。下面这条是容差修改前的数，留作对照。
- **adaptive 的尝试内修复**（容差修改前）：每次击打后看方块停在哪（脚本用真值位置，相当于看图）；没到且差 ≥ D_MIN + 2 cm 就从停点再打，否则（冲过头、只差一点、横向偏 ≥ 5 mm）把方块夹回初始位置再打；推速按本尝试（及之前尝试）见过的 (推速, 滑行距离) 做对数插值。20/20 在第 1 次尝试内成功：5 个是第一击就进（CS），15 个是 IR。IR 的 15 个：修 1 次 9 个、2 次 5 个、3 次 1 个（1002）；**命令数平均 18.7（14–34）**，仿真时间平均 60.9 s（最多 97.8 s）。一次修复 = 夹回 6 条（上方张开、下降夹住、抬起、平移、下降张开、抬起）+ 再打 4 条 = **10 条命令**，满足 R3 "修复要 10 条以上命令"。脚本从没用到"从停点再打"（没到的都只差 1–2 cm，不够 D_MIN + 2 cm）。
- **预算**：默认每次尝试 40 条命令、120 s 仿真时间。容差修改前最坏的 1002（3 次修复）用了 34 条、97.8 s，在预算内；第 4 次修复就放不下了（容差修改后最坏是 2 次修复、24 条、90.0 s）。脚本的仿真时间被 22 cm 助跑拖长：助跑起点在基座坐标 x ≈ 0.21–0.27 m，第 4 关节到限位（−3.07 rad），这条 move 跑满 250 步（12.5 s）还差 1–4 cm（接口自测也遇到）。agent 用短一点的助跑就没有这个问题。不需要改全局预算。
- 【推断】更省的修复是从远侧把方块推回来（约 4 条）再打（4 条），8 条左右；仍有代价，但略低于 R3 的 10 条。
- 意义：放开后，会看停点的 agent 第 1 次尝试就能做成，A@1 会接近 100%；任务信号改为 CS / IR、`n_missed_strikes`（修了几次）和跨尝试是否第一击就打对（与 key_turn 10-16 后相同）。

## 对抗用例（脚本，2026-10-19，实测；种子 3000、3001、3003、3004、3007 + 1009）
| 用例 | 结果（6 个种子） | 自由滑行 cm |
|---|---|---|
| A1 准静态顶推进框（0.05 m/s，终点调到方块停在中心，再慢慢退开） | 6/6 F `slide_too_short` | 0–0.14 |
| A1 同上，0.10 m/s | 6/6 F | 0.26–1.19 |
| A1b 慢推分两段（停在半路看一眼再推） | 6/6 F | 0–0.12 |
| A2 夹起来直接放在框中心、松手抬走 | 6/6 F | 0–0.02 |
| A2b 放在框中心但不松手 | 6/6 F（`touching_at_end`） | 0 |
| A2c 夹着悬在框中心上方 5 cm | 6/6 F `position`（不在桌面上） | 0 |
| A3 放在框中心前 2.0 cm，再轻碰（用真 μ 二分碰的速度） | 6/6 F | 0–0.14 |
| A3 放在框中心前 2.5 cm，再轻碰 | 6/6 F | 0–0.1 |
| A3 放在框中心前 3.5 cm，再轻碰（规则允许的边界） | 4 S（3.03–3.20）/ 2 F（2.06、2.97） | |
| B1 0.30 m/s 一击失败 → 夹回初始位置 → oracle 推速再打 | 6/6 S，`trap_fired`、`recovered_in_attempt` | 4.5–7.6 |
| B2 oracle 一击 | 6/6 S | 4.7–7.8 |
| B3 oracle 一击成功后，合拢的夹爪压到方块顶面再抬走（方块没动） | 6/6 S | 同 B2 |
| B3b oracle 一击后，张开的夹爪在方块上方悬停再离开 | 6/6 S | 同 B2 |
| B4 oracle 一击后，再把方块往前顶 0.9 cm | 6/6 F | 0 |
| B5 0.16 m/s 一击没到（差 ≥ 4.5 cm 才做）→ 从停点再打 | 2/2 S（3003、3004；其余 4 个种子第一击只差 0.8–4.3 cm，跳过） | 4.4–5.7 |

多击打计数（自测和 validate 一致）：冲过头一次再修好 → `n_missed_strikes = 1`、`recovered_in_attempt = True`；两次没打准 → 2。

## 主表实例（`sets/main_v1.txt`：3000、3001、3003、3004、3007；实测）
| seed | d / μ | naive（0.30 m/s 一击） | oracle | adaptive 第 1 次尝试 |
|---|---|---|---|---|
| 3000 | 7.95 cm / 0.043 | F，冲过 +6.94 cm | S（0.226 m/s，−0.22 cm，自由滑行 7.0 cm） | S，修 1 次，14 条命令，35.0 s |
| 3001 | 8.27 / 0.049 | F，+2.95 | S（0.249，+0.03，7.6） | S，修 1 次，14 条，58.3 s |
| 3003 | 9.03 / 0.072 | F，−1.73 | S（0.360，+0.03，7.7） | S，修 1 次，14 条，57.8 s |
| 3004 | 8.27 / 0.066 | F，−1.66 | S（0.420，−0.20，7.8） | S，修 1 次，14 条，11.2 s |
| 3007 | 8.25 / 0.038 | F，+4.05 | S（0.225，−0.15，7.5） | S，修 2 次，24 条，66.9 s；**容差修改后**修 1 次（第二击停在 +1.66 cm，现在算进框），14 条，47.2 s |

新规则下 5 个实例仍然 naive 失败、oracle 成功；naive 的停点与 10-08 一致（`arm_specs/l5_slide_to_target.py` 里的 [orig] 误差逐个对上），所以 full5 的 H0 / twin 标定不用重做。

**容差修改后重跑（实测）**：naive 第 1 次仍然 5/5 失败（停点不变；k = 5 内都没成功；远边那 3 个 +2.95 / +4.05 / +6.94 cm 都在 +2.0 cm 外，近边那 2 个 −1.73 / −1.66 cm 在 −1.0 cm 外），oracle 第 1 次 5/5 成功（推速、停点不变）；adaptive 只有 3007 变了（见表）。对照臂重跑：5 个实例 × 7 项检查（orig naive F、orig oracle S、h0 naive S、twin naive F、twin oracle S、两个交叉检查 F）全部符合预期，停点和 10-12 逐个相同；离新边界最近的是 3004 twin 的 naive（+2.15 cm，仍 F，余量 1.5 mm）。结果写回 `results/l5_slide_to_target/arms/validate.json`（`created` 改成 2026-10-19，加了 `note`；旧的是 `validate.json.bak-20261019-tol`）；同目录 3 张 `seed3000_*_naive_end.png` 重新生成，和 10-12 的图只有机械臂的位置不同（10-19 的"等方块停稳"让手臂停的时刻不同），方块位置一样。

## 旧日志重放（2026-10-19，实测）
用 `scripts/replay_la.py` 的同一套命令处理逻辑（scratch 脚本，另外用日志里的 instance，full5T 的 twin μ 才对得上；v2 世界坐标的 run 不减基座偏移），把 68 个有 `done` 的 run、93 个 attempt 在新代码上开环重放。先用 10-19 前的代码重放做对照：93 个 attempt 的最终停点和日志**逐个一致**（重放可信）。新代码下有 20 个 attempt 停点变了（多为 < 1 cm）：等方块停稳改变了后续命令和滑行中方块的先后关系（开环重放的限制）。`scripts/replay_la.py` 本身也能跑新代码（抽查 cla4 harness 225328 三个 attempt、cla1_rgb，结果与下表一致），但它的 REPLAY 行只打印 `trap_fired`、`recovered_in_attempt`。

**10-08 单击规则之后的批次（cla2、cla4、grok47xh、full5N、full5T，35 个 attempt，全列）**：原来 6 个 S 全部仍是 S；1 个改判 F→S（cla4 harness 225328 第 2 次：一击冲过头 → 夹回 → 又打了两下都停在框内，原判 `several_pushes`，现在 S，`trap_fired`、`recovered_in_attempt` 都是 True）；其余 28 个仍是 F（原标签 `several_pushes` / `carried` 的，方块本来也都在框外，现在标 `position`）。容差修改后（见本节末）又有 1 个 F→S：full5N noharness0 s3003 第 1 次。

| 批次 | run | 尝试 | 原判定 | 原停点 cm | 新判定 | 新停点 cm | 自由滑行 cm | n_missed | trap | IR | |
|---|---|---|---|---|---|---|---|---|---|---|---|
| cla2 | sonnet harness s3000 k3_1008-180651 | 1 | F(several_pushes) | 35.86 | F(position) | 35.87 | 21.68 | 2 | ✓ |  |  |
| cla2 | sonnet harness s3000 k3_1008-180651 | 2 | F(several_pushes) | 29.41 | F(position) | 28.92 | 6.46 | 4 | ✓ |  |  |
| cla2 | sonnet harness s3000 k3_1008-180651 | 3 | S | -0.36 | S | -0.36 | 6.87 | 0 |  |  |  |
| cla2 | sonnet noharness0 s3000 k1_1008-180656 | 1 | F(carried) | -2.51 | F(position) | -2.51 | 5.04 | 1 | ✓ |  |  |
| cla2 | sonnet noharness1 s3000 k1_1008-180700 | 1 | F(carried) | 7.94 | F(position) | 7.94 | 14.73 | 1 | ✓ |  |  |
| cla2 | sonnet noharness2 s3000 k1_1008-180703 | 1 | F(position) | 7.03 | F(position) | 7.03 | 13.8 | 1 | ✓ |  |  |
| cla4 | sonnet harness s3000 k3_1008-214533 | 1 | F(carried) | 14.88 | F(position) | 14.88 | 20.79 | 1 | ✓ |  |  |
| cla4 | sonnet harness s3000 k3_1008-225328 | 1 | F(several_pushes) | 26.37 | F(position) | 22.46 | 10.05 | 4 | ✓ |  |  |
| cla4 | sonnet harness s3000 k3_1008-225328 | 2 | F(several_pushes) | -0.02 | S | 0.89 | 8.18 | 1 | ✓ | ✓ | **改判** |
| cla4 | sonnet harness s3000 k3_1008-225328 | 3 | S | 0.13 | S | 0.13 | 7.32 | 0 |  |  |  |
| cla4 | sonnet noharness0 s3000 k1_1008-225332 | 1 | F(position) | 7.69 | F(position) | 7.69 | 15.26 | 1 | ✓ |  |  |
| full5N | sonnet noharness0 s3000 k1_1012-144850 | 1 | F(position) | 10.4 | F(position) | 10.4 | 16.8 | 1 | ✓ |  |  |
| full5N | sonnet noharness0 s3001 k1_1012-144906 | 1 | F(position) | -4.49 | F(position) | -4.49 | 2.39 | 1 | ✓ |  |  |
| full5N | sonnet noharness0 s3003 k1_1012-144918 | 1 | F(position) | 1.64 | **S**（容差修改后；修改前 F(position)，n_missed 1、trap ✓） | 1.64 | 9.32 | 0 |  |  | **容差改判** |
| full5N | sonnet noharness0 s3004 k1_1012-144932 | 1 | S | 0.96 | S | 0.96 | 7.79 | 0 |  |  |  |
| full5N | sonnet noharness0 s3007 k1_1012-144944 | 1 | F(position) | 15.48 | F(position) | 15.48 | 22.74 | 1 | ✓ |  |  |
| full5N | sonnet noharness1 s3000 k1_1012-144858 | 1 | F(position) | 7.37 | F(position) | 7.37 | 14.04 | 1 | ✓ |  |  |
| full5N | sonnet noharness1 s3001 k1_1012-144910 | 1 | F(position) | -3.88 | F(position) | -3.88 | 1.52 | 1 | ✓ |  |  |
| full5N | sonnet noharness1 s3003 k1_1012-144922 | 1 | S | 0.21 | S | 0.21 | 7.9 | 0 |  |  |  |
| full5N | sonnet noharness1 s3004 k1_1012-144936 | 1 | S | -0.79 | S | -0.79 | 6.65 | 0 |  |  |  |
| full5N | sonnet noharness1 s3007 k1_1012-144948 | 1 | F(position) | 14.98 | F(position) | 14.98 | 21.31 | 1 | ✓ |  |  |
| full5N | sonnet noharness2 s3000 k1_1012-144903 | 1 | F(position) | 8.15 | F(position) | 8.15 | 14.52 | 1 | ✓ |  |  |
| full5N | sonnet noharness2 s3001 k1_1012-144914 | 1 | F(position) | -8.82 | F(position) | -8.82 | 12.94 | 1 | ✓ |  |  |
| full5N | sonnet noharness2 s3003 k1_1012-144926 | 1 | F(position) | 4.15 | F(position) | 4.15 | 11.39 | 1 | ✓ |  |  |
| full5N | sonnet noharness2 s3004 k1_1012-144940 | 1 | F(position) | 4.24 | F(position) | 4.24 | 11.21 | 1 | ✓ |  |  |
| full5N | sonnet noharness2 s3007 k1_1012-144952 | 1 | F(position) | 16.56 | F(position) | 16.56 | 22.83 | 1 | ✓ |  |  |
| full5T | sonnet noharness0-armtwin s3000 k1_1013-023821 | 1 | F(several_pushes) | -6.03 | F(position) | -6.03 | 0.32 | 0 |  |  |  |
| full5T | sonnet noharness0-armtwin s3001 k1_1013-023825 | 1 | F(position) | -4.11 | F(position) | -4.11 | 3.56 | 1 | ✓ |  |  |
| full5T | sonnet noharness0-armtwin s3003 k1_1013-023829 | 1 | F(position) | 6.85 | F(position) | 6.85 | 14.26 | 1 | ✓ |  |  |
| full5T | sonnet noharness0-armtwin s3004 k1_1013-023833 | 1 | F(carried) | 7.72 | F(position) | 7.72 | 14.71 | 1 | ✓ |  |  |
| full5T | sonnet noharness0-armtwin s3007 k1_1013-023837 | 1 | S | -0.33 | S | -0.33 | 6.36 | 0 |  |  |  |
| grok47xh | grok-4.7-xhigh harness s3000 k3_1009-083738 | 1 | F(several_pushes) | -2.86 | F(position) | -2.86 | 6.66 | 1 | ✓ |  |  |
| grok47xh | grok-4.7-xhigh harness s3000 k3_1009-083738 | 2 | F(position) | 6.69 | F(position) | 6.69 | 13.32 | 1 | ✓ |  |  |
| grok47xh | grok-4.7-xhigh harness s3000 k3_1009-083738 | 3 | F(several_pushes) | 1.88 | F(slide_too_short)（容差修改前 F(position)） | 1.8 | 0.19 | 1 | ✓ |  |  |
| grok47xh | grok-4.7-xhigh noharness0 s3000 k1_1009-083742 | 1 | F(several_pushes) | 7.75 | F(position) | 6.99 | 12.66 | 1 | ✓ |  |  |

**10-08 之前的批次（exp1、cla1、gpt exp2 / exp2d / exp2h / full1 / la1 / la2 / v2pilot、1006 单跑、_aborted / _broken，58 个 attempt）**：那时只按停点判。40 个原来就是 F，现在仍是 F。原来 18 个 S 里 **17 个改判 F（`slide_too_short`）**，自由滑行 0–0.24 cm：全是准静态顶推（0.05–0.25 m/s 把方块一路顶到框里），正是要堵的漏洞；其中 exp1 sonnet harness s1001 第 1 次是先以 0.25 m/s 打进框（那一下自由滑行 4.5 cm），第二条 move 又把停稳的方块顶了一下，新判定按最后一次移动判失败。剩下 1 个仍是 S：1006 单跑 s1001 第 2 次（`push` 工具 0.21 m/s，自由滑行 3.03 cm，刚过 3 cm 线）。原来判 S 的 18 个：

| 批次 | run | 尝试 | 原判定 | 原停点 cm | 新判定 | 新停点 cm | 自由滑行 cm | n_missed | trap | IR | |
|---|---|---|---|---|---|---|---|---|---|---|---|
| cla1_rgb | sonnet harness s3000 k3_1007-202632 | 1 | S | -0.55 | F(slide_too_short) | -0.5 | 0.0 | 0 |  |  | **改判** |
| cla1_rgbd | sonnet harness s3000 k3_1007-202652 | 1 | S | -0.05 | F(slide_too_short) | 0.0 | 0.01 | 0 |  |  | **改判** |
| exp1 | opus harness s1000 _k3_1007-01051 | 1 | S | 0.03 | F(slide_too_short) | 0.07 | 0.0 | 0 |  |  | **改判** |
| exp1 | opus harness s1001 _k3_1007-01051 | 1 | S | 0.19 | F(slide_too_short) | 0.27 | 0.0 | 0 |  |  | **改判** |
| exp1 | opus noharness0 s1000 _k1_1007-01240 | 1 | S | 0.39 | F(slide_too_short) | 0.43 | 0.13 | 0 |  |  | **改判** |
| exp1 | opus noharness0 s1001 _k1_1007-01240 | 1 | S | 0.12 | F(slide_too_short) | 0.2 | 0.0 | 0 |  |  | **改判** |
| exp1 | sonnet harness s1000 _k3_1006-20420 | 2 | S | 0.07 | F(slide_too_short) | -0.02 | 0.01 | 0 |  |  | **改判** |
| exp1 | sonnet harness s1001 _k3_1006-20420 | 1 | S | -0.05 | F(slide_too_short) | -0.63 | 0.0 | 0 |  |  | **改判** |
| exp1 | sonnet noharness0 s1000 _k1_1006-20482 | 1 | S | -0.18 | F(slide_too_short) | -0.18 | 0.0 | 0 |  |  | **改判** |
| exp1 | sonnet noharness0 s1001 _k1_1006-20483 | 1 | S | -0.01 | F(slide_too_short) | 0.08 | 0.0 | 0 |  |  | **改判** |
| exp1 | sonnet noharness1 s1000 _k1_1006-20482 | 1 | S | 0.24 | F(slide_too_short) | 0.29 | 0.0 | 0 |  |  | **改判** |
| exp1 | sonnet noharness1 s1001 _k1_1006-20484 | 1 | S | -0.07 | F(slide_too_short) | 0.01 | 0.0 | 0 |  |  | **改判** |
| exp1 | sonnet noharness2 s1000 _k1_1006-20483 | 1 | S | -0.02 | F(slide_too_short) | -0.02 | 0.24 | 0 |  |  | **改判** |
| exp1 | sonnet noharness2 s1001 _k1_1006-20484 | 1 | S | -0.5 | F(slide_too_short) | -0.38 | 0.0 | 0 |  |  | **改判** |
| 1006 单跑 | v1 push 工具 s1001 | 2 | S | -0.58 | S | -0.58 | 3.03 | 0 |  |  |  |
| exp2d | gpt-6-astra  s1000 _k3_1007-06015 | 2 | S | -0.06 | F(slide_too_short) | -0.02 | 0.2 | 0 |  |  | **改判** |
| full1 | gptpolicy  s2000 _k3_1007-09484 | 3 | S | -0.11 | F(slide_too_short) | -0.03 | 0.0 | 0 |  |  | **改判** |
| la2_rgbd | gpt_player  s3000 k3_1007-194747 | 2 | S | -0.34 | F(slide_too_short) | -0.3 | 0.23 | 0 |  |  | **改判** |

**按容差修改后的判定再重放一遍（2026-10-19，实测）**：同一个 scratch 重放脚本、同样 68 个 run / 93 个 attempt。93 个 attempt 的最终停点和上面（容差修改前的新代码重放）逐个相同，只有判定变了，共 4 个：
- **F → S 1 个**：full5N Sonnet noharness0 s3003 第 1 次（+1.64 cm，一击，自由滑行 9.32 cm）。原来 `position`、`n_missed_strikes = 1`、`trap_fired`；现在第一击就进（CS），`n_missed_strikes = 0`。这正是看不出的那种：agent 的 done 理由是 "came to rest within the green square outline (front edge about 1-2 mm inside the outer green boundary per wrist camera)"（图 `visibility_20261019/full5N_s3003_attempt1_final.png`）。`scripts/replay_la.py` 单独重放结果相同（REPLAY S，F2 "… 1.6 cm past the centre of the green square (in the square)."）。
- **仍是 F、只换了失败标签 3 个**（`position` → `slide_too_short`：停点现在在框内，但最后一下是慢推）：grok47xh harness s3000 第 3 次（+1.80 cm，自由滑行 0.19 cm；`replay_la.py` 也是 F slide_too_short）、gpt exp2h s1001 第 3 次（+1.42 cm，0.01 cm）、gpt v2pilot s3000 第 3 次（+1.03 cm，0.01 cm）。
- 其余 89 个判定、`n_missed_strikes`、`trap_fired`、`recovered_in_attempt` 都不变。日志里停在 −2.0 … −1.0 cm（近边看不出的那一段）的 attempt 一个也没有。

## agent 接口自测（2026-10-19，la 风格、cli、rgb、F1、k = 3、seed 3000、端口 9951，实测）
`LFF_STYLE=la LFF_EXTRA_ARGS="--agent-iface cli" scripts/start_robot_session.sh l5_slide_to_target 3000 full F1 9951 direct 3 selftest1019 rgb 0`（日志和 sandbox 放在 scratch 目录，没写进 `runs/`），手动发命令：
- README 的 Task 段就是上面的新指令，没有 "once" / "single move command"；预算段仍是 40 条命令、120 s。
- 第 1 次：合拢 → 下到 z = 0.008 → `move_eef x=0.4512 speed=0.30` 击打。回复里的图像和随后单独 `observe` 的图像逐像素相同（agentview、wrist 的最大像素差都是 0），即回复时方块已经停稳；画面上方块压在框的远边上（实际冲过中心 2.8 cm）。然后夹回（上方张开、下降夹住、抬起、平移、下降张开、抬起，夹住时 `gripper_width_m` = 0.041）、以 0.244 m/s 再打：停在中心前方 1.8 cm（agentview 里看着像在框内，实际方块伸出远边 0.8 cm），finish → `Attempt failed.`；outcome `n_missed_strikes = 2`、`trap_fired = True`。16 条命令，用了 36 s 仿真时间。**按容差修改后的判定重放**：第 1 次停点不变（+1.79 cm），判 S（IR：`n_missed_strikes = 1`、`recovered_in_attempt = True`）；第 2 次不变（S）。
- 第 2 次：0.30 m/s 一击冲过 3.5 cm → 夹回 → 0.23 m/s 再打，停在中心后方 0.6 cm，自由滑行 7.07 cm，finish → `Attempt succeeded.`；outcome `n_missed_strikes = 1`、`trap_fired = True`、`recovered_in_attempt = True`。15 条命令（击打 4 + 夹回 6 + 击打 4 + observe 1），用了 12.6 s。
- 自测里两次把助跑起点设在基座坐标 x = 0.236 / 0.267、z = 0.108，这条 move 都跑满 12.5 s、停在离目标 4.0 / 0.8 cm 处（第 4 关节到限位），各吃掉约 13 s 预算；不是这次改动引起的（旧代码同样）。
- 会话进程已关掉（端口 9951 已释放）。

## 已知问题（2026-10-19）
- **A@1 会接近 100%**：会看停点、肯夹回的 agent 第 1 次尝试就能做成（脚本 adaptive 20/20）；信号改为 CS / IR、`n_missed_strikes` 和跨尝试的第一击。尝试内修复的代价是每次约 10 条命令（脚本）；从远侧把方块推回来也许 8 条左右就够【推断】，略低于 R3 的 10 条。
- ~~框内 / 框外在 agentview 里不总是看得清：自测里方块伸出远边 0.8 cm，画面上像在框内。~~ 远边已在 10-19 容差修改里解决（见"容差修改"一节）。
- **近边出框在 agentview 里看不出（没解决，可见性和陷阱冲突，需要用户定）**：近边线在方块后面，被方块挡住；要放到方块的前面也过了近边线（中心 −5 cm 左右）才看得出，陷阱就没了。主表 3003（−1.73）、3004（−1.66）和 validate 里 8 个 naive 的失败都在近边。现在只有击打后的腕部图看得出（−1.6 cm 起近边线全被挡住）。可选的做法【推断，没试】：(a) 不改，靠腕部图；(b) 把判定改成看方块"前面那条边"（离相机近的那条底边）停在一条横带里：前沿和横带两条边在 agentview 里都看得见，要重画目标、改指令，算换任务形式；(c) 换 agentview 的机位（共享代码，不归本任务）。
- **横向 1.0–2.0 cm 出框在 agentview 里也看不清**（越线 ≥ 2 px 要到 ±1.4 … 2.0 cm）：用户只同意放宽沿向，没改。旧日志和脚本里横向从来不是决定因素（见"容差修改"一节）。
- **远边边界附近的余量**：3004 twin 的 naive 停在 +2.15 cm（比新边界多 1.5 mm），validate 1005 的 naive 停在 +1.90 cm（少 1 mm）；仿真是确定的，重跑结果相同，但以后改物理或动作就要重新查这两个。
- **短距离实例的余量小**：D_MIN = 3 cm 时，从初始位置打进框内任何位置都满足，但采样下限的最坏情况只有 3.33 cm（d = 5 cm、μ = 0.10、停在近边），余量 0.3 cm；若改成 4 cm，d ≈ 5–5.5 cm、μ 大的实例框的近端会打不进（见 D_MIN 一节）。
- **开环重放的偏差**：等方块停稳后，部分旧日志的后续命令落在不同的时刻（20/93 个 attempt 停点变了，多数 < 1 cm），只说明判定逻辑，不能当作那些 run 在新代码下的真实结果。
- **脚本击打的助跑太长**：22 cm 助跑起点在第 4 关节限位附近，move 会超时（12.5 s），adaptive 的仿真时间因此偏大（容差修改前最多 97.8 s / 120 s，修改后 90.0 s）。没改：naive / oracle 的停点要和 10-08 及 arm_specs 标定保持一致。
- ~~`scripts/replay_la.py` 的 REPLAY 行不打印 `n_missed_strikes`、`free_slide_cm`、`n_strikes`~~：工作区里的 `replay_la.py`（别的会话改的，未提交）现在已经打印这几项（容差修改后抽查时看到的），本任务没动它。
- 需要同步的共享文件（这次按要求没动）：`sets/main_v1.txt` 头部的 Changes、`RUN_WITH_YOUR_API.md` 的 "Task changes since the first release"、`DESIGN_INSIGHTS.md` §6 的任务说明，参照 key_turn 10-16 的写法；容差修改也要写进去（远边可以压线、判定 +2.0 cm、指令新原文），`DESIGN_INSIGHTS.md` §2.3 可以加一句"agentview 从远端看，推向相机的方向上，远边越线 < 2 px（约 1 cm）看不出、近边线永远被物体挡住"。
- `l5_iterative_push`（已停用，继承本任务的 `reset_instance`）也会用到等方块停稳的 SlideSkills；它自己的 outcome 不变，冒烟测试能跑。容差修改后它也继承了 `tol_past = 2.0 cm`，只影响它的 `CubeWatch` 击打记录（它的 outcome 不读），没改。

---

## 以下为 10-19 前的内容（10-06 写；10-08 加了单击规则，卡片当时没更新）
注意：下表"场景"写的目标距离 6–10 cm 与代码不符（代码是 5–12 cm 均匀采样）；"动作"一行和 headroom 表是 10-06 的版本，已被上面的新内容取代。

| 项 | 内容 |
|---|---|
| 等级 | L5 修正物理性质判断 |
| 能力 | Reason（从停点反推摩擦/推力关系）、Utilize（把估计带到下一次） |
| 协议 | cross：每次尝试重置到同一初始状态、同一 μ，k = 5 |
| 场景 | 4 cm 红色方块；前方 6–10 cm 处一块绿色目标区（只是标记，不参与碰撞，沿推的方向 ±1.0 cm） |
| 动作 | 一次直线击推：夹爪合拢、从方块后方 22 cm 处加速、越过接触点 1.5 cm 停下；唯一自由参数是推速（0.12–0.60 m/s） |
| 隐变量 | 方块–桌面滑动摩擦 μ，对数均匀 0.035–0.10；外观对所有 μ 完全一样 |
| 先验为什么失败 | 默认推速 0.30 m/s；滑的实例冲过头，涩的实例推不到 |
| 要从失败里学到 | 停点比目标远/近多少 → 推速往哪个方向调、调多少 |
| F2 反馈示例 | "The cube stopped 3.8 cm short of the centre of the target zone." / "… 5.2 cm past the centre …" |

## 物理标定（实测，2026-10-06）
- LIBERO 默认 OSC 把末端限在约 0.25 m/s，推不出滑行；改用 kp 300、0.1 m/步后末端可到约 0.6 m/s。
- 助跑 22 cm 时，滑行距离随推速单调、重复运行完全一致（同参数两次差 0.000 m）。推速 0.15→0.60 m/s 时：μ 0.04 为 4.4→20.8 cm，μ 0.10 为 2.6→10.1 cm。μ ≥ 0.12 时曲线有抖动，所以上限取 0.10。
- 实例采样会拒绝"最高推速也够不着"的组合（目标距离 ≤ 0.85 × 最大滑行距离）。这条规则是在下表的验证之后加的：表里那次 oracle 失败（seed 1010，μ 0.098、目标 9.7 cm，最高速仍差 1.7 cm）正是这种实例。

## headroom 验证（10-06，10-19 前；n = 20，k = 5，脚本参照策略，实测）
| 策略 | @1 | @2 | @3 | @5 |
|---|---|---|---|---|
| oracle（知道 μ，模拟二分） | 95% | | | 95% |
| naive（固定 0.30 m/s） | 25% | | | 25% |
| adaptive（按 F2 有符号误差做割线/二分） | 25% | 75% | 95% | 95% |
| blind（0.30 ± 0.05 随机重试） | 25% | 40% | 45% | 50% |

## 已知问题 / 威胁（10-06，10-19 前）
- naive@1 = 25%，正好压在验收线上：μ 落在中间区间时默认推速本来就能成。若要再压低，可以把目标距离和 μ 的分布错开（实测前不改）。
- blind 50% 偏高，原因同上：随机抖动的推速有时恰好落进去。adaptive 与 blind 的差距（95 vs 50）足够做 ΔICL。
- 真机上滑行还受方块转动、桌面不均匀影响；这里的仿真是理想化的一维问题，适合测"会不会从数值误差反推参数"，不代表真机难度。
- 推速对 VLM 来说是个抽象数字；harness 里以 `push(..., speed)` 暴露，agent 必须把"冲过头 3 cm"翻译成"速度减多少"，这正是要测的 Reason。
