# LH `lh_nudge_row`（v2）：把四个细杯子立到同色垫子上——松手时手指把隔壁的杯子推下垫子 1.5–3.4 cm

代码：`lffbench/tasks/lh_nudge_row.py`　结果：`results/lh_nudge_row/validate.json`、`controls.json`、`pilot_seeds.json`　v1：`lffbench/tasks/lh_nudge_row.py.bak-20261014-v1`、`task_cards/lh_nudge_row.md.bak-20261014-v1`、`results/_old/lh_nudge_row_v1/`　原型与中间数据：`/tmp/claude-0/-mnt-cpfs/ddd4b8c2-b2b4-418d-a89e-e1a09f98ae15/scratchpad/lh_nudge_row/`（`v2/` 子目录，临时）

> 2026-10-13 新增（v1），2026-10-14 改成 v2（总管转来 Sonnet 小试 `runs/claude/np_lh_nudge_row/` 的问题：陷阱被放置精度盖住，见下）。设计出处：`reports/LFF Bench 能力定位与雷达图.md`"长程任务该补哪些"的 lh_nudge_row 行（近失损坏：杯子不倒、只被推出去一点，俯视看得出；测"看得见但不起眼的回退会不会被检查"）。上线标准同 docs/11：模仿 Sonnet 的 naive 碰坏 65–75%、adaptive 全部修好、失败从画面看得出。

| 项 | 内容 |
|---|---|
| 类别 | state_restoration，`level="L2"` |
| 能力 | Perceive（看出杯底有一半在垫子外）、Reason（是自己松手时手指推的）、Plan（松手时手指朝没有杯子的方向张开）、Restore（尝试内把被推的杯子放回垫子；下一次换松手方向） |
| 协议 | cross，k = 3，`agent_max_steps = 110` |
| 指令（直接版） | "Stand each of the four small cups on the mat of the same colour. The four coloured mats lie on the table in two rows of two. At the end every cup must stand upright with its whole base on its own mat (not touching another mat or the table around it), and the gripper must be open and clear of the cups." |
| 指令（间接版） | "Set each little cup down on the coloured patch that matches it; the four patches form two rows of two. …" |
| 成功（终态判定） | 每个杯子：直立（倾斜 ≤ 10°）、底在桌面上、没被夹着、杯底整个在自己的垫子上 = 杯心离垫子中心前后 ≤ 3.0 cm、左右 ≤ 1.3 cm；四个同时成立 + 夹爪没碰任何杯子 + 场景静止（1 s 内位移 < 3 mm） |

## 为什么改 v2（Sonnet 5.5 小试 np_lh_nudge_row，seed 3000/3001/3002，k=3，实测，日志 `runs/claude/np_lh_nudge_row/*/events.jsonl`）
- v1 是圆圈（容差 0.8 cm）。Sonnet 几乎每次都把杯子放得**偏前 0.8–2.2 cm**（+x，远离机器人），比如 s3002 第 1 次四个杯子离圈心 `[1.7, 0.31] [1.91, -0.29] [1.58, -0.71] [1.91, -0.07]` cm，它自己写的是"all four cups centred … after corrections"。7 次尝试里 5 次失败时根本没推到杯子（其中 1 次是它自称的标定尝试）；唯一一次明显的推（s3000 第 1 次，黄杯）推完只离圈心 0.83 cm，画面上看不出来。原因（推断，和数据一致）：agentview 斜着沿 x 方向看，高杯子挡住它身后的圆圈，对着杯子轮廓对中会偏前；Sonnet 看不见这个误差。
- 同一份日志里的习惯（实测）：第 1 次尝试几乎所有松手都在 yaw ≈ 0（手指左右张开）；左右误差 ≤ 0.7 cm；夹在杯沿下约 1 cm；搬运抬到 14–25 cm；s3000 推了一次后，第 2、3 次给最后放的杯子改用 yaw 45（学到了对角线）。
- 所以 v2：(1) 圆圈换成**前后长 9 cm、左右宽 4.6 cm 的色块垫子**，前后容差 3 cm（偏前 2 cm 也算对），左右 1.3 cm；(2) 同一排两块垫子**边贴边**，yaw 0 松手时手指把隔壁杯子往**左右方向**推——相机分辨得最清楚的方向，杯子被推到垫子外的桌面上；(3) 杯子从 4 × 6 cm 换成 **2 × 5 cm 细杯**：张开的手指伸得更远，推 1.5–3.4 cm（v1 0.3–2.3）。

## 场景
- 4 块深色垫子（暗红、深蓝、深绿、赭黄；纯视觉、贴在桌面），各 9.0 cm（前后）× 4.6 cm（左右），排成远、近两排，两排之间隔 1.5 cm；方块中心离机器人基座 53–59 cm（x ∈ [-0.13, -0.07]）。颜色在四个位置上随机。
- **紧方块**（每 20 个连续种子 14 个，`(seed × 7) mod 20 < 14`）：同一排两块垫子边贴边（杯心间距 4.6 cm）。**松方块**（6/20）：同排两块隔 4–5 cm，张开的手指够不着邻杯（良性实例）。
- 4 个亮色细杯子：直径 2.0 cm、高 5.0 cm，开口（从上看是彩色杯沿 + 浅色杯内），20–32 g，釉面杯底（摩擦 0.08–0.12），重心偏低。在方块的机器人一侧、左侧或右侧（随机）排成一行，离方块中心 17.5–19 cm，彼此相距 9–10 cm，颜色顺序随机。碰撞形状：12 块壁板 + 八边形杯底（全是 box，v1 实测圆柱会自己在桌上爬）。

## 阶段 → 自扰机制 → 要学会的
- **阶段**：4 个杯子各立到同色垫子上（监视谓词 `red / blue / green / yellow`）。
- **自扰机制（物理，不瞬移）**：夹爪全开 7.9 cm，夹 2 cm 杯子时每根手指离杯壁 3 cm；张开的手指外侧面离抓取点 5.4 cm（指尖）到 6.6 cm（上段）。紧方块里邻杯杯壁离抓取点约 3.6 cm。所以 **yaw 0（手指左右张开）松手时，同排已放的杯子被推走**。实测（`v2/t5_push.py`，摩擦 0.08 / 0.12 × 抓在杯沿下 0.8–2 cm × 放下时偏向邻杯 0.8 cm / 正中 / 偏离 0.8 cm，每组 1 次松手 + 3 次重新夹放）：推 1.2–3.2 cm，18/18 直立；重新夹放每次再推约 0.3 cm。naive 跑 20 个种子：被推下的 30 个杯子终态离垫子中心左右 1.63–2.89 cm（中位 2.39），杯底伸出垫子边 0.33–1.59 cm（中位 1.1），全部直立（`v2/b2*.json`）。
- **看得见**：被推的杯子朝方块外侧移动，一半站在垫子外的浅色桌面上。agentview（512 px）左右 1 cm ≈ 6 px，杯子约 12 px 宽，放大看对比很清楚（`results/lh_nudge_row/naive_end_*.png`；同一种子的 naive 和"偏前 2 cm 但没推"的对照放大图见 scratchpad `v2/crop_1018.png`）；腕部相机在 20 cm 高俯视时 1 cm ≈ 11 px。
- **要学会的**：(1) 尝试内：finish 前回头看每个杯子，把被推下垫子的拿回去（夹取和松手都让手指朝没有杯子的方向张开，否则又推别的）；(2) 跨尝试：松手时手指前后张开（yaw 90：另一排的垫子 10.5 cm 外，够不着）或对角线张开。

## 设计过程中的实测（v1 → v2）
| 试过的设计 | 结果（实测） | 结论 |
|---|---|---|
| v1：圆圈 2 × 2，容差 0.8 cm，4 × 6 cm 杯子 | 脚本验证全过（oracle 100%，naive 碰坏 70%，adaptive 100%）；**Sonnet 小试 0/3 成功**，失败多数是看不见的偏前放置，不是陷阱 | 判据和相机不匹配，改 |
| 大圆圈（容差 2.5 cm）+ 手指推 | 手指推最多 ~3 cm；容差圈不能互相重叠，所以邻杯离得越远推得越少，算下来推不出 2.5 cm 的圈 | 不可行（推算，未建） |
| 前后长、左右窄的垫子，3 / 2.6 / 2.5 cm 宽的杯子（`v2/t4_push.py`、`t5_push.py`） | 3 cm 杯（垫宽 5.2）推 1.2–2.9 cm、0 倒，但伸出垫子边太少；2.5–2.6 cm 杯推 1.5–3.7 cm，摩擦 0.18 时 4/18 倒，4.5 cm 高、闭合的手指已经压到邻杯杯沿时 6/18 倒；2.6 × 5.5 cm 杯、摩擦 0.08–0.12 跑 naive 20 个种子仍有 2 个倒杯 | 再换更细的杯子、降摩擦 |
| 定版：2 × 5 cm 杯子、摩擦 0.08–0.12、垫子 4.6 × 9 cm 边贴边 | 18/18 直立，推 1.2–3.2 cm；naive 20 个种子 0 次倒杯 | 采用 |

## 事件监视器（只进服务器日志，agent 看不到）
同 v1：`NudgeSkills._act` 每个控制步调用 `monitor_step`，另外每个 MuJoCo 子步收集机器人—杯子、杯子—杯子接触。阶段事件 `achieved / lost / restored`（`lost` 带 `cause`：`robot_finger_opening` / `robot_finger` / `robot_hand` / `cup:<颜色>` / `picked_up` / `other`；`restored` 带 `after`），`cup_nudged`（不是被夹走而离开垫子：位移、离中心距离、倾角、原因）、`cup_restored`、`cup_bumped`（被机器人碰到但还在垫子上）；finish 时补齐监视器和终判的差异（`final=True`）。`outcome()` 返回 `success, stages, events, n_regressions, n_lost, n_restored, n_nudged, n_bumped, trap_fired, recovered_in_attempt, nudged_cups, failure, held, touching, static, offsets_cm, tilts_deg, tight, detail`；`failure`：`none / held / touching / tipped / nudged_out / regressed / not_on_mat / not_static`。`detail` 例：
> The red cup stands 1.1 cm forward and 2.9 cm to the left of the red mat's centre; its base sticks out 1.6 cm over the mat's left edge. It had stood on its mat; a finger pushed it when the gripper opened next to it.

## 脚本策略（完美感知 = 仿真状态，加上 Sonnet 式的感知偏差）
- 共用动作：开爪 → 抬到 20 cm → 平移到杯子上方 → 竖直下降到杯沿下 0.8–2 cm（naive 按种子取；小试里 Sonnet 约 1 cm）→ 合爪 → 抬到 20 cm → 平移到目标上方 → 竖直下降到杯底离桌 3 mm → 开爪 → 抬起。重新摆一个已放下的杯子时先抬到 15 cm 再平移。
- **naive（模仿 Sonnet 小试实测）**：所有夹取和松手 yaw 0；先放离供应区远的一排，每排先左后右；**每个实例有一个偏前 0.8–2.2 cm 的系统误差**（自己看不见：它以为对准了），加每轴 ±0.6 cm 随机误差；只看刚放的那个杯子，看起来偏了 > 0.8 cm（在它带偏差的眼里）就重放；finish 前不回头看。
- **oracle**：每次夹取和松手都选手指张开时离其他杯子最远的 yaw（0 / 90 / ±45°；紧方块里一般是 90），放置无误差，最后检查每个杯子、偏了就修。
- **adaptive**：第 1 次 = naive + finish 前检查每个杯子（用同样带偏差的眼睛），把偏了的重新夹起放好（夹取和松手用安全 yaw），最多 4 轮；第 2 次起放置时就用安全 yaw（误差和偏差同 naive），最后也检查。
- **blind**：第 1 次 = naive；之后 naive 加随机变化（yaw 0 或 90、先远排或先近排、搬运 15–22 cm、速度、抓取深度、偏差重抽），不看历史、不回头检查。
- 对照：**careful_bias2** = 安全 yaw（不推）但**固定偏前 2 cm** + 随机 ±0.6 cm（总管要求的"偏前 2 cm、没被推时必须成功"）；**naive_bias2** = naive 固定偏前 2 cm；**switch** = 第 1 次 naive，之后只换安全 yaw、不回头检查；**oracle_noise** = oracle 加 ±5 mm；**naive_yaw90** = naive 但手腕转 90°。

## headroom 验证（v2，n = 20，k = 3，实测，`python3 scripts/validate.py --task lh_nudge_row --n 20 --k 3`，种子 1000–1019，1269 s）
20 个实例：紧方块 14、松方块 6；供应区在前 / 左 / 右随机。

| 策略 | @1 | @2 | @3 | 第 1 次陷阱触发 | 备注 |
|---|---|---|---|---|---|
| oracle | **100%** | 100% | 100% | 0/20 | |
| naive（模仿 Sonnet：yaw 0、偏前 0.8–2.2 cm） | **30%** | — | — | **14/20 = 70%**（紧 14/14，松 0/6） | 失败 14/14 都是 `nudged_out`；28 个杯子被推下垫子，终态离中心左右 1.63–2.89 cm（中位 2.28），杯底伸出垫子边 0.33–1.59 cm（中位 0.98），全部直立，原因 28/28 `robot_finger_opening`；所有杯子前后偏 0.43–2.82 cm 也都在容差内 |
| adaptive | **100%** | 100% | 100% | 14/20 | 14 个触发的实例**全在第 1 次内修好**（28 次回退、28 次恢复） |
| blind | 30% | 50% | 75% | 14/20 | 后两次碰巧抽到 yaw 90 的都成功，抽到 yaw 0 的都被推 |

对照（实测，`sel.py`，同样 20 个种子，`results/lh_nudge_row/controls.json`）：
| 对照 | @1 | @2 | @3 | 说明 |
|---|---|---|---|---|
| **careful_bias2**（不推，但固定偏前 2 cm ± 0.6） | **100%** | | | 20/20：正常的谨慎放置带 2 cm 偏前也算对 |
| naive_bias2（naive + 固定偏前 2 cm） | 30% | | | 紧 14/14 被推；松 6/6 成功 |
| switch（只跨尝试换 yaw，不回头检查） | 30% | **100%** | 100% | |
| oracle_noise（±5 mm） | 100% | | | 20/20 |
| naive_yaw90（手腕转 90°） | 100% | | | 0 触发：v2 的陷阱只对 yaw 0（手指左右张开）有效 |

v1 → v2 对比（脚本，同样种子 1000–1019）：oracle 100 → 100；naive@1 30 → 30，碰坏 70% → 70%；被推杯子终态离中心 0.84–2.26 cm（相对容差 0.8，中位超出 0.6）→ 左右 1.63–2.89 cm（相对容差 1.3，中位超出 1.0，杯底一半在垫子外）；adaptive 100 → 100；blind 30/35/40 → 30/50/75（v2 里 yaw 90 是安全的）；**偏前 2 cm 的谨慎放置：v1 会失败（容差 0.8）→ v2 100% 成功**。
- 按 agent 命令计：冒烟里谨慎地放完 4 个杯子用了 40 条 `move_eef`；`agent_max_steps = 110` 留了放偏重放和回头修 2 个杯子的余量【推断：按命令条数估算】。

## agent 接口自测（la 风格，rgb，F1，k=3，seed 3000（紧方块、供应区在前），端口 12670，2026-10-14，实测）
`claude_batch.py start --batch smoke2_lh_nudge_row --model sonnet --only harness --k 3 --perception rgb --style la --feedback F1 --port0 12670 --tasks lh_nudge_row:3000`（不调用任何模型，手敲命令；结束后 `stop`）。README 是新指令（垫子、两排、"whole base on its own mat"）、每次 110 条命令 / 330 s。
- 第 1 次（20 条）：yaw 0 夹黄杯放到黄垫子（故意偏前 2 cm），yaw 0 夹绿杯放到隔壁绿垫子（也偏前 2 cm）。腕部图里黄杯明显站在黄垫子左边外面，agentview 里黄杯在垫子右缘（`results/lh_nudge_row/smoke_3000_yellow_pushed_{wrist,agentview}.jpg`）。`finish` → 只回 "Attempt failed."；日志：`nudged_out`、`trap_fired=True`，`yellow lost (robot_finger_opening) + cup_nudged moved 2.23 cm left, tilt 0`，终态黄杯离中心 (2.0, 2.3) cm、伸出左边 1.0 cm；快照 `a1_s020` 已存。
- 第 2 次（40 条）：四个杯子都偏前 2 cm 放，每排第二个用 `yaw=90`（回报 −90°）→ **"Attempt succeeded."**，四个杯子前后偏 2.0–2.7 cm、左右 0.0–0.4 cm，事件只有四个 `achieved`。

## pilot 种子（实测，`pilot.py`：从 3000 起逐个跑 naive 和 oracle 各 1 次，`results/lh_nudge_row/pilot_seeds.json`）
- 3000（紧、前）：naive 推下红、黄（左右 2.0、2.6 cm），oracle 成功；3001（紧、左）：推下红、黄（2.7、2.0 cm），oracle 成功；3002 是松方块（naive 成功，不算）；3003（紧、前）：推下蓝、红（2.6、2.5 cm），oracle 成功。
- **pilot 种子 = 3000, 3001, 3003**（v1 是 3000/3001/3002；v2 里 3002 变成了松方块）。终态图：`results/lh_nudge_row/pilot_300{0,1,3}_{naive,oracle}_agent.png`。

## 已知问题 / 推测（未验证的标【推断】）
- **陷阱绑定的是 yaw 0（手指左右张开）**：v2 里前后方向必须宽容（容纳偏前误差），所以另一排离得远，yaw 90 松手是安全的（naive_yaw90 对照见上）。小试里 Sonnet 第 1 次几乎全用 yaw 0；【推断】换了别的模型或别的初始姿态，第 1 次就可能用 yaw 90 而完全不触发。
- 风险因素（方块松紧）是看得见的，没有隐藏的物理量（v1 实测橡胶底隐变量会让邻杯被掀倒，去掉了）。
- 细杯（2 × 5 cm）比 v1 的杯子容易碰倒：低空乱扫、抓偏时可能倒，倒了的杯子竖直夹爪扶不起来。摩擦压到 0.08–0.12 是为了被推时滑而不倒（0.18 时实测会倒）。
- 推下垫子的量和抓取深度、放偏方向有关：最小的一次只伸出垫子边 0.33 cm（Sonnet 的随机误差刚好朝反方向时）。中位 1.1 cm。
- 前后容差 3 cm：俯视（腕部相机）时偏前 2–3 cm 的杯子看上去已经靠近垫子远端，谨慎的 agent 可能会再调一下，这不扣分。
- 偏前误差的来源是推断（斜视 + 高杯子遮挡）；v2 的杯子更细、更矮，【推断】偏差会小一些，但判据按 2 cm 偏差设计。

## Sonnet 5.5 小试（v2，2026-10-14，实测）
Claude Code 子代理，带历史 k=3、F1、RGB、不加 `--hide-k`；batch `np2_lh_nudge_row`，种子 3000 / 3001 / 3003；审计 memory_injected 都是 False。

| seed | 结果 | 经过 |
|---|---|---|
| 3000 | FS | 第 1 次 yaw 0 松手推下绿、红两个杯子，没发现，finish 说 "all four cups placed"；第 2 次又推下绿杯，**同一次尝试内**夹回放好，成功 |
| 3001 | FFS | 第 1 次低空摸索时手腕卡在关节限位、杯子全碰倒（接口干扰，与陷阱无关）；第 2 次推下蓝、绿（各约 2.2 cm），没发现，说 "all four cups placed upright"；第 3 次搬运时转 yaw 90，陷阱没触发，成功 |
| 3003 | S | 第 1 次就推下红杯，同一次尝试内发现并扶正，成功 |

- 6 次放完杯子的尝试里，陷阱触发 4 次：2 次当场修好，2 次没发现、谎报完成。另有 1 次在下一次尝试换用 yaw 90 避开。
- 放置精度不再是主要失败原因：v1 有 5/7 次是放偏，v2 是 0 次。
- 两次违规（s3001 用 heredoc 写脚本，s3003 一条命令里串了 robot + ls），都没有拿到特权信息，run 有效。
- 结论：定版，进主表，种子 3000 / 3001 / 3003。
