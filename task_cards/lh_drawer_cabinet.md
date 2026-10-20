# LH `lh_drawer_cabinet`：收拾桌子——关抽屉时自闭弹簧把抽屉"啪"地拽上，轻柜子猛地一晃，柜顶的瓶子倒了

代码：`lffbench/tasks/lh_drawer_cabinet.py`　结果：`results/lh_drawer_cabinet/validate.json`、`controls.json`、`robustness.json`、`pilot_seeds.json`　原型与中间数据：`/tmp/claude-0/-mnt-cpfs/2aa0daff-4c37-4ec7-b3e7-10b838c28688/scratchpad/lh_drawer_cabinet/`（临时；`t2_slam.py` 物理原型、`t8_cubemap.py` 方块位置可行性、`t14_robust.py` 鲁棒性扫描）

> 2026-10-16 新增。设计出处：总管的 lh_drawer_cabinet 设计意图（"收拾桌子"长程任务，后面的正常动作——关抽屉——把前面做好的阶段——柜顶的瓶子——震倒）。上线标准同 docs/11：模仿 Sonnet 习惯的 naive 碰坏约 65–75%，adaptive 能修好，失败从画面看得出。
> 和已有任务的区别：lh_tray_delivery 是"搬运时托盘过门槛晃倒瓶子"（隐变量是瓶子重心）；lh_lid_box 是"开盖时盖子砸到后面的方块"（隐变量是盖子的止挡角）。这里是**关抽屉**这个动作通过**柜体**把冲击传给柜顶，隐变量在抽屉的闭合机构上，和放慢推的速度无关。

| 项 | 内容 |
|---|---|
| 类别 | state_restoration，`level="L2"` |
| 能力 | Perceive（看出瓶子倒在柜顶的围栏上）、Reason（是关抽屉那一下把柜子拽动了，不是手碰的，也不是推得太快）、Restore（抽屉关好以后捏住瓶盖把瓶子扶回柜顶）、Plan（下一次先放方块、关抽屉，再放瓶子） |
| 协议 | cross，k = 3，`agent_max_steps = 120` |
| 指令（直接版） | "Tidy up the table: stand the green bottle and the brown bottle upright on top of the small wooden cabinet, put the red cube and the blue cube into the cabinet's open drawer (it is pulled out towards the robot), and close the drawer. When you finish, both bottles must stand upright on the cabinet top, both cubes must be in the closed drawer, the cabinet must still stand where it is, and the gripper must be open and not touching anything. (Each bottle has a round cap; held by its cap, a bottle hangs straight down.)" |
| 指令（间接版） | "Clean up: set the two tall bottles (the green one and the brown one) upright on the little wooden cabinet, drop the two small cubes (red and blue) into its pulled-out drawer and shut the drawer. …" |
| 顺序 | 指令先列瓶子，但**不规定顺序**（没有 first / then）：先关抽屉再放瓶子是合法的预防 |
| 成功（终态判定） | 两个瓶子都直立（≤ 10°）、瓶底整个在柜顶上（围栏以内）、没被夹着；两个方块都在抽屉里（方块在抽屉内腔里、没被夹着）；抽屉关上（≤ 5 mm）；柜子直立、离原位 ≤ 8 cm（抽屉自己弹上会把柜子拽动 2.3–4.2 cm，机器人接着推又会推回去）；夹爪张开、没夹东西、不碰任何任务物体；场景静止 |

## 场景
- **小木柜**（自由刚体，不是固定件）：柜体 18（前后）× 22（左右）× 11 cm，正面朝向机器人，中心离机器人基座 53–56 cm（x ∈ [−0.13, −0.10]，y ∈ ±4 cm）。柜顶后沿有一道 6 cm 高的围栏（见"设计过程"）；底板向前多伸出 4 cm（踢脚），柜体重心偏后。
- **抽屉**：一个，滑轨（slide joint），朝机器人拉出 11 cm；内腔 15.9 × 16.2 cm，侧壁 3 cm；前面板 1.8 cm 厚、盖住柜体正面，面板上只画了一个暗色的扣手槽（没有能抓的把手）。抽屉 0.8 kg（实木抽屉），柜体 0.35–0.45 kg。agentview 相机从柜子背面看过来，抽屉被柜体挡住，只能用腕部相机看；柜顶和瓶子在 agentview 里看得很清楚。
- **两个高瓶子**（绿瓶黄盖、棕瓶蓝盖）：方瓶身 4 × 4 cm，总高 19.4 cm，0.18 kg，重心离瓶底 9 cm；瓶底是圆角玻璃底，平的接触面只有 2.2 × 2.2 cm，倾斜超过约 7° 就倒；瓶颈 5 cm，顶上是一个圆球瓶盖（直径 2.4 cm，扭转摩擦很小：从上面横着捏住躺倒瓶子的瓶盖往上提，瓶子会绕瓶盖转下来、竖直吊着——同 lh_bystander）。瓶颈有扭转摩擦，捏住瓶颈（瓶盖下方）提起时瓶子不晃。起始站在柜子一侧的桌面上（离柜子中线 21–24 cm）。
- **两个方块**（红、蓝，3.2 cm，23 g）：在柜子另一侧，离柜子中线 25–29 cm（更近的位置去夹时，前臂会扫过柜顶、碰倒已经放好的瓶子，见"设计过程"）。

## 隐变量、阶段、自扰机制、要学会的
- **隐变量（外观完全相同）**：抽屉的闭合机构。**自闭**（每 20 个连续种子 14 个，`(seed × 7) mod 20 < 14`）：最后 8 cm 行程里有 12.5–15 N 的弹簧往里拉、没有阻尼；**缓闭**（6/20）：同样的弹簧加阻尼器，抽屉以 4–6 cm/s 自己滑进去。干扰量（不影响类别）：柜体质量 0.35–0.45 kg、毛毡脚垫摩擦 0.15–0.22。
- **阶段（监视谓词）**：`green_bottle_on_top`、`brown_bottle_on_top`、`red_cube_in_drawer`、`blue_cube_in_drawer`、`drawer_closed`。
- **自扰机制（物理，不瞬移）**：弹簧是柜子内部的力。抽屉被推进最后 8 cm 时弹簧把抽屉往里拽，同时把轻柜体往抽屉方向（朝机器人）拽：柜体在约 0.07 s 里朝机器人窜 2.3–4.2 cm（实测，validate 里 naive 的 14 次关抽屉；抽屉相对柜体撞上止挡时 1.9–2.4 m/s），然后突然停住。站着的瓶子底部跟着柜顶走、上部跟不上，向**远离机器人**的方向倒，靠在柜顶后沿的围栏上（45–75°）。机器人的手在抽屉面板前面、比瓶子低、离瓶子 10 cm 以上，**倒的时候没碰瓶子**（实测：所有自闭实例 0 次）。缓闭抽屉柜体几乎不动（< 3 mm），瓶子不倒。
- **推慢了也没用**：关上的是弹簧，不是手；2–10 cm/s、推的高度 3–8 cm 都倒（实测，见鲁棒性）。抓住抽屉也没用（没有把手；即使有，按住抽屉时弹簧会把轻柜体拽向抽屉，【推断，按受力分析，未建模测】）。
- **看得见**：瓶子从站立变成倒向相机、靠在围栏上，agentview 里瓶子的高轮廓消失，只看见朝向相机的瓶盖和瓶肩（`results/lh_drawer_cabinet/naive_end_*.png`；对比成功的终态图里两个瓶子高高立着）。腕部相机从上面看是两个横躺的瓶子。
- **R4（同一失败外观的几种解释）**：最省事的解释"推得太快 / 太猛"是错的（放慢不管用）；"瓶子没放稳"也是错的（放得再准也倒）；"手 / 胳膊碰的"不成立（瓶子朝远离机器人的方向倒）。对的解释是"关抽屉这一下拽动了柜子"，对策是换顺序。
- **要学会的**：(1) 尝试内：关好抽屉以后回头看柜顶，把倒下的瓶子捏住瓶盖提起、吊直、放回柜顶；(2) 跨尝试：先放方块、关抽屉，最后放瓶子。
- **不可逆的地方**：抽屉一旦关上就打不开（没有把手；面板贴着柜体），所以在放进方块之前误关抽屉，这次尝试就做不成了。打开状态离弹簧区还有 3 cm 空行程，放方块时轻碰不会触发。

## 设计过程中的实测（原型脚本在 scratchpad）
| 试过的设计 | 结果（实测） | 结论 |
|---|---|---|
| 抽屉朝侧面开（沿 y 推） | 量了 Panda 手的碰撞网格：yaw 0 时手掌沿 y 方向伸出抓取点 ±10.5 cm，推到底时手掌伸进柜顶上方，正好扫到瓶子；沿 x 推时只伸出 +3.9 cm（手掌）/ +5.6 cm（link6，高度 13–35 cm） | 抽屉朝机器人开（沿 +x 推），代价是 agentview 看不到抽屉内部 |
| 0.35 kg 抽屉、0.6–2 kg 柜体、4 cm 弹簧区 4–16 N、桌面摩擦 0.5（当时的瓶子 21 cm 高、瓶底接触面 3.6 cm） | 抽屉撞止挡 0.4–1.8 m/s，瓶子最多倾斜 8°，**一个都没倒** | 冲击后柜子被摩擦刹住，前后两下互相抵消；瓶子的倾角大约跟着支撑面的**位移**走（θ ≈ m·h·Δx / I ≈ 7 rad/m），要快速移动 2.5–3 cm 以上才倒 |
| 2 kg 柜体 + 14 N 弹簧 | 弹簧拽时柜体被摩擦按住、撞止挡后柜子往远处滑，瓶子**朝机器人倒、砸在胳膊上**（42–54° 时碰到 link5–7） | 朝机器人倒混淆"手碰的"，不用 |
| 中等柜体质量 / 摩擦 | 上面两种效应互相抵消，瓶子晃一下（6–11°）又站回去；不单调 | 参数必须落在"轻柜体被拽向抽屉"那一侧 |
| **定版**：0.8 kg 抽屉、0.35–0.45 kg 柜体、脚垫 0.15–0.22、8 cm 弹簧区 12.5–15 N、瓶底接触面 2.2 cm | 柜体朝机器人窜 2.3–4.2 cm，瓶子全部朝远离机器人方向倒，0 次机器人接触 | 采用 |
| 瓶子从柜顶后面掉到桌上（没有围栏） | 瓶子翻个跟头，瓶盖落在柜子背面 0–4 cm 处；手掌伸出抓取点 3 cm，够瓶盖时撞柜顶后沿、把柜子推走 10 cm，扶瓶失败 | 加 6 cm 高的后围栏：瓶子靠在围栏上，瓶盖翘在 19–23 cm 高处，从上面就能捏住 |
| 开着的重抽屉 | 柜体加抽屉的重心离前沿只有 2.3 cm，往开着的抽屉前部按约 3 N 柜子就前倾（放方块时方块压到面板边上，瓶子朝机器人倒了，实测 1 次） | 底板向前伸 4 cm、柜体重心后移 4 cm：前倾要 ~14 N【按力矩估算】 |
| 方块离柜子中线 20 cm、在柜子前半边 | 去夹方块时前臂（link5–7）扫过柜顶，碰倒已经放好的瓶子（可行性图 `t8_cubemap.py`：12 个位置里 dy = 20 cm 的 2 个出事，25 cm 以外 0 个） | 方块放在 25–29 cm 处 |
| 三个 3.6 cm 方块在 11 cm 长的抽屉里排成一行 | 张开的手指 13.2 cm 宽，排成横行时手指压到侧壁或旁边的方块；离柜体太近时手掌撞柜体前沿 | 改成两个 3.2 cm 方块、沿抽屉中线前后排（脚本）；agent 放哪都行 |
| 抓瓶身 | 从上面够不到：手指只有 ~5 cm 长，手掌会压到瓶盖 | 抓瓶颈上端 / 瓶盖（手掌在两指之间离抓取点 3.1 cm，实测） |

## 鲁棒性（实测，`t14_robust.py`：把瓶子直接放到柜顶、方块放进抽屉，只做关抽屉这一步）
| 条件 | 自闭（种子 1000/1001/1003/1004） | 缓闭（1002/1005） |
|---|---|---|
| 瓶子位置：中间并排 / 靠前 / 靠后（贴近围栏 2 cm）/ 左右分开 / 只放一个 / 斜放转 40°，推 4 cm/s、高 5 cm | **46/46 个瓶子倒**，0 次机器人碰瓶子 | 0/22 倒 |
| 推的速度和高度：2 / 3 / 5 / 10 cm/s，高 3–8 cm（中间并排） | **32/32 倒**，0 次碰 | 0/16 倒 |
| 最快速（不给 speed，~0.6 m/s） | 5/8 倒（机器人把柜子往前顶，抵消了一部分窜动） | 4/4 倒（机器人自己把抽屉撞上、把柜子顶走 10 cm，瓶子朝机器人倒在胳膊上） |
最快速推不是 Sonnet 的习惯（它推 3–5 cm/s），但缓闭实例里猛推也会倒——这是"推太猛"的真实后果，和陷阱不同类，日志里原因记为 `robot_hand / robot_arm`。

## 事件监视器（只进服务器日志，agent 看不到）
`DrawerSkills._act` 每个控制步调 `monitor_step`；每个 MuJoCo 子步（包在 `sim.step` 外）施加抽屉弹簧 / 阻尼力（`qfrc_applied`，按当前状态算，可复现），并收集机器人—瓶子、瓶子—瓶子接触和抽屉撞止挡的速度。事件：
- 阶段 `achieved / lost / restored`（防抖 0.5 s；`lost` 带 `cause`：`drawer_slam` / `robot_hand` / `robot_arm` / `bottle` / `picked_up` / `other`；`restored` 带 `after`）；
- `drawer_slam`：抽屉以 > 0.3 m/s 撞上止挡（速度、柜体窜动 `cabinet_jerk_cm`、当时柜顶有哪些瓶子、之后因此倒了哪些、倒之前机器人有没有碰过它）；`drawer_shut`：慢慢关上；
- `bottle_fell`：倾斜超过 40°（原因、倒的方向、倒之前机器人有没有碰过）。
`outcome()` 返回 `success, failure, stages, events, n_regressions, n_lost, n_restored, trap_fired（柜顶的瓶子因抽屉弹上而丢失）, recovered_in_attempt, n_slams, slam_speeds, slam_jerk_cm, bottles_lost_to_slam, robot_touched_before_fall, bottle_max_tilt_deg, n_fallen, held, touching, static, gripper_open, drawer_q_cm, cabinet_moved_cm, cabinet_tilt_deg, bottle_tilts_deg, self_closing, detail`。`failure`：`none / held / cabinet_moved / bottle_knocked（站上去过又倒了）/ bottle_not_on_top / cube_not_in_drawer / drawer_open / touching / gripper_closed / not_static`。
F2 `detail` 例（agent 接口自测 seed 3000 第 2 次，原文）：
> The green bottle has tipped over forward on the cabinet top and leans 58 degrees against the rail along the top's far edge; its base is 2.1 cm backward and 6.7 cm to the right of the centre of the cabinet top. It had stood on the cabinet top; it fell when the drawer snapped shut and the cabinet jerked. The brown bottle has tipped over forward on the cabinet top and leans 55 degrees against the rail along the top's far edge; its base is 1.9 cm backward and 7.0 cm to the left of the centre of the cabinet top. It had stood on the cabinet top; it fell when the drawer snapped shut and the cabinet jerked.

## 脚本策略（完美感知，只用 move_to + 开合爪）
- 共用动作：搬运时先抬高（越过路上的东西：柜顶 + 围栏、站着的瓶子），平移，竖直下降。瓶子：捏瓶颈上端，抬到瓶底高出柜顶 5 cm 以上，竖直放下，离柜顶 1–4 cm 松手。方块：夹起、移到抽屉中线上、离抽屉底 2 cm 松手。关抽屉：合爪，降到面板前 3 cm、高 4–6 cm，朝前直推到面板和柜体齐平。扶瓶：从上面横着捏住瓶盖，提到瓶底高出围栏 3 cm，挪到柜顶空位（路径避开另一个瓶子），放到离柜顶 3 mm 松手。
- **naive（模仿 Sonnet 的习惯）**：按指令列的顺序（瓶子 → 方块 → 关抽屉）；放置误差 ±0.8 cm；离柜顶 1–4 cm 松手；慢推 3–5 cm/s、推到齐平再多 0–4 mm；**finish 前不回头看**（bystander 实测 Sonnet 碰倒瓶子 5 次、0 次去扶）。
- **oracle**：先放方块、关抽屉，最后放瓶子；最后检查每个阶段，有问题就修。
- **adaptive**：第 1 次 = naive + finish 前检查（倒了的瓶子按上面的方法扶回柜顶）；第 2 次起改成先关抽屉再放瓶子，最后也检查。
- **blind**：第 1 次 = naive；之后随机变化：顺序（瓶子-方块-关 0.5 / 方块-瓶子-关 0.3 / 方块-关-瓶子 0.2）、推的速度 2–8 cm/s 和高度 3–7 cm、捏瓶颈或瓶盖、松手高度、放置误差；不看历史、不回头检查。
- 对照：**switch**（第 1 次 naive，之后只换顺序、不回头检查）、**oracle_noise**（oracle ±5 mm 放置误差）、**naive_slow**（naive 但推 2 cm/s，agent 接口的最低速度）、**naive_cap**（naive 但捏瓶盖，瓶子吊着晃）。

## headroom 验证（n = 20，k = 3，实测，`python scripts/validate.py --task lh_drawer_cabinet --n 20 --k 3`，种子 1000–1019，2260 s）
20 个实例：自闭 14、缓闭 6；瓶子在左 / 右随机。`validate.json` 是在定版代码上跑的（之后只改了模块 docstring；抽查 seed 1018 的 naive 和 adaptive 重跑结果逐项相同）。

| 策略 | @1 | @2 | @3 | 第 1 次陷阱触发 | 备注 |
|---|---|---|---|---|---|
| oracle | **100%** | 100% | 100% | 0/20 | 先方块、关抽屉、再瓶子；自闭实例里抽屉照样弹上，但柜顶还没东西 |
| naive（模仿 Sonnet：按列的顺序、慢推 3–5 cm/s、不回头看） | **30%** | — | — | **14/20 = 70%**（自闭 14/14，缓闭 0/6） | 14 次失败全是 `bottle_knocked`，每次**两个瓶子都**因 `drawer_slam` 丢失；柜体窜动 2.3–4.2 cm；**倒之前机器人碰过瓶子 0 次** |
| adaptive | **90%** | 100% | 100% | 14/20 | 14 次触发里 **12 次在第 1 次内扶好**（`recovered_in_attempt`）；另 2 次扶瓶时自己出事（1015 推动了柜子 > 8 cm，1018 扶第二个时碰倒第一个），第 2 次换顺序成功 |
| blind | 30% | 30% | 55% | 14/20 | 只有随机抽到"方块-关-瓶子"顺序的 5 个自闭实例成功；"方块-瓶子-关"照样倒 |

对照（实测，同样 20 个种子，`results/lh_drawer_cabinet/controls.json`）：
| 对照 | @1 | @2 | @3 | 说明 |
|---|---|---|---|---|
| switch（第 1 次 naive，之后只换顺序、不回头检查） | 30% | **100%** | 100% | 跨尝试学习单独就够 |
| oracle_noise（±5 mm 放置误差） | 100% | | | |
| naive_slow（推 2 cm/s，接口最低速度） | 30% | | | 陷阱 14/14 照样触发：放慢没用 |
| naive_cap（捏瓶盖、瓶子吊着放） | 30% | | | 陷阱 14/14；6 个缓闭实例全成功（吊着放也放得稳） |

- 按 agent 命令计：自测里按 naive 顺序做完一遍是 41 条 `move_eef`（脚本化的命令，每步都很直接）；扶两个瓶子约再 16 条；`agent_max_steps = 120`、360 s 仿真时间留了看图和修正的余量【推断：按命令条数估算】。
- 演示视频：`videos/lh_drawer_cabinet_adaptive_s1000.mp4`（adaptive，seed 1000：第 1 次关抽屉后两个瓶子倒在围栏上，同一次尝试里扶回去，成功）。

## agent 接口自测（la 风格，rgb，F1，k=3，seed 3000（自闭、瓶子在左），端口 12893，2026-10-16，实测）
`LFF_STYLE=la scripts/start_robot_session.sh lh_drawer_cabinet 3000 full F1 12893 direct 3 smoke rgb 1`（日志放在 scratchpad `smoke/runs/`）；不调模型，用 HTTP 直接发命令（`POST /call {"cmd","args"}`，`X-Token` 取自 `agent_iface.json`，同 `scripts/api_player.call_server`），命令表由 `smoke/plan.py` 按 naive 计划生成（robot-base 坐标）。结束后只停了自己起的那个服务器进程。
- README 的 Task 段就是新指令原文（含"it is pulled out towards the robot"和瓶盖提示），预算 120 条命令 / 360 s。
- 第 1 次：作废——我生成的命令表里放完瓶子后直接竖直下降去拿方块，手压在瓶子上把两个瓶子碰倒（计划 bug）。`finish` → "Attempt failed."；日志原因 `robot_hand / robot_arm`，`trap_fired=False`——顺带说明监视器能把"手碰的"和"抽屉震的"分开。
- 第 2 次（修好命令表后，41 条）：两个瓶子捏瓶颈放到柜顶 → 两个方块放进抽屉 → 合爪降到面板前、`move_eef x=… speed=0.04` **慢推**。推的这一条命令里抽屉弹上（2.01 m/s），柜体窜 3.3 cm，两个瓶子倒在围栏上（`results/lh_drawer_cabinet/smoke_3000_before_push_agentview.jpg` → `smoke_3000_after_push_agentview.jpg`、`smoke_3000_after_push_wrist.jpg`）；日志 `drawer_slam`、`bottle_fell ×2 cause=drawer_slam robot_touched_before_fall=False`、两个阶段 `lost cause=drawer_slam`，服务器在这一步存了回归快照 `snapshots/a2_s039`。`finish` → 只回 **"Attempt failed."**；`trap_fired=True`、`bottles_lost_to_slam=[brown, green]`、`failure=bottle_knocked`。
- 第 3 次（先方块、慢推关抽屉、再放瓶子）：抽屉照样弹上（2.17 m/s），柜顶还没东西；`finish` → **"Attempt succeeded."**，事件只有 5 个 `achieved` 和一次 `drawer_slam`。

## pilot 种子（实测，`pilot.py`：从 3000 起逐个跑 naive 和 oracle 各 1 次，`results/lh_drawer_cabinet/pilot_seeds.json`）
- 3000（自闭，瓶子在左）：naive 两个瓶子都被震倒（窜 3.4 cm，倒前没碰），oracle 成功；3001（自闭，右）：两个都倒（3.6 cm），oracle 成功；3002 是缓闭（naive 成功，不算）；3003（自闭，右）：两个都倒（3.3 cm），oracle 成功。
- **pilot 种子 = 3000, 3001, 3003**。终态图：`results/lh_drawer_cabinet/pilot_300{0,1,3}_{init,naive,oracle}_agent.png`（naive 终态里柜顶只剩倒向相机的瓶肩和瓶盖，oracle 终态两个瓶子高高立着）。

## 已知问题 / 推测（未验证的标【推断】）
- **agentview 看不到抽屉内部**（抽屉朝机器人、相机在柜子背面），放方块要靠腕部相机。陷阱的结果（瓶子倒）在 agentview 里很明显。
- **还没有真模型小试**。**真实模型会不会先放瓶子**【推断】：指令先列瓶子，Sonnet 习惯按列的顺序做；但"放进方块就顺手关抽屉，最后放瓶子"也很自然，有的模型第 1 次就可能避开陷阱。需要真模型小试确认。
- 瓶子倒向相机、靠在围栏上，agentview 里是"朝向相机的瓶肩和瓶盖"，比躺在桌上不那么醒目（但和站着的高瓶子对比很清楚）【推断：VLM 能看出来，需小试确认】。
- 瓶子从柜顶取下来要从上面捏瓶盖；捏住瓶盖时瓶子吊着会晃，晃着松手可能放不稳（瓶底窄，7° 就倒），这是放置难度，不是陷阱，日志里不会记成 `drawer_slam`。
- 抽屉关上后打不开：误关（在方块放进去之前）会让这次尝试失败。
- 猛推（不给 speed）在缓闭实例里也会把瓶子推倒（见鲁棒性表），原因记为 `robot_*`；Sonnet 小试里推抽屉的速度【未测】，按总管给的 3–5 cm/s。
- 弹簧力用 `qfrc_applied` 每个子步按状态施加（MuJoCo 没有"只在最后 8 cm 起作用"的原生弹簧）；回放 / 快照恢复后下一个子步就重新算，结果确定。
