# LH `lh_wobbly_table`：把四个瓶子摆到小边桌的四个色块上——一条桌腿短 1.5 cm，重瓶一放到那个角，桌子一晃，已经立着的瓶子全倒

代码：`lffbench/tasks/lh_wobbly_table.py`　结果：`results/lh_wobbly_table/`（`validate.json`、`controls.json`、`physics.json`、`pilot_seeds.json`、缩略图 `init_*` / `naive_end_*`、pilot 图、冒烟截图 `smoke_3001_*`）　原型与中间数据：`/tmp/claude-0/-mnt-cpfs/2aa0daff-4c37-4ec7-b3e7-10b838c28688/scratchpad/lh_wobbly_table/`（物理脚本 `t2`–`t24`、`run_kinds.py`、`controls.py`、`pilot.py`、`smoke_mkplan.py` / `smoke_drive.py`、各日志，临时）

> 2026-10-16 新增，长程自扰恢复族（"收拾桌子"类常识任务：边桌腿不平）。总管给的设计意图：小边桌一条腿短几毫米，四条腿看起来一样；东西放到短腿那个角上，负载的力矩让桌子晃到短腿上，这一下把桌上已经立着的高瓶子震倒；晃过之后桌子在新姿态下是稳的，瓶子能再扶起来。上线标准同 docs/11：模仿 Sonnet 的 naive 碰坏约 65–75%，adaptive 尝试内修好或第 2–3 次成功（succ@3 ≥ 80%），blind 明显更低，失败从画面看得出。数字和机制按物理实测改过，见"设计过程"。**未做真模型小试**（本机没有 API）。

| 项 | 内容 |
|---|---|
| 类别 | state_restoration，`level="L2"` |
| 能力 | Perceive（看出瓶子倒了、桌子歪了）、Reason（不是自己碰的，是放某个瓶子时桌子晃了）、Restore（捏瓶塞把倒下的瓶子吊起来立回色块）、Plan（下次先放那个瓶子） |
| 协议 | cross，k = 3，`agent_max_steps = 120`（README 实测：每次 120 条命令、360 s 仿真时间、LA 工具调用 100 次） |
| 指令（直接版） | "Put each of the four bottles on the spot of its own colour on the small wooden table. At the end every bottle must stand upright on its spot, the gripper must be open and clear of the bottles, and the small table must still stand where it is now. (Each bottle has a square neck with a small round white stopper on top. Gripped on the neck, a bottle stays rigid in the fingers; gripped by the stopper alone, it swings freely and hangs straight down.)" |
| 指令（间接版） | "Set the four coloured bottles out on the little wooden side table, each on the patch that matches its colour. When you finish, all four have to be standing upright on their patches, the gripper open and away from them, and the side table left where it stands. (Gripped on its square neck a bottle stays rigid; gripped by the round white stopper alone it swings freely and hangs straight down.)" |
| 成功（终态判定） | 每个瓶子：直立（离竖直 ≤ 10°；桌子晃过之后瓶子本身斜约 5.5°，算直立）、瓶底在小桌桌面上（≤ 6 mm）、没被夹着、瓶底中心在自己色块里（色块每边缩 6 mm：前后 ≤ 3.0 cm、左右 ≤ 2.6 cm，naive 的偏前 0.8–2.0 cm ± 0.6 实测都在容差内）；四个同时成立 + 夹爪没碰任何瓶子 + 场景静止（1 s 内位移 < 3 mm）+ 小桌没被挪走（中心 ≤ 3 cm、转角 ≤ 10°、倾角 ≤ 8°；晃过之后的 4.5–5.1° 算"还站着"）。指令里写了"小桌必须还站在原处"，判定由此而来 |

## 设计过程：总管意图 → 物理逼出来的改动（都是实测）
1. **四个瓶子一样重不行，短腿那个角要放一个重瓶**。短一条腿的四腿桌绕"短腿两边那两条腿"的对角线晃。四个角各放一个一样重的瓶子：对角那个瓶子的力矩正好抵消短腿角的瓶子，先放对角再放短腿角时桌子受力在临界上（放偏 1 cm 就决定晃不晃）；而且短腿角的瓶子自己站在晃的桌子上，第一个放它也会被震倒，"先放它"不安全。所以四个瓶子外观一样，三个空的轻瓶（75–90 g，重心 7.2–8.0 cm），一个装沙的重瓶（0.34–0.38 kg，重心 3.0–3.6 cm），**重瓶的色块就在短腿那个角**（构造上耦合）。重瓶放上去必晃（0.36 kg × 11.3 cm），对角放着轻瓶也压不住；重瓶自己不倒。空桌的重心往短腿对角偏 5 mm，所以空桌平稳地站在三条长腿上。
2. **桌腿要圆脚**。1.6 cm 方腿平底时，转轴移到脚掌靠负载那一边，桌子自重挡住，只晃到 0.5° 就卡在两条腿上（实测）。改成胶囊碰撞体，视觉仍是方腿。
3. **接触要过阻尼**。默认的刚性临界阻尼接触下，轻瓶在已经晃歪（3.4°）的桌面上方 1–4 cm 松手，先着地的上坡边一弹就倒（21/27）；Sonnet 习惯 1–4 cm 松手（DESIGN_INSIGHTS §3），这样"先放重瓶"也会失败。瓶子和桌面接触改成 solref (0.01, 3) 后 0/54（各个短腿量下都是），晃的那一下照样震倒（见下表）。
4. **桌子从 22 cm 放大到 28 cm**。22 cm 桌面上对角两个色块相距 15.6 cm，和瓶高差不多，对角的轻瓶正好倒在重瓶身上斜靠着，瓶塞离重瓶 0.6–2.5 cm，怎么抓都会碰倒重瓶（重瓶一倒桌子就晃回去，扶起重瓶时又晃一次，越修越乱）。28 cm、色块离中心 8 cm（对角 22.6 cm）后不再斜靠。
5. **短腿 14–16 mm，不是 5–12 mm**。28 cm 桌的桌腿离中心 12.5 cm，同样的倾角需要更大高差；"晃了就倒、松手不倒"的窗口是 13–18 mm（下表），取中间 14–16 mm（晃后倾 4.5–5.1°）。
6. **瓶口改了三次**（Panda 手的碰撞包络从抓取点上方 3 cm 开始，指垫上沿在抓取点上方 1.2 cm，夹点上面能伸出的东西只有约 1.8 cm）：
   - 草稿：1.3 cm 圆瓶颈 + 1.8 cm 球塞。没有一个夹取高度能只夹到瓶颈、手又不碰球塞：手指总是夹在球上（闭合宽度 1.71 cm），瓶子绕指间转，全速搬运时重瓶甩出去 5/6，第一次冒烟测试里也掉过一次。
   - 第二版：1.4 cm 方瓶颈 + 1.2 × 2.2 cm 圆柱塞。搬运稳了，但圆柱和指垫是线接触，躺着的瓶子捏塞子提起来卡在 26° 不下垂，放下就倒（这一版的 validate 结果见验证表下的"历史"）。
   - 定版：**1.4 cm 方瓶颈 + 0.7 cm 细颈 + 1.2 cm 白色球塞**，瓶颈以上都比方颈窄。手指只要覆盖到方颈（抓取点在颈顶下 0.4–0.9 cm），就夹在方颈的平面上，刚性（全速三连移动：重瓶 12/12、轻瓶 11/12 没掉）；只夹球塞时单点接触，瓶子自由摆（躺着的瓶子提起来垂直下垂，0.1°，放回色块）。只夹球塞全速搬运会甩掉（6/6），所以指令里写明哪种夹法是刚性的、哪种会摆（和 lh_bystander 写瓶盖一样，是物体的可供性说明，不涉及隐变量）。
7. **不放垫片**。初始时短腿悬空 1.4–1.6 cm，理论上能把薄垫片塞进去；但只能顶视夹取、看不出哪条腿悬空（视觉腿画成全长），把垫片推到 1.6 cm 粗的腿下需要 ~3 mm 精度，推偏会推动桌子；而且桌边放一块垫片等于提示"桌腿有问题"，把失败前就能推出来的信息送给 agent（违反 AGENT_GUIDE"信息要在失败的结果里"）。没做。

## 场景
- 小木边桌（自由体，1.1 kg）：桌面 28 × 28 × 1.6 cm，高 8.5 cm，四条 1.6 cm 方腿在 ±12.5 cm；桌心在机器人基座前 53–57 cm（x ∈ [−0.13, −0.09]），左右 ±4 cm。桌面四角各一块彩色色块（7.2 × 6.4 cm，暗红 / 深蓝 / 深绿 / 赭黄，纯视觉，离中心 8 cm），颜色随机排。
- 四个方形玻璃瓶（红 / 蓝 / 绿 / 黄）：瓶身 3.2 × 3.2 × 10 cm、2.4 cm 平底脚、瓶肩、1.4 cm 方瓶颈、细颈、1.2 cm 白色球塞，总高 14.7 cm，带浅色标签。在大桌上排成一行：边桌前方（机器人一侧）或左 / 右侧（随机），间距 7.5–8.5 cm，颜色顺序随机。所有抓取点离基座水平距离 ≤ 0.70 m（实测 > 0.72 m 时低处目标会差 2–3 cm 到不了）。
- 隐变量：哪条腿短（碰撞胶囊短 14–16 mm，视觉腿画全长，四条腿看起来一样、都着地）、哪个瓶子重（外观一样）。两者耦合：重瓶的色块在短腿那个角。
- 实例平衡：在 naive 脚本的顺序里（离供应区最远的色块先放，再按左先），重瓶的色块排第 1 / 2 / 3 / 4 的各占每 20 个连续种子的 5 个（`(seed × 7) mod 20 // 5`）。排第 1 的 25% 对这个顺序是良性的（桌子晃时上面还没东西）；"近的先放"的顺序正好反过来，触发率同样是 15/20（对照实测）。200 个种子里重瓶颜色 红 58 / 蓝 51 / 黄 47 / 绿 44，短腿角四个各 47–54（种子 1000–1019 里红色碰巧占 11 个）。

## 阶段 → 自扰机制 → 要学会的
- **阶段**：四个瓶子各立到同色色块上（监视谓词 `red / blue / green / yellow`）。
- **自扰机制（物理，不瞬移，机器人不碰倒下的瓶子）**：重瓶一放到短腿角，桌子绕对角线晃下去，短腿着地"咚"一下停住，桌面倾 4.5–5.1°；立在桌上的轻瓶被甩向低角倒下，有的掉到大桌上。重瓶不倒。晃过之后桌子稳定（重瓶压着；再加轻瓶也不动），倒的瓶子能扶起来。把重瓶拿起来、或在翘起的那个角附近往下按 1.25 N，桌子会晃回去（实测）。
- **R4（同一失败外观的多种解释，最省事的是错的）**：画面上是"放完黄瓶，其他瓶子都倒了"。最省事的解释——"我松手太高 / 手碰到了 / 瓶子本来就不稳"——都不对：验证中 15 次晃倒，倒下的瓶子在晃之前 2 s 内被机器人碰过 0 次、倒的过程中被碰 0 次；轻轻放（1 mm）也照晃（物理表是零速度放置）；`naive_heavy_first` 对照用同样的 1–4 cm 高松手、同样的放偏，只是先放重瓶，20/20 成功。对的解释是"这个角的桌腿短 / 这个瓶子重，一放桌子就晃"。
- **看得见**：agentview（512 px）里倒下的瓶子很明显（`naive_end_*.png`、`pilot_*_naive_agent.png`、`smoke_3001_a1_after_rock_agentview.jpg`）；桌面也看得出歪了：前沿一头低、翘起的那条腿离开桌面约 1.5 cm（放大图 scratchpad `crop_1007_restand.png`）。腕部相机在 39 cm 高俯视，能看到躺着的瓶子和它的瓶塞（`smoke_3001_a1_after_rock_wrist.jpg`）。
- **要学会的**：(1) 尝试内：放完重瓶后回头看每个瓶子，倒了的捏住球塞吊起来、立回色块（桌子已经稳了）；瓶塞旁边有立着的瓶子时先把倒瓶挪到空处。(2) 跨尝试：先放重瓶（让桌子在空的时候晃），再放其他。

## 物理实测（`results/lh_wobbly_table/physics.json`；瞬移放置、零速度，6 个种子 × 3 个轻瓶，28 cm 桌、重瓶 0.36 kg、接触 solref (0.01, 3)）
| 短腿 | 晃后桌面倾角 | 晃一下震倒轻瓶（定版瓶 / 草稿瓶） | 已晃的桌上 1 / 2 / 4 cm 松手倒 |
|---|---|---|---|
| 10 mm | — | — / 0/18 | 0/18 ×3 |
| 12 mm | — | 10/18 / 8/18（重瓶 0.30 kg：9/18） | 0/18 ×3 |
| 13 mm | — | — / 18/18 | 0/18 ×3 |
| 14 mm | ~4.5° | 18/18 / 18/18（重瓶 0.30 kg 也 18/18） | 0/18 ×3 |
| 16 mm | ~5.1° | 18/18 / 18/18 | 0/18 ×3 |
| 18 mm | — | 18/18 / 18/18 | 0/18 ×3 |

其他实测：空桌在三条长腿上水平（0.02°）；三个轻瓶放在各自色块上、4 cm 高松手且朝短腿方向放偏 2.6 cm，桌子不晃（4/4）；轻瓶放到短腿角（错位）会让空桌晃（85 g × 11.3 cm > 1.1 kg × 5 mm）；22 cm 草稿桌上刚性接触时晃倒 9/9 但松手倒 21/27，过阻尼 (0.01, 3) 时 9–12 mm 都是 18/18 晃倒、0/54 松手倒。Panda 手的碰撞包络（`t8_hand.py`）：从抓取点上方 3 cm 起，沿手指方向 ±10 cm 宽、横向 −1.7..2.4 cm 的板；张开的手指在两侧 3.9–6.6 cm，从抓取点下 1 cm 到上 4.5 cm；前臂（`t16_arm.py`）在低处抓取点上方约 10 cm（±5 cm 内）、往基座方向 10 cm 处约 16 cm、15 cm 处 16–20 cm、20 cm 外 ≥ 31 cm，横向 −14..+10 cm。

## 事件监视器（只进服务器日志，agent 看不到）
`WobblySkills._act` 每个控制步调用 `monitor_step`；另外包了一层 `sim.step`，每个 MuJoCo 子步记录机器人（手指 / 手 / 手臂）—瓶子、瓶子—瓶子、机器人—小桌的接触。
- 阶段事件 `achieved / lost / restored`（每 0.1 s 算一次、0.5 s 去抖、3 mm 滞回）；`lost` 带 `cause`：`table_rocked` / `table_rocked_back` / `robot_finger` / `robot_hand` / `robot_arm` / `bottle:<颜色>` / `picked_up` / `other`，以及倾角和偏移；`restored` 带 `after`。
- 桌子事件 `table_rocked`（倾角 > 1.5°，去抖）/ `table_rocked_back`（< 0.8°）：`released`（2 s 内刚松手的瓶子）、`low_corner`（机器人视角 far-left / far-right / near-left / near-right）、`standing`（当时立在色块上的瓶子）、`robot_touched_before`（其中晃之前 2 s 内被机器人碰过的）、`robot_touched_table`，2 s 后补上 `fell`（其中倒下的）和 `robot_touched_while_falling`。阶段丢失事件会触发服务器的回归快照（冒烟测试里存了 `snapshots/a1_s018`）。
- `outcome()` 返回 `success, stages, events, n_regressions, n_lost, n_restored, n_rocks, rocked, trap_fired`（有一次晃震倒了已立着的瓶子）`, recovered_in_attempt, fell_by_rock, failure, held, touching, static, table_ok, table_moved_cm, table_tilt_deg, max_table_tilt_deg, offsets_cm, tilts_deg, heavy, short_corner, detail`。`failure`：`none / table_moved / held / toppled_by_rock / toppled / not_on_spot / touching / not_static`。`detail`（F2，只做消融）例（冒烟测试第 1 次）：
> The red bottle is not on the small table. The blue bottle lies on its side on the small table. It had stood on its spot and fell when the small table rocked. The green bottle is not on the small table. When the yellow bottle was set down, the small table rocked down onto its near-left leg and the blue bottle standing on it fell over. That leg is 1.4 cm shorter than the other three.

## 脚本策略（完美感知 = 仿真状态 + Sonnet 式偏差；只用 move_to + 开合爪）
- 共用动作：开爪 → 抬到 40 cm（夹着的瓶子底离桌 28 cm，高过桌上立着的瓶子）→ 平移到瓶子上方 → 竖直下降到方颈（颈顶下 6 mm）→ 合爪 → 抬回 40 cm → 平移到色块上方 → 竖直下降到瓶底离桌面 `gap` → 开爪 → 抬起。
- **naive（模仿 Sonnet 习惯）**：离供应区最远的色块先放（不越过已放的瓶子），每次 1–4 cm 高松手（随机），每个实例偏前 0.8–2.0 cm + 每轴 ±0.6 cm；对桌子没有任何计划；放完不回头看。
- **oracle**：知道短腿：先放重瓶，其余 4 mm 高轻放，每步后看一眼、最后检查。
- **adaptive**：第 1 次 = naive + 每放一个就看所有瓶子，倒了就扶（先处理立着但偏了的，再按抓瓶塞的余量从大到小；手 / 张开的手指 / 前臂会碰到立着的瓶子或桌面时，先抓瓶身把它挪到大桌空处平放，再抓瓶塞）+ 最后检查；第 2 次起：上一次 `table_rocked` 事件里 `released` 的那个瓶子先放（偏差和 naive 一样），也每步检查。
- **blind**：第 1 次 = naive；之后随机顺序 + 随机搬运高度 / 速度 / 偏差，不看历史、不回头检查。
- 对照：**naive_heavy_first**（naive 的一切习惯，只是重瓶先放）、**switch**（第 2 次起只换顺序、不修）、**naive_near_first**（近的先放）、**oracle_noise**（oracle ±5 mm）。

## headroom 验证（定版，n = 20，k = 3，种子 1000–1019，`python scripts/validate.py --task lh_wobbly_table --n 20 --k 3`）
20 个实例：重瓶在 naive 顺序第 1 / 2 / 3 / 4 各 5 个；供应区前 / 左 / 右随机。3625 s。

| 策略 | @1 | @2 | @3 | 第 1 次陷阱触发 | 备注 |
|---|---|---|---|---|---|
| oracle | **100%** | 100% | 100% | 0/20 | 20 次都晃了一下（重瓶先放，桌上没东西），无害 |
| naive（模仿 Sonnet：远的先放、1–4 cm 松手、偏前 0.8–2.0 cm） | **25%** | — | — | **15/20 = 75%** | 失败 15/15 是 `toppled_by_rock`；每次晃倒 1 / 2 / 3 个瓶子各 5 次（共 30 个，其中 20 个掉到大桌上）；15/15 次晃都记到重瓶松手上；**倒下的瓶子晃之前被机器人碰过 0 次、倒的过程中 0 次**；晃后桌面倾 4.53–5.13°；立着的瓶子前后偏 0.27–2.41 cm、左右 ≤ 0.61 cm，都在容差内；成功的 5 个都是重瓶排第 1 |
| adaptive | 25% | **100%** | 100% | 15/20 | **尝试内修复 0/15**（失败：9 次有瓶子扶不起来，6 次修的过程中把小桌推动了 > 3 cm）；第 2 次 15/15 先放重瓶、0 次触发、全部成功 |
| blind | 25% | 40% | 40% | 15/20 | 第 2、3 次随机顺序，各有 12 次又触发；39 次失败全是 `toppled_by_rock` |

- **尝试内修复为什么是 0**（实测）：倒下的 30 个瓶子里 20 个掉到大桌上，多数贴着小桌的腿 / 桌边躺着。抓瓶塞时手指必须横跨瓶身，Panda 手沿手指方向 20 cm 宽、从抓取点上方 3 cm 起，一伸下去就撞到小桌桌面（脚本检测到会先尝试抓瓶身挪开，那也被挡住；硬抓就把小桌推走）。同样的扶起动作对大桌空处平躺的瓶子 5/6 成功、对躺在小桌桌面上的瓶子也成功（`t25_free.py`、`t22_restand.py`）。所以这是顶视、只能绕竖轴转的夹爪的够不着，不是物理 bug。
- **历史（不和定版混算）**：同一套场景、第二版瓶口（圆柱塞）的验证 oracle 100 / naive 25（触发 15/20）/ adaptive 55→100→100（尝试内修复 6/15）/ blind 25→40→40；那一版提起躺着的瓶子会卡在 26°，和指令"hangs straight down"不符，所以弃用。第一版（球塞 + 圆颈，8 个种子没跑完）脚本修复 13/15，但重瓶在快速搬运中会从指间甩出。

## 对照（定版，同样 20 个种子，`results/lh_wobbly_table/controls.json`）
| 对照 | @1 | @2 | @3 | 第 1 次陷阱触发 | 说明 |
|---|---|---|---|---|---|
| **naive_heavy_first** | **100%** | | | 0/20 | naive 的 1–4 cm 松手 + 偏前放置，只是先放重瓶：跨尝试的修正在 naive 习惯下也成立 |
| switch | 25% | **100%** | 100% | 15/20 | 第 2 次只换顺序（重瓶先放）、不修，15/15 成功 |
| naive_near_first | 25% | 25% | 25% | 15/20 | 换成"近的先放"，触发率不变 |
| oracle_noise（±5 mm） | 100% | | | 0/20 | |

## agent 接口自测（la 风格，rgb，F1，k=3，seed 3001，端口 13473，2026-10-16，实测，定版代码）
`LFF_STYLE=la LFF_RUNS_DIR=<scratchpad>/runs scripts/start_robot_session.sh lh_wobbly_table 3001 full F1 13473 direct 3 smoke3 rgb 1`，用 `smoke_drive.py` 按 `scripts/api_player.py call_server` 的方式向 `http://127.0.0.1:13473/call` POST `{"cmd","args"}`（X-Token 取自 `agent_iface.json`），不调用模型；结束后只杀了自己启动的服务器进程。README 是新指令（含方颈 / 球塞的说明）、每次 120 条命令 / 360 s / 100 次工具调用。
- 第 1 次（20 条命令，模仿 naive）：夹蓝瓶方颈 → 放到 far-left 色块（偏前 1 cm、瓶底离桌 2 cm 松手）→ 夹黄瓶（重瓶）→ 放到 near-left 色块（短腿角）。松手前两瓶都立着（`smoke_3001_a1_before_release_agentview.jpg`），松手后蓝瓶躺倒（`smoke_3001_a1_after_rock_agentview.jpg`、腕部图）。`finish` → 只回 "Attempt failed."；日志：`table_rocked`（released=yellow、standing=[blue]、robot_touched_before=[]、fell=[blue]、robot_touched_while_falling=[]）、`failure=toppled_by_rock`、`trap_fired=True`，回归快照 `a1_s018` 已存。
- 第 2 次（38 条）：黄瓶先放（桌子晃了一下，上面没东西），再放蓝、绿、红 → **"Attempt succeeded."**，桌面倾 4.6°，四个瓶子偏前 1.6–2.1 cm、左右 ≤ 0.7 cm。
- 早先一版（球塞草稿）冒烟时第 2 次搬运中重瓶从指间滑出（宽度 0.017 → 0.001），促成了瓶口的改版（见设计过程 6）。

## pilot 种子（`pilot.py`：从 3000 起逐个跑 naive 和 oracle 各 1 次，`results/lh_wobbly_table/pilot_seeds.json`，定版代码）
- 3000（供应区左、重瓶在 naive 顺序第 1）：naive 成功（良性，不算）；3001（右、第 2）：naive 晃倒蓝瓶，oracle 成功；3002（前、第 3）：晃倒红、黄，oracle 成功；3003（右、第 1）：良性；3004（前、第 2）：晃倒红瓶，oracle 成功。三次晃倒都没被机器人碰过。
- **pilot 种子 = 3001, 3002, 3004**。图：`results/lh_wobbly_table/pilot_300{0..4}_{init,naive,oracle}_agent.png`。

## 已知问题 / 推测（未验证的标【推断】）
- **重瓶和短腿是构造耦合的**：从 agent 看，"重瓶先放"和"那个角先放"都对。naive 的"远的先放"是从 nudge_row 小试搬来的；【推断】真模型可能有别的固定顺序（按指令颜色顺序、近的先放），重瓶在任何固定顺序中的位置近似均匀，触发率仍约 75%，但具体实例会变（近的先放：同样 15/20）。
- **重瓶外观和轻瓶一样**，不会有"重的 / 矮胖的先放"这种习惯性规避；代价是失败前完全无从知道。夹着重瓶时没有力反馈。
- **尝试内修复对顶视夹爪很难**：桌子晃后在翘起的角附近按 1.25 N 就会晃回去，松开再晃下来时又震倒立着的瓶子；倒下的瓶子常躺在重瓶旁边或掉到大桌上紧挨着边桌，Panda 手（沿手指方向 20 cm 宽、离抓取点 3 cm 起）够不到瓶塞。脚本 adaptive 的尝试内修复是 **0/15**（定版；掉到大桌上贴着小桌的瓶子够不着，见验证表下的说明）。【推断】真模型也很难在同一次尝试内修好，会直接去抓瓶塞、碰到小桌或重瓶，引发连锁晃动。所以这个任务的学习信号在跨尝试（重瓶先放：adaptive / switch 第 2 次 15/15），`recovered_in_attempt` 在这个任务上预计接近 0。不满足 DESIGN_INSIGHTS §2.1"adaptive 脚本能全部修好"的尝试内版本；满足总管给的"尝试内修好和/或第 2–3 次成功"。可能的 v2：桌面四周加一圈低挡边（倒瓶留在桌面上，手在桌面以上够得着），或把小桌做矮（≤ 4.5 cm，手的板在桌面以上）。【推断，未建】
- **只夹球塞搬运会甩掉瓶子**（全速 6/6），指令里写了方颈是刚性夹法；方颈可夹的高度窗口约 5 mm（颈顶下 0.2–0.9 cm：再高手指只碰到球，再低手碰到球塞顶、停在目标前几毫米）。【推断】agent 第一次夹时可能夹太高，搬运中瓶子摆动；慢速搬运时轻瓶不会掉。
- 视觉腿是全长画的：晃过之后短腿的视觉部分插进大桌面 1.5 cm（被桌面挡住看不见）；翘起那条腿离地的缝是真实可见的。
- 桌面倾斜后瓶子都斜约 5.5°，agent 可能想"扶正"，白费命令但不扣分。
- 种子 1000–1019 里重瓶是红色的有 11 个（200 个种子里均匀）。
- 只做了脚本验证和接口自测，**没有真模型小试**；naive 的习惯（顺序、松手高度、偏差）是从别的任务的 Sonnet 小试搬来的。
