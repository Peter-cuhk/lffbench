# LH `lh_false_bottom_lid`：假底盒子——红块立着放进去盖子合不上，要把它放倒

代码：`lffbench/tasks/lh_false_bottom_lid.py`　结果：`results/lh_false_bottom_lid/validate.json`　原型与中间数据：`/tmp/claude-0/-mnt-cpfs/ddd4b8c2-b2b4-418d-a89e-e1a09f98ae15/scratchpad/lh_false_bottom_lid/`（临时目录）

> 2026-10-12 新增，P2 常识任务（`research_notes/P1P2_加深与常识任务设计.md` §4 行 P2-1），长程自扰恢复族（和 `lh_lid_box` 共用铰链盖和 `LidSkills`）。设计规则 §2：隐变量不能靠视觉伺服抵掉（R1）、动手前看不出（R2）、第一次按常识的交互有代价（R3：取出、放倒、再放进去要 10 条以上命令）、失败要在 agentview 里看得见（R5）。

| 项 | 内容 |
|---|---|
| 类别 | state_restoration，`level="L2"` |
| 能力 | Perceive（红块是否高出盒沿、盖子是否搭在上面）、Reason（为什么合不上：盒子比看起来浅）、Plan（取出 → 放倒 → 躺着放进去 → 再合盖）、Restore（尝试内修复；下一次直接先放倒） |
| 协议 | cross，k = 3，`agent_max_steps = 120` |
| 指令（直接版） | "Put the red block and the two blue cubes into the wooden box and close its lid. The lid opens by lifting the black knob on its edge." |
| 指令（间接版） | "Pack the tall red piece and both small blue cubes into the wooden chest and shut it (the lid lifts by the black ball on its edge)." |
| 成功（终态判定） | 盖子关上（≤ 5°）+ 三件都在盒内（碰撞盒的整个底面投影在内壁之内，容差 0.4 cm；顶点不高于盒沿 0.4 cm；没被夹着）+ 没有东西被夹着 + 机器人没有压着盖子或物品 + 场景静止。不检查顺序，也不检查红块的姿态 |

## 场景
- 木箱（fixture）：外形 20.6（前后）× 13.6（左右）× 9 cm 高，壁厚 0.8 cm，底板 0.6 cm；内腔 19 × 12 cm。内壁、底和假底都是纯黑无高光的材质。盖子 0.6 cm 厚、0.15 kg，铰链沿箱子**左边或右边的长边**（随机），对面长边正中伸出 3.2 cm 细杆 + 黑色球形把手（半径 1.3 cm）。开过竖直后盖子倒向铰链一侧，停在止挡（182–192°，随机）上，略低于水平地挂在箱子旁边（自由端离桌 6.5–7 cm），不挡搬运。箱子中心 x ∈ [-0.17, -0.14]、y ∈ [-0.02, 0.02]（世界系，机器人基座在 x = -0.66）。
- 红块 4 × 4 × 7.5 cm（密度 600，72 g，μ 0.8），**立着**；两个 4 cm 蓝方块（32 g）。三件在箱子前面（靠机器人一侧，x ≈ -0.34）排成一排，四个位置 y = ±0.08、±0.24（相距 16 cm）随机占三个，各自转 ±17°。
- 隐变量：**盒子的内深**。深盒内深 8.4 cm；假底盒在底板上多一块黑色假底，内深只有 4.5 cm。每 20 个连续种子里正好 10 个假底（`(seed × 7) mod 20 < 10`，和 `lh_lid_box` 同一套固定配比）。
- 看不出来（实测，`t_view2.py`）：开盖、空箱、同一手臂位姿下，假底盒和深盒的 agentview 和腕部相机（离桌 30 cm 和 16 cm 俯视）渲染**逐像素相同**（0 个像素不同）。Panda 手掌沿手指方向 21 cm，比内腔两个方向都长，手不能伸进盒沿以下，指尖够不到假底。
- 红块立着 7.5 cm：深盒里顶面低于盒沿 0.9 cm；假底盒里**高出盒沿 3 cm**。躺着 4 cm 高，两种盒子都放得下（横着放，7.5 cm < 12 cm 内宽）。

## 阶段 → 自扰机制 → 要学会的
- **阶段**：S1 开盖；S2 红块进盒；S3、S4 两个蓝方块进盒；S5 关盖。监视的谓词：`red_block_1_in_box`、`blue_cube_1_in_box`、`blue_cube_2_in_box`、`lid_closed`。
- **自扰机制（物理）**：红块被找到时是立着的，竖直夹爪只能绕竖轴转，自然就立着放进去。假底盒里它高出盒沿 3 cm，关盖时盖子落在红块顶上，停在 **28–41°**（实测：验证集 10 个假底实例 27.9–39.8°，开发种子 2000–2007 29.5–40.8°），两个相机都看得见（agentview：盖子斜搭着、红块露出一截；腕部相机：盖板和把手下面是红块）。深盒里一切正常。
- **代价（R3）**：把盖子按下去没用（见下），只能重新开盖、把红块取出、放倒、躺着放进去、再关盖，脚本动作约 30 条命令。
- **要学会的**：看到盖子搭在红块上（或者放进去时红块就高出盒沿）→ 取出红块，推它的上部把它推倒，躺着放进去（尝试内恢复）；下一次尝试在放进去之前就先把它推倒（跨尝试）。
- **判定不检查红块姿态**：深盒里立着放也算成功，所以 naive 在深盒实例上应当成功；假底实例上只有放倒才能关盖。

## 盖子和放倒：单步要简单（`FBLidSkills`，实测）
| 问题 | 现象 / 处理 |
|---|---|
| 夹着把手走直线会拉脱 | 沿用 `lh_lid_box` 的 `LidSkills`：夹着把手时每步指令位移投影到把手绕铰链的圆的切向（≤ 1 cm/步），加径向和轴向纠偏；这里把铰链轴推广到任意水平方向（本任务沿前后方向）。开盖 = 夹把手 → 上抬 10 cm → 移到铰链外 4 cm、上方 16 cm → 松手（盖子自己倒到止挡）；关盖 = 夹把手 → 上抬 10 cm → 移到铰链内侧 8 cm、上方 10 cm → 松手 |
| 盖子压穿红块 | 原型里用默认接触参数、手臂把盖子往下压时，盖板穿进 72 g 的红块。处理：盖板 `solref 0.005 1`、`solimp 0.99 0.998 0.001`、`solmix 1000`（盖板的硬接触在和物品的接触里占主导） |
| 手臂把盖子硬压到物品上 | 夹着把手时，如果盖子正搭在物品上，指令里"往关"的分量清零；手（不夹把手）直接压盖子时，去掉往关的分量并每步退让 3 mm（`BACKOFF`，模拟限力手臂）。实测（`t_press2.py`，3 个假底实例 × 5 种压法：夹把手往下压 5 cm、闭合手指压盖板中部 2 / 5 / 8 cm、压盖板边缘 5 cm）：压的过程中盖角最低到 −4 … 23°，松手后都回到 14–40°，终态全部 `lid_blocked`，没有一次合上 |
| 怎么把 7.5 cm 的块放倒 | `tip_over`：闭合手指，在块的上部（离桌 4.5–6.5 cm）沿 ±y（朝远离其他物品的一侧）推 9.5 cm。实测：4 个实例 × 推高 4.5 / 5.5 / 6.5 cm × 速度 0.05 / 0.1 / 0.3 m/s = **36/36 倒下**，再加种子 2000–2019 随机推高 / 速度 / 横向偏移 ±5 mm：**20/20 倒下**；邻近的蓝方块（16 cm 外）位移全部 0 cm。推时手掌不会先碰到块（块只有 7.5 cm 高，手指先接触）。倒下的块躺在推的方向上，之后沿它的长轴转手腕、从上面夹它的中段，横着放进盒子（夹取槽位比立着时往里 1 cm，手指才不碰后壁） |
| 躺着放进去 | 种子 2000–2019：倒下后夹起、放进开着的盒子 19/20 在盒内（2004 一次抓空；之后夹取加了"大幅转腕后停 8 步再下降"和抓空重试一次，2004 的 oracle 已成功）；躺着的红块顶面低于盒沿 0.5 cm（假底）/ 4.4 cm（深盒） |

## 事件监视器（只进服务器日志，agent 看不到）
`FBLidSkills._act` 每个控制步调用 `monitor_step`：
- 阶段谓词 `*_in_box`（每 2 个控制步算一次，变化要连续 6 次（0.6 s）才记事件，防止手指摩擦刚放下的物品抖动）：`achieved / lost / restored`，丢失带 `cause`（`picked_up` / `robot` / `lid` / `other`）和当时的盖角。
- 盖子：`opened_past_vertical`、`closed`、**`lid_blocked`**（盖子从竖直以上落下、没被夹着、在 5–80° 之间静止 1 s 且正接触某个物品；记盖角和压着的物品）。
- 红块姿态（debounce 1 s）：**`item_relaid`**（立 → 躺，记在桌上还是盒里、是谁碰的）、`stood_up`、`placed_upright_in_box`（立着放进盒里，记顶面高出盒沿多少 cm）。
- `outcome()` 返回 `success, failure, stages{...}, events[...], n_regressions, n_restored, false_bottom, n_lid_blocked, lid_blocked_max_deg, red_relaid, red_placed_upright_in_box, red_pose, red_top_above_rim_cm, lid_on, lid_open_max, lid_final_deg, lid_closes, knob_grasps, knob_steps, detail`。`failure` 取值：`lid_blocked`（盖子开着且压在物品上）/ `held` / `robot_touching`（finish 时手还压着盖子或物品）/ `red_sticks_out` / `item_not_in_box` / `lid_open` / `not_static`。`detail` 是 F2 风格文字（机器人视角方向词，不给坐标），例如：
  > The lid is not closed (open 31 degrees): it rests on the red block. The red block is in the box but sticks out 3.0 cm above the rim (upright). The lid came down on an item 1 time(s) during the attempt.

  F1 下 agent 只看到 "Attempt failed."。

## 脚本策略（完美感知 = 仿真状态）
- 共用动作：从上面夹物品中段（立着的红块夹离底 4.25 cm 处），夹取点离桌 16 cm 搬运（用户给的 11–20 cm 区间；盖子挂在旁边 6.5–7 cm 高，不挡），放进盒子时按"深盒"下降（手停在盒沿上方），松手；每放一件都看一眼，掉到盒外的放回去。
- **naive**：开盖 → 红块按原样（立着）放进去 → 两个蓝方块 → 关盖 → 盖子没关上就夹把手往下压一次（还开着超过 90° 就重做关盖）→ finish。不会放倒红块。
- **oracle**：知道内深：假底盒先把红块推倒再放进去，深盒直接放；最后检查每个谓词、不对就修（最多 2 轮）。
- **adaptive**：第 1 次按 naive 的顺序，但 finish 前检查每个谓词：盖子搭在红块上就重新开盖、取出红块、推倒、躺着放进去、关盖（尝试内恢复）；第 2 次起放进去之前就先推倒。
- **blind**：第 1 次 = naive；之后 naive 加随机变化（放置 ±5 mm、搬运高度 13–19 cm、开盖松手位置、关盖位置），不看历史、不放倒。
- 额外对照（不在 validate 表里）：**switch** = 第 1 次 naive，第 2 次起只是"先推倒"、不做尝试内检查；**oracle_noise** = oracle 加 ±5 mm 放置噪声。

## headroom 验证（n = 20，k = 3，实测，`python3 scripts/validate.py --task lh_false_bottom_lid --n 20 --k 3`，种子 1000–1019，1125 s）
20 个实例里 10 个假底、10 个深盒；铰链在左 12、在右 8。跑了两轮：第一轮（`results/lh_false_bottom_lid/validate_v1_before_knobfix.json`）之后把 `_grasp_knob` 固定到 yaw −90（见"已知问题"），第二轮（`results/lh_false_bottom_lid/validate.json`，1234 s）是定版代码。两轮 @k 数字只差 blind 的运气成分；下表是第二轮。

| 策略 | @1 | @2 | @3 | 第 1 次 `lid_blocked`（陷阱触发） | 备注 |
|---|---|---|---|---|---|
| oracle（假底盒先放倒；最后检查） | **100%** | 100% | 100% | 0/20 | 假底 10/10 都放倒（`item_relaid` 10 次），深盒 10/10 立着放；两轮都是 20/20 |
| naive（立着放、不放倒；盖子没关就压一次） | **50%** | — | — | **10/20 = 50%**（= 全部假底实例，盖角 27.9–39.8°） | 失败 10/10 都是 `lid_blocked`；深盒 10/10 成功；两轮完全一致 |
| adaptive（第 1 次 naive + finish 前检查并修复；第 2 次起先放倒） | **95%** | **100%** | 100% | 10/20 | 10 个假底里 9 个在第 1 次内修好（取出 → 推倒 → 躺着放 → 关盖）；第 1 轮 1004、第 2 轮 1000 各有 1 个最后关盖没关上（`lid_open`，见"已知问题"），第 2 次都成功 |
| blind（naive + 随机放置 / 搬运高度 / 松手位置，不放倒） | 50% | 55% | 60% | 10/20 | 假底实例第 2、3 次各有 1 次成功，都是低空搬运把红块撞倒在盒里（`item_relaid where=box cause=robot`，倒了就放得下）；另有 1 次红块撞前壁掉到盒外（第 1 轮：50 / 50 / 55%） |

- **单步可靠性**：oracle 20/20；oracle 加 ±5 mm（均匀）放置噪声，种子 4000–4019：**20/20**（实测，`onoise.log`）；推倒 56/56、躺着放进去 19/20（见上表）；盖子开 / 关在验证集 20 个实例的 oracle 里 40/40（两次把手夹取都成功）。
- 脚本每次尝试的墙钟时间 9–40 s（本机 CPU，机器被别的任务占着）；按 agent 命令计：谨慎做法（开盖 6 + 推倒 5 + 三件取放 21 + 关盖 6）约 38 条 + 观察；尝试内修复（重开盖、取出、推倒、放进、关盖）再加约 30 条；`agent_max_steps = 120` 留了余量【推断：按命令条数估算，未用真实 agent 测】。

## agent 接口自测（la 风格，rgb，F1，k=3，seed 3000（假底、铰链在左），端口 12630，2026-10-12，实测）
`claude_batch.py start --batch smoke_lh_false_bottom_lid --only harness --style la --perception rgb --feedback F1 --k 3 --port0 12630 --tasks lh_false_bottom_lid:3000`。README 里是指令、3 次尝试、每次 120 条命令 / 360 s 仿真时间。手动发了 18 条命令：`status`；夹把手（yaw 90 移到上方、下降、合爪：宽度 0.0265 = 球把手）；`move_eef z=0.18` 夹着把手——手臂沿圆弧走到 (y −0.093, z 0.167)，不是直线目标；移到铰链外 4 cm、上方 16 cm，松手，抬起，`observe`：盖子倒到 184°、挂在箱子左侧；夹红块（宽度 0.037）、搬到箱子上方 16 cm、下降到手停在盒沿上方、松手、抬起、`observe`：**红块立在假底上，顶面高出盒沿 3.0 cm，agentview 和腕部相机都看得很清楚**（`results/lh_false_bottom_lid/smoke_3000_red_upright_*.jpg`）；`finish reason="smoke"`。回报只有 "Attempt failed."，并自动开始第 2 次（场景复位）。服务器日志的 outcome：`failure=item_not_in_box`（两个蓝方块还在桌上）、`red_placed_upright_in_box=True, red_top_above_rim_cm=3.0, lid_open_max=184.2, knob_grasps=1, knob_steps=118`；事件 `lid opened_past_vertical (step 162) → red_pose placed_upright_in_box, top_above_rim_cm 2.8 (324)`；`detail`："The lid is not closed (open 184 degrees). The red block is in the box but sticks out 3.0 cm above the rim (upright). A blue cube is not in the box: its centre is 18.4 cm backward and 8.3 cm to the left of the box's centre. …"。之后 `claude_batch.py stop` 关掉。

## pilot 种子（实测，`t_pilot.py`：从 3000 起逐个跑 naive 和 oracle 各 1 次）
- 3000（假底）naive `lid_blocked` / oracle 成功；3001（假底）同；3002（深盒）naive 和 oracle 都成功（不算）；3003（假底）naive `lid_blocked` / oracle 成功。
- **pilot 种子 = 3000, 3001, 3003**。终态图：`results/lh_false_bottom_lid/pilot_300{0,1,3}_naive_agent.png`（盖子斜搭在红块上）、`pilot_3000_oracle_agent.png`（盖子关上）。

## 设计过程中的实测（为什么是现在这些尺寸）
| 试过的设计 | 结果（实测） | 结论 |
|---|---|---|
| 内腔 17 × 10 cm（draft0） | 夹着 4 cm 物品放进去时，闭合的手指离物品中心约 4 cm，靠后壁的手指刮到内壁，红块被挂住掉歪；深盒里的 naive 也有一次蓝方块掉到盒外（种子 2005） | 内腔放大到 19 × 12 cm，躺着的红块槽位往里 1 cm |
| 内壁深灰（0.05，默认材质，proto1） | 担心俯视时底板 / 假底边缘的明暗能看出深浅（深灰版本没有单独量过，【推断】） | 改成纯黑、`specular=0 shininess=0 reflectance=0` 的材质 → 实测两种盒子的渲染逐像素相同 |
| 盖子止挡 160–172°（draft0） | 开着的盖子自由端翘到 12–14 cm 高，搬运的物品会扫到它 | 止挡改到 182–192°：盖子略低于水平挂在箱旁，自由端 6.5–7 cm |
| 盖板默认接触参数 | 手臂压盖子时盖板穿进红块（红块从盖子里"长"出来） | 盖板硬接触 + `solmix 1000` + 压盖时手臂退让 |
| `in_box` 用物品中心 + 固定边距 | 掉进深盒后斜靠在前壁的蓝方块（倾 52°）被判在盒外 | 改成碰撞盒各顶点的投影在内壁内（容差 0.4 cm）、顶点不高于盒沿 0.4 cm |

## 已知问题 / 推测（未验证的标【推断】）
- **交互后深度就能看出来**：放第一件东西进去时就能看出深浅——4 cm 蓝方块在深盒里完全没入（顶面低于盒沿 4.4 cm），在假底盒里顶面几乎齐沿；立着的红块在假底盒里直接高出盒沿 3 cm。这是有意的（R2 只要求动手前看不出）；【推断】会看图的 agent 在放红块时就会发现，之后能否想到"放倒"是任务要测的；没回头看就关盖、finish 的在 attempt 级失败。
- **放倒的手段**：脚本用闭合手指推块的上部。【推断】agent 也可能用别的办法（夹着块在盒沿上放斜、从高处松手让它倒），没有测这些；判定只看终态，不限手段。
- **盖子的顺从是接口辅助**（和 `lh_lid_box` 一样）：夹着把手往任意方向 move 都会沿圆弧转盖子；盖子搭在物品上时手臂不会把它压穿，agent 夹着把手往下压会"压不动"（`stopped_from_target` 大）。
- **按盖子不会合上**（实测，见上表），所以"再压一次"这条最便宜的修法是错的（R4）。
- **`robot_touching`**：finish 时手还压着盖子或物品算失败，防止用手把盖子按住算"关上"。
- **把手只在 yaw −90 夹**（脚本，实测）：第一轮验证里 adaptive 在种子 1004 的尝试内修复失败（19/20）：躺着的红块长轴沿 y，夹它的 yaw 按数值噪声落在 +90 而不是 −90，之后夹把手也沿用 +90，手臂去够挂在左侧止挡上的把手时停在离把手 2.7 cm、低 1.6 cm 处（顶在盖子上），两次都没夹住。改成 `_grasp_knob` 固定 yaw −90（从 +90 过来先转到 0，避免 180° 的腕转），1004 重跑尝试内修复成功；表里的数字是改后重跑的。
- **低空搬运会把立着的红块碰倒**（实测，blind）：blind 的搬运高度随机到 13–15 cm 时，夹着的蓝方块底面（约 11–13 cm）低于高出盒沿的红块顶（12 cm），扫过去把红块撞倒在盒里——倒下就放得下，盖子反而合上了（种子 1000 第 3 次，`item_relaid where=box cause=robot`）；另一次（种子 1018 第 3 次，搬运 13.1 cm）红块底面低于盒沿，撞在前壁上掉到盒外。两者都是物理，不是判定漏洞；【推断】真实 agent 低空搬运时也会出现，`item_relaid` 事件能区分"有意放倒"和"撞倒"（看 `cause` / `where` 和之前有没有取出）。
- **并行构建的竞态**（公共代码，没改）：`register_generated` 每次构建都重写 `assets_gen/*.xml`，两个进程同时构建同一任务可能读到半个文件；错开几秒启动即可。
