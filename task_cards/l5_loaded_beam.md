# L5 `l5_loaded_beam`：提起两端放着小方块的金属梁并保持水平（`l5_slippery_lift` 的加深版）

代码：`lffbench/tasks/l5_loaded_beam.py`（子类化 `SlipperyLift`，旧任务不动）　结果：`results/l5_loaded_beam/validate.json`（n = 20，k = 3）、`results/l5_loaded_beam/validate_k5.json`（n = 20，k = 5，和 v1 的 k = 5 对齐）

> 2026-10-12 新增，就地替换已饱和的 `l5_slippery_lift`（设计依据：`research_notes/P1P2_加深与常识任务设计.md` §1、§2、§3.1）。下文没标的数字都是本机实测；**【推断】** 表示没验证。

| 项 | 内容 |
|---|---|
| 类别 / 等级 | dynamics_adaptation / L5（"搞清楚这个物体怎么动"），`level="L5"`，和 `l5_slippery_lift` 同类 |
| 能力 | Perceive（看出哪头沉、方块掉了没有）、Reason（沉的一头 = 重心那一侧）、Plan（放下 → 捡回方块 → 重抓）、Utilize（下一次直接抓在合重心处） |
| 协议 | cross（每次尝试重置到同一实例），k = 3（`max_attempts = 3`）；另跑了 k = 5 的 validate 以便和 v1 比 |
| 指令（direct） | Lift the metal bar together with the two small red blocks resting near its ends, and hold it level (tilted less than 8 degrees) with its lowest point at least 10 cm above the table. Both blocks must stay on the bar near its ends. Keep holding the bar when you finish. |
| 指令（indirect） | Raise the long grey rod with the two little red cubes sitting near its ends and hold it horizontal (…)… Both cubes must stay on the rod near its ends. … |
| 场景 | v1 的灰色金属梁（4 × 28 × 4 cm，长轴沿机器人左右方向，同一 seed 同一位置、同一隐藏配重）；梁顶面两端各放一个红色小方块（2.5 cm 立方、30 g），中心在梁中点左右各 12.5 cm。两个方块是两个不同的生成类别（`LffBeamBlockA/B`，LIBERO 同一生成类别实例化两次会报错） |
| 隐变量 | 梁内配重：重心偏离中点 e，\|e\| ∈ [2.5, 7] cm，偏左或偏右；梁 0.8–1.2 kg，指–梁摩擦 μ 0.40–0.60（全部沿用 v1）。梁 + 两块方块的合重心在 M·e/(M + 2m) 处（2.3–6.7 cm） |
| 方块摩擦 | 0.2（geom priority 3：方块对梁、对桌面、对手指都用 0.2）。梁倾斜超过约 11° 方块就滑。理由见"物理标定" |
| 成功 | v1 条件（保持 1.5 s 后仍夹着、倾角 ≤ 8°、最低点离桌 ≥ 10 cm）**且**两块方块都在梁上、各在一端原位 ±3 cm 内（哪块在哪端不限） |
| 预算 | `agent_max_steps = 40`（同 v1 和 l3_card_shelf），动作时间 120 s |

## 为什么 v1 饱和、这一版改了什么
- v1 实测（cla4，Sonnet 5.5，seed 3000）：两个 run 都在第 1 次尝试里中点抓 → 提起 → 看到倾斜 45–60° → 放下 → 改抓 → 提平，分别用了 11 条和 19 条命令。倾斜一次没有任何代价（规则 R3 不满足）。
- 这一版让"第一次按常识提起来"有代价：梁一歪，低端（重端）的方块滑落到桌上，高端的方块滑到手指边。要修就得放下梁、松手、从桌上捡回方块放回梁端、再把另一块推回/搬回原位、重抓、再提。脚本修一次要 20 条命令，v1 同样的"放下重抓"只要 7 条。
- 学习信号：沉的那一头就是重心那一侧（方块从哪头掉的也直接指向重端）；跨尝试要学的是"一开始就抓在偏重端的合重心处"，或者"先只提 1 cm 看往哪边歪"。

## 物理标定（实测，2026-10-12，20 个 seed：1000–1019，中点横抓，脚本在 scratchpad `l5_loaded_beam/t_basic.py`、`t_mu.py`）
| 做法 | 方块摩擦 = 梁的 μ（0.43–0.60，原型设计） | 方块摩擦 0.2（最终版） |
|---|---|---|
| naive：提 15 cm，0.1 m/s | 20/20 低端那块落桌（倾斜 31–45°） | 20/20 落桌（倾斜 29–53°） |
| 慢提：15 cm，0.03 m/s | 20/20 落桌（27–47°） | 未测（【推断】只会更容易落：力矩是静态的，方块更滑） |
| "半提"：5.5 cm（Sonnet 在 v1 N run 的习惯），0.03 m/s（左）/ 0.05 m/s（右） | **0/20 落**（19–21°） | **20/20 落**（20–22°） |
| 试提 3 cm | 0/20（8.5–10.6°） | 2/20 落，另 1/20 方块滑出原位 3 cm（10–11°） |
| 试提 1.5 cm | 0/20（2.5–4.5°） | 0/20（2.5–4.5°） |
| 试提 1 cm | 0/20（0.4–2.5°） | 0/20（0.6–2.5°） |
| oracle：抓合重心，提 15 cm | 0/20 落，20/20 成功（倾斜 0°） | 0/20 落，20/20 成功 |

- **为什么把方块摩擦降到 0.2**：方块摩擦等于梁的 μ 时，方块要倾斜 22–31° 才滑，"提 5 cm 看一眼"是免费的试探（0/20 落），等于又给谨慎的 agent 留了 v1 那条零代价的路。降到 0.2 后，只有刻意的 ≤ 1.5 cm 试提是安全的，这正是要学的东西。
- 倾斜时低端那块从梁端滑下、掉到桌上；梁放回桌面后，它的中心在梁端外 1.8–5.1 cm（20/20 都在桌上、能捡），高端那块停在离梁中点 0.5–4.2 cm 处（紧挨手指）。
- 梁的隐藏重心沿用 v1 的 `set_hidden_weight`（已调用 `envs.refit_bounds`）；方块是普通生成盒子，运行时只改了摩擦和 priority，没有改尺寸和惯性，不需要 refit。
- **LIBERO 放置失败修正**：原型的方块区域（梁区域旁 2 cm）在 400 次 `env.reset()` 里失败 13 次（"Cannot place all objects"）。原因：LIBERO 的区域采样器要把物体的水平半径（梁 14.1 cm）"装进" 2 cm 的区域，结果梁的采样范围变成区域中心 ±13 cm，方块区域落在这个范围的碰撞半径内。方块区域挪到 y = ±0.33 后，400 次 0 失败。`apply_instance` 随后会把所有物体移到实例位置，所以这不影响布局。

## outcome 与事件（只进服务器日志，agent 只看到 F1）
- 每个控制步的监视器（`BeamSkills._act`）记录：`bar_lifted`（梁中心比静置高 2 cm 以上且被夹着；1 cm 试提不算）、`bar_tilted`（这次提起中首次 \|倾角\| > 8°）、`load_fell`（方块不经夹爪离开梁落到桌上 = 陷阱触发）、`load_displaced`（方块还在梁上但离原位 > 3 cm）、`load_removed`（agent 夹起方块后放到桌上）、`load_restored`（离开过原位的方块回到某一端 2 cm 内；有滞回，防止在 3 cm 边界抖动；`by_robot` = 期间被夹爪夹过，否则是物理自己滑回去的：高端那块在另一块掉落、梁回摆时偶尔滑回原位，最终 k = 3 validate 的 blind 59 次尝试里 4 次）、`bar_set_down`。方块状态去抖 5 步。
- 字段：`success`、`failure`（tilted / too_low / load_off / dropped / no_grasp）、`tilt_deg`、`bottom_cm`、`grasp_along`、`stages`（bar_held_level_high、loads_on_ends）、`loads`（每块的状态和沿梁位置）、`trap_fired`、`n_load_falls`、`n_restored`、`n_restored_by_robot`、`n_regressions`、`recovered_in_attempt`（触发过陷阱且最终成功）、`first_interaction_ok`（第一次提起没有掉块）、`lifts`（每次提起的抓点、最大倾角、掉了哪块）、`events`、`detail`（F2 文字，机器人视角方向词）。
- F2 示例（smoke test 第 1 次，实测原文）："The bar did not stay level: it rotated in the gripper and now hangs tilted by 48 degrees, its left end lower than its right end. Its lowest point is 4 cm above the table. The left block slid off the bar's left end while the bar hung tilted by up to 48 degrees with its left end lower. The left block lies on the table, 8 cm beyond the bar's left end. The right block is still on the bar but 4 cm in from its right end position (8 cm right of the bar's middle). You last grasped the bar 0.2 cm to the right of its middle."

## 脚本策略（每次尝试按 agent 命令计数，预算 40：一条 move_eef（可带夹爪动作）或一次 observe = 1 条）
- **oracle**：在合重心处横抓，0.1 m/s 提 15 cm，看一眼。5 条命令。
- **naive**（Sonnet 的实测习惯）：中点横抓，0.1 m/s 提 15 cm，看一眼，结束。5 条。
- **adaptive**：第 1 次同 naive，但提起后检查；倾斜或掉块就修：放下松手（2）→ 两块方块逐个夹起放回梁端（各 6）→ 看（1）→ 往下沉的那一头重抓再提（5），共 20 条。重抓点用 v1 的夹逼法（只有一侧有界时往重端挪 3 cm，两侧有界取中点，边界留 0.8 cm 余量，限制在 ±9 cm 内免得手碰到方块）。剩余命令不够一次完整修复（20 条）就停手，带着歪梁结束。第 2 次起直接用目前学到的抓点。
- **blind**：naive 加噪声，无历史（抓点 ±1 cm 正态、提 12–18 cm、速度 0.05–0.2 m/s）。

## headroom 验证（实测，n = 20，seed 1000–1019）
`python3 scripts/validate.py --task l5_loaded_beam --n 20 --k 3` → `results/l5_loaded_beam/validate.json`

| 策略 | @1 | @2 | @3 | 第 1 次掉块 | 备注 |
|---|---|---|---|---|---|
| oracle | 100% | 100% | 100% | 0/20 | 5 条命令 |
| naive | 0% | 0% | 0% | **20/20** | 倾斜 29–54°，全部 `tilted` |
| adaptive | 55% | 100% | 100% | 20/20 | 第 1 次都修了一次（共 25 条命令；放回方块 40 次全部由夹爪完成），11/20 在第 1 次内修好并成功（`recovered_in_attempt`）；另 9/20 第二次提起（往重端挪 3 cm）仍歪、又掉块，预算不够再修，第 2 次用夹逼的抓点成功 |
| blind | 0% | 5% | 5% | 20/20 | 59 次尝试里 58 次掉块 |

k = 5（`results/l5_loaded_beam/validate_k5.json`，同 20 个 seed）：oracle 100/100/100/100/100%，naive 0/0/0/0/0%，adaptive 55/100/100/100/100%，blind 0/5/5/5/5%（97 次尝试 96 次掉块）。

同一 seed 两次运行并不逐位一致：两次 validate 之间 naive 第一次提起的最大倾角 16/20 完全相同、4/20 差 1–5°（【推断】来自 LIBERO 每次 reset 的随机放置留下的接触 warmstart；v1 同样）。adaptive 第二次提起落在 ±1.5 cm 窗口边上的几个 seed 会因此翻转：本任务各次运行的 adaptive@1 在 50–60% 之间（开发时 50%、60%，最终两次都是 55%）。

**v1 对照（同 seed 1000–1019）**：
- `l5_slippery_lift` 自己的 validate（今天 refit 之后重跑，n = 30，k = 5）：naive 0%（这 20 个 seed 0/20，倾斜 31–56°），adaptive 0/47/100%（跨尝试夹逼，每次只提一次）。
- 用**同一套脚本策略**跑去掉负载的同一根梁（方块挪到桌边、不计入成功，物理即 v1）：naive 0/20；会在尝试内"放下重抓"的 adaptive 第 1 次就 **20/20** 成功，用 12 条（10 个）或 19 条（10 个）命令，提 2–3 次。这正是 v1 饱和的机制。同一策略在 loaded_beam 里修一次要 20 条，40 条预算只够修一次，@1 降到 55%。

## 其他实测
- **oracle 抓点 ±5 mm**（沿梁 ±5 mm、横向 ±5 mm，各 20 个 seed）：80/80 成功，0 次掉块，最大倾角 0.1°（四种偏移各 20/20；脚本 `t_robust.py`）。
- **可见性**：掉落的方块在 agentview 里清楚可见（红色方块在灰梁低端外的桌面上，见 `results/l5_loaded_beam/naive_end_1000..1002.png`；smoke test 的 `a1_s004_agentview.jpg` 同样）。滑到手指边的那块在腕部相机里几乎占满画面（smoke test 实测）。
- **pilot seeds**（≥ 3000 中 naive 掉块失败、oracle 成功的前 3 个）：**3000、3001、3002**（3003–3005 也满足）。seed 3000 和 v1 cla4 用的是同一个实例（e = +3.98 cm，μ 0.43，0.91 kg），合重心 +3.73 cm。

## agent 接口 smoke test（实测，2026-10-12，la 风格，rgb，F1，k = 3，seed 3000，端口 12640）
`python3 scripts/claude_batch.py start --batch smoke_l5_loaded_beam --model sonnet --only harness --k 3 --perception rgb --style la --feedback F1 --port0 12640 --tasks l5_loaded_beam:3000`
- README 只有通用说明、任务指令、预算（40 条命令、120 s），没有重心、质量、摩擦。
- 第 1 次（4 条 move_eef）：中点横抓（yaw 90），0.1 m/s 提到 z = 0.17 → agentview 里梁左端下垂 48°、左边方块落在桌上、右边方块滑到手指边；`finish` → 只回 "Attempt failed."。服务器日志：`trap_fired = true`、`load_fell`（block_1）、`load_displaced`（block_2）、`first_interaction_ok = false`。
- 第 2 次：抓在中点左 3.7 cm（合重心）→ "Attempt succeeded."，episode 结束。之后 `claude_batch.py stop` 停掉了服务器。

## 已知问题 / 推断
- **谨慎的试提仍然免费**：刻意只提 1–1.5 cm 的 agent 能在第 1 次尝试里零代价看出往哪边歪（实测 0/20 掉块）。这是有意保留的"可学的解法"；【推断】强模型第一次就这么做的比例不高（cla4 里 Sonnet 一次直接提到 20 cm、一次提 5.5 cm），但需要小试确认。试提 3 cm 已有 2/20 掉块，是灰区。
- **尝试内修复是允许的**：40 条预算够修一次（脚本 20 条）。【推断】Sonnet 的命令更碎（v1 里每次下降分 3–4 步），修一次可能要 30 条以上，第 1 次尝试里修好的比例会低于脚本的 55%。论文里要和 `trap_fired`、`recovered_in_attempt` 一起报，不能只看 @1。
- 方块摩擦 0.2 也作用于手指：最终 validate 里脚本修复 20 次，放回方块 40 次全部成功（`n_restored_by_robot` = 40），没见到夹不住；【推断】agent 夹桌上方块时如果手指先碰到它，方块容易被推走（摩擦低）。
- `grasp_along` 记的是合爪那一刻的位置。v1 的 H run 里"夹着放下"会让梁在指间滑动重新落座，之后的真实抓点会变，日志里的抓点不变（v1 同样）。
- 两块方块外观一样，成功判据不区分哪块在哪端。agent 也可以先把方块拿下来、单独找梁的平衡点、再放回去（合法，多花约 12 条命令）。
- 质量 / μ 的全范围组合没逐一扫，只扫了 20 个 seed 自带的组合（μ 0.43–0.60，0.8–1.2 kg）。
