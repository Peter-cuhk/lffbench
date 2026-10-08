# L6-C `l6_sequence_resume`：按顺序逐个称重四件物品，中途急停把一部分工作复位

> 2026-10-07。"实测" = 本机仿真跑出来的；"推测" = 没验证。

**任务 → 隐藏信息 → 所需能力 → 调整后的动作**
- 任务：按指令给的顺序逐个称重四件小盒装食品：放到秤上、松手（秤记下读数并"嘀"一声），再放回它自己的圆垫，然后才拿下一件；每件只能称一次（秤的记录不能改）。
- 隐藏信息：搬运途中触发一次急停：夹爪张开，安全程序把手里那件放回它的垫子，机械臂回到初始位姿。急停时已经称了几件（w ∈ {1..4}），以及手里那件是正要去称（A 相，还没称）还是称完正往回放（B 相，已经称了）。急停后每件都在自己垫子上、手臂在初始位——画面和开局一样。
- 所需能力：Save（记下秤每次"嘀"的是哪件）、Retrieve（急停后取回）、Utilize（判断被打断的那一步有没有完成：A 相要重称这一件，B 相直接称下一件；w = 4 时什么都不用做，直接 done）；Perceive（认出四件物品）。
- 调整后的动作：从正确的下一件继续，而不是从第一件重来（重复称 → 不可挽回）或凭感觉跳着来（顺序错 → 不可挽回）。

| 项 | 内容 |
|---|---|
| 类别 / 等级 | procedural_memory / L6 |
| 能力 | Perceive, Save, Retrieve, Utilize |
| 协议 | within：一个 episode 不重置，最多 5 次 done，粒度同 l6_swap_interrupt |
| 代码 | `lffbench/tasks/l6_sequence.py`，`SequenceResume`；验证 `python scripts/validate.py --task l6_sequence_resume --n 20 --k 5` |
| 场景 | 奶油奶酪、黄油、巧克力布丁、爆米花（LIBERO HOPE 扁盒，平放）各在一块相同的深灰圆垫（直径 11 cm，不参与碰撞）上；一台秤：浅灰 12 × 12 cm 平台、顶面高 2 cm，深色底座（静态 fixture）。垫子和秤的位置每个 seed 随机（垫间距 ≥ 16 cm，垫-秤 ≥ 17.5 cm，离基座 ≤ 0.70 m）；物品在垫上的偏移 ±0.8 cm、朝向 ±0.3 rad 随机；顺序随机 |
| 指令（直呼其名） | "Weigh the four grocery items one at a time, in this order: chocolate pudding, cream cheese, popcorn, then butter. To weigh an item, put it on the grey scale and let go of it; the scale beeps when it has recorded the reading. Then put the item back on its own round pad before you pick up the next item. Weigh every item exactly once: the scale logs every reading, and the log cannot be edited."（顺序按实例；`agent_instruction` 钩子） |
| 指令（间接描述） | 同上，物品换成外观描述：the pale blue box with the oval logo（奶油奶酪）/ the red box with white lettering（黄油）/ the dark brown box（巧克力布丁）/ the blue-and-yellow box（爆米花） |
| 秤的读数 | 物品中心在平台内、平放在平台上、没被夹着、静止 10 个控制步（0.5 s）→ 记一条读数并发事件 "The scale beeped: reading recorded."；物品离开平台后才能再次被记 |
| 成功 | 读数记录 = 指令顺序（每件恰好一次），且四件都在自己的垫上（中心离垫心 ≤ 4 cm、平放、不在夹爪里） |
| 终止失败 | 记录不再是指令顺序的前缀（重复称或顺序错）：当场事件 "The scale beeped three times: error, …"，`outcome["terminal"] = True` |
| 失败注入（急停） | 每个 episode 一次，在第 1 段内。按状态触发：A 相 = 第 w+1 件被夹着、比静止高度高 ≥ 4 cm、离自己的垫心 ≥ 触发距离（5–9 cm），且此时记录恰是前 w 件；B 相 = 第 w 件刚称完（记录 = 前 w 件）、从秤上被夹起、离秤心 ≥ 触发距离。触发后：夹爪张开，这件被放回自己的垫子（偏移/朝向和开局同分布），机械臂被驱动回初始位姿（当前这条运动指令剩下的控制步都用来回家，指令报告 reached=False），事件文字 "Emergency stop! The gripper opened, the safety routine put the item the robot was holding back on its pad and moved the arm to its home pose. You may continue the task."（不说是哪件） |
| 隐变量 | w ∈ {1..4} 均匀；相位 A/B（w < 4 时各 1/2，w = 4 只有 B）；顺序、布局、触发距离、复位偏移和朝向 |
| 先验为什么失败 | 急停后的画面和开局几乎一样（四件都在垫上、手臂在家）。只看当前画面的 agent 不知道称到哪了：从第一件重来 → 第一件被称第二次 → 终止；猜一个进度 → 期望只有 1/5 对 |
| 要从失败里带走 | 秤已经"嘀"过哪几件（= 记录了几件），被打断的那一件是在称之前还是称之后 |

## 1. 尝试粒度和事件
- 尝试（segment）= 从当前状态开始操作直到 agent 宣布 done；脚本策略里 done 前把机械臂收回初始位姿（`go_home`）。done 后测 outcome、给 F0/F1/F2；没成功也没终止就从当前状态继续，最多 5 次。succ@j = 第 j 段结束时已经成功的 run 占比。
- 急停每个 episode 只注入一次，在第 1 段内部；oracle / adaptive 的 @1 包含"急停后在同一段内恢复"。
- 尝试内部的即时事件（`task.drain_events()`，harness / robot_server 附在当时那次工具调用的报告里，也进入 memory=full 时历史里的动作摘要）：
  - 每条正确读数："The scale beeped: reading recorded."
  - 读数不符合顺序："The scale beeped three times: error, this reading does not follow the requested order. The log cannot be edited."
  - 急停（见上表，不说是哪件物品）。
- `run_scripted` 返回的 history 与 cross 协议同格式：`params` 记每一步（物品、ok / stop_A / stop_B / no_grasp、是否听到嘀声），`outcome` 记成功、终止、记录、急停（物品、相位、当时已称件数）、不在垫上的物品和 F2 文本。
- 注意（实测代码行为）：`robot_server.py` 的 `_done` 只在成功或用完 k 次时结束 episode，不看 `outcome["terminal"]`；harness（`harness.py`）会在 terminal 时停。所以 Claude CLI 会话里终止失败之后 agent 还能继续操作，但结果不会再变（F2 每次都说无法完成）。

## 2. 反馈（F2 = 结论 + 记录是否完整 + 哪件不在自己的垫上（只给距离）+ 本段的急停事件；不报已称件数、不报坐标）
validate 实测原文（seed 1000–1019）：
- 成功（adaptive，seed 1001，w = 3 A 相）："Attempt succeeded. During this attempt: Emergency stop! The gripper opened, the safety routine put the item the robot was holding back on its pad and moved the arm to its home pose. You may continue the task. All four items were weighed once, in the requested order, and are back on their pads."
- 记录不完整（naive，seed 1005，w = 1，猜"4 件都称完了"直接 done）："Attempt failed. During this attempt: Emergency stop! … You may continue the task. Not every item has been weighed yet."
- 终止（blind，seed 1001，从第一件重来）："Attempt failed. During this attempt: Emergency stop! … The scale log does not follow the requested order (an item was weighed a second time or out of turn). The log cannot be edited, so the task can no longer be completed."
- 物品不在自己垫上时（代码格式，本次验证没有出现）："… Also: the butter is 7 cm from the centre of its pad; the popcorn is on the scale."——只给离自己垫心的距离，不给坐标、不给方向。

刻意不给的：已经称了几件 / 称了哪几件。F2 若说"已称 2 件"，就是裁判替 agent 记住了进度。"Not every item has been weighed yet" 只说明记录还没完整（非终止失败本来就意味着记录是正确前缀或有物品没放回）。

## 3. 脚本参照策略（"感知"用仿真状态）
| 策略 | 有什么记忆 | 急停后 / 之后的尝试 |
|---|---|---|
| oracle | 特权：秤的真实记录 | 总是称 order[len(记录)] |
| adaptive | 自己的读数记录：某件放到秤上后听到"嘀"才记上 | 从 order[len(自己的记录)] 继续：A 相重称被打断那件，B 相称下一件，w = 4 直接 done；之后的段按 F2 把没回垫的放回 |
| naive（1 次尝试） | 急停时丢失进度（只剩当前画面） | 画面看不出进度：在 0..4 里均匀猜"已称几件"，从那里继续 |
| blind（k 次尝试） | 无 | "不知道做到哪了，从头来"：从第一件开始重称 → 重复读数 → 终止 |
| skip（附加对照） | 数的是"开始称过几件"（把被打断那件也算上） | B 相正确；A 相跳过被打断那件——w < 3 时下一次读数顺序错（终止），w = 3 时第 1 段记录不完整，第 2 段按 F2 + 记忆补称被打断那件 |

## 4. headroom 验证（实测，n = 20，seed 1000–1019，k = 5，`results/l6_sequence_resume/validate.json`）
| 策略 | @1 | @2 | @3 | @5 | 验收线 |
|---|---|---|---|---|---|
| oracle | 100% | | | 100% | @1 ≥ 95% ✓ |
| naive | 25% | | | 25% | @1 ≤ 30% ✓ |
| adaptive | 100% | 100% | 100% | 100% | @5 ≥ 90% ✓ |
| blind | 0% | 0% | 0% | 0% | 明显低于 adaptive ✓ |

同一次运行的其他实测：
- 80 个 run 全部触发了急停，(w, 相位) 与采样一致；20 个实例的分布：w=1 A 3、w=1 B 1、w=2 A 4、w=3 A 7、w=4 B 5（A 相 14/20，抽样偏多；2000 个实例上 A 相 36.5%，w 各约 1/4）。
- 0 次抓取失败；oracle / adaptive 的 40 个 run 全部一次成功，记录都等于指令顺序。
- naive 的 5 次成功：3 次猜中 w（1001、1012、1017），2 次是 w = 4 时猜"已全部称完"直接 done（1000、1004）。15 次失败里 12 次终止（重复读数或跳号），3 次是猜"已称完"但其实没有（记录不完整，非终止）。期望值 1/5 = 20%。
- blind 20/20 在急停后的第一条读数就终止（重复称第一件）。
- 整次验证 1036 s（与另一个验证并行，8 核机器上还有别的进程）。

缩略图：`init_<seed>.png`（开局）、`naive_end_<seed>.png`（naive 结束）、`interrupt_<seed>.png`（急停后约 3 s 的 agentview | 腕部画面，见 §5）。

## 5. 对照（实测，`results/l6_sequence_resume/controls.json`）
| 对照 | 结果 | 说明 |
|---|---|---|
| skip（数"开始称过几件"而不是读数） | @1 30%，@2 65%，@5 65% | @1 的 6 个成功全是 B 相；7 个 w = 3 的 A 相在第 2 段读到 "Not every item has been weighed yet" 后补称被打断那件（@2）；其余 7 个 A 相（w = 1、2）跳号 → 终止。2000 个实例上的期望终值 75.5%（这 20 个种子 A 相偏多） |
| 关掉急停（`task.inject = False`），前 10 个种子 | adaptive 100%，naive 100% | 不打断时任务本身对脚本策略没有难度，失败全部来自急停后丢失进度 |

急停后的画面：`results/l6_sequence_resume/interrupt_<seed>.png`（急停后 60 个控制步的 agentview | 腕部）。和 `init_<seed>.png` 对比（seed 1001，w = 3 A 相，实测）：四件物品都在自己的垫上、手臂在初始位，唯一差别是被复位那件的朝向/偏移重新抽了一次（与开局同分布），不知道开局朝向就看不出来。

演示视频：`videos/l6_sequence_resume_adaptive_s1001.mp4`（w = 3 A 相：称完三件，第四件去秤的路上急停，从自己的记录恢复，重称第四件，成功）；`videos/l6_sequence_resume_blind_s1001.mp4`（同一实例，急停后从第一件重来 → 黄油被称第二次 → 终止）。

## 6. 无记忆策略最多能拿多少（解析）
假设执行完美，只算"急停后从第几件继续"：

| 无记忆策略 | @1 | 来源 |
|---|---|---|
| 从第一件重来（= blind） | 0% | 构造：急停前至少称了一件 |
| 在 0..4 里均匀猜已称件数（= naive） | 20%（期望）；这 20 个种子实测 25% | 解析 / validate |
| 固定猜一个最常见的进度 | 26%（2000 个实例） | `controls.json` |
| 直接 done（赌已经全称完） | 25%（w = 4 的比例） | 解析 |
| 记得 F2 但不记得进度：先 done，F2 说没称完再在 1..3 里猜 | @2 约 25% + 75% × 1/3 = 50% | 解析；猜错就是终止，所以没有"安全试探" |

- 和 l6_count_into_box 不同，这里任何一次错误读数都是终止，所以不存在"每段只做一点、靠成功即停来探"的无状态捷径。
- 最后一行需要跨段记住 F2，本身也是记忆；memory=none 的新 agent 连这个都没有，实际会更接近第一行（推测：画面像开局时，最自然的做法是从头开始）。

## 7. 设计决定和理由
1. **急停把手里那件放回它自己的垫子，并让手臂回家。** 草案是"物品掉回自己垫子附近"：那样掉在垫外的物品会告诉无记忆的 agent 刚才拿的是哪件，再假设按顺序做，就只剩 A/B 相要猜（50%）。放回垫上（偏移/朝向和开局同分布）后画面里不再有进度线索。手臂回家同理：停在半路的手臂朝向秤还是朝向垫子会泄露相位。
2. **相位是真正要记的东西，不只是"做了几步"。** A、B 两相画面一样，下一步却不同（重称这件 / 称下一件）。只数"拿起过几件"的 agent（skip 对照）在 A 相会跳步。
3. **w 均匀、4 件物品。** 3 件时无记忆"猜一个进度"的最好情况是 1/3（w 均匀）；4 件 + w ∈ {1..4} 均匀 → 1/4，均匀乱猜（含 0）→ 1/5。w = 4（只差最后一件放回、急停已代劳）保留：考 agent 能否认出"已经做完了"。
4. **秤的"嘀"声是即时事件，不进 F2。** 它是 agent 当时能感知到的东西（相当于听到声音），所以有记忆的 agent 能确认 B 相前那次读数确实记上了；F2 只汇总急停，不汇总嘀声次数（否则裁判替它数了）。
5. **记录错误当场报警并终止。** 真实的秤/扫码台在重复扫描时会报警；继续操作也无法挽回。
6. **急停实现**：在 `env.step` 外包一层（同 l3_wall 的做法），急停期间把动作替换成"回到初始位姿 + 夹爪张开"，直到当时那条 `Skills` 原语返回（最多 250 个控制步，约 12.5 s 仿真时间）。所以 VLM agent 那条 move_to 会报 reached=False 并带急停事件。

## 8. 已知问题 / 效度威胁
实测到的：
- **每段命令预算不够。** CLI 自测：称一件（张夹爪、到上方、下降、夹、抬起、到秤上方、下降、松手、抬起）用了 9 条命令，放回垫子再要约 6 条，一件约 15 条；四件 + 急停后重做 ≈ 60–75 条。`robot_server` 默认 `--max-steps 40`、`HarnessConfig.max_steps` 默认 30，第 1 段必然被预算截断（截断 = 自动 done，within 协议下还能继续，但 succ@1 会被压低，memory=none 时还会在截断处换一个没有记忆的新 agent）。建议这个任务跑 VLM 时把每段预算设到 100。没改共享脚本。
- `robot_server.py` 不看 `outcome["terminal"]`（只有 harness 看），CLI 会话里终止之后 agent 还能继续操作，结果不变。
- 急停是传送：物品瞬间回到垫上，视频里会"跳"一下；手臂回家是真实运动（被打断的那条 move_to 会占满 250 个控制步、报 reached=False）。
- 秤要物品静止 0.5 s 才记录。CLI 自测里"松手"那条命令（夹爪保持 15 个控制步）结束时已经嘀了；如果 agent 用别的方式放（例如松手后立刻重新夹住）可能没记上，要靠"有没有听到嘀声"来判断——这是任务的一部分。
- 这 20 个验证种子里 A 相 14 个（期望约 37%），skip 对照因此偏低；不影响四个验收策略（它们和相位无关地 100% / 0%）。

推测、未验证的：
- 全历史在上下文里的强 VLM 大概率能记住嘀了几次（事件文字在每条命令的报告和历史摘要里），难点更可能是 B 相：急停文字说"把物品放回了垫子"，容易被理解成"这件没称完"而重称（→ 终止）。
- 要让它更依赖 Save（而不是在长上下文里回看），建议和 l6_swap_interrupt §8 一样在 harness 层做"有限历史 + 自己的笔记"条件；或把急停放在一段的结尾（需要 harness 支持任务侧结束一段，目前没有这个钩子）。
- 间接指令下四个小盒在 512 px agentview 里只有 20–30 px，靠外观认物品可能是主要难点之一。

## 复现
```bash
cd /mnt/cpfs/workspace/agentic-robotics/lffbench && source run_env.sh && export OMP_NUM_THREADS=1
python scripts/validate.py --task l6_sequence_resume --n 20 --k 5 > logs/validate_l6_sequence.log 2>&1
python scripts/make_video.py --task l6_sequence_resume --kind adaptive --seed 1001
python scripts/l6_controls.py l6_sequence_resume 20   # 对照 -> results/l6_sequence_resume/controls.json
```
