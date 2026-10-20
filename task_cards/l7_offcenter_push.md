# L7 `l7_offcenter_push`：把重心偏的长盒子推进目标框（10-19 前：一次推进）

代码：`lffbench/tasks/l7_offcenter_push.py`　结果：`results/l7_offcenter_push/validate.json`　视频：`videos/l7_offcenter_push_adaptive_s1000.mp4`

> **2026-10-19 肘关节上限修改后重跑 validate（实测）**：共享层 `lffbench/envs.py` 把第 4 关节上限收紧到 −0.55（任务侧会话改，用户同意）。重跑 `validate.py --n 20 --k 5`：80 个 run 里只有 adaptive seed 1010 变了。下文"修不好的 1 个（seed 1010）"原来是手指爬上盒子、之后每条 move 跑满 12–20 s、预算耗尽；现在第 1 次尝试 4 推、15 条命令、14.2 s 修好（first_push_yaw 36.1°，2 次错误事件）。**adaptive @1 95% → 100%（20/20 都是尝试内修好）**，其余策略和 run 逐项相同。推断：旧失败是肘关节卡死造成的。改前的 validate.json 副本在本会话 scratchpad（`elbow/`）。
>
> **2026-10-19 措辞修改（同日第二次改，用户同意）：指令只描述最终状态。** 规则、判定、场景都没改。
>
> | | 改前 | 改后 |
> |---|---|---|
> | direct | `Slowly push the long brown box straight forward (away from the robot) with the closed gripper so that it ends up completely inside the green rectangle, without turning.` | `Slowly push the long brown box with the closed gripper so that it ends up completely inside the green rectangle ahead of it (farther from the robot).` |
> | indirect | `Slowly push the long cardboard-coloured block straight forward (away from the robot) with the closed gripper so that it ends up completely inside the marked green area, without turning.` | `Slowly push the long cardboard-coloured block with the closed gripper so that it ends up completely inside the marked green area ahead of it (farther from the robot).` |
>
> - **判定实际要求（核对代码，没改）**：成功 = 结束 settle 30 步后盒子四个角都在框内（`box_fit` 的 inside）、upright cos > 0.95、没有一次推动是张爪推的。**没有单独的角度阈值**：转角上限由"完全在框内"的几何决定（盒子居中时 10.9°，偏离框中心时更小）。失败码里 |yaw| > 6° 记 `turned`、否则记 `position`，这只是标签，不影响成败。判定也不看推的方向、路径和速度。
> - **理由**
>   1. "without turning" 读起来像过程约束（推的途中不许转），但判定只看最终状态。10-19 起推歪后当场纠正、转回来进框也算成功（IR）；按字面理解，这种成功违反了指令，agent 会以为推歪了就没救、不去纠正（§2.1：判据要能从指令推出来；指令比判据严，同样会误导 agent）。"completely inside" 已经隐含了转角上限，所以删掉这句不会让判据推不出来。
>   2. "straight forward" 也是过程约束，判定同样不看。实测能修好的纠偏是**绕到盒子前面把领先端往回推**（朝机器人方向推），以及推端面（侧向推）；只从后面往前推的纠偏 0/25 修好（见下面"纠正代价"）。按字面"只许往前直推"的 agent，正好会被引到修不好的做法上。原句 "(away from the robot)" 带的方位信息（框在盒子正前方）保留了下来，改成描述框的位置："ahead of it (farther from the robot)"。
>   3. 保留 "Slowly"：判定不看速度，但这只是建议，不排除任何成功的做法（脚本 oracle / adaptive 都用 8 cm/s 推）；去掉反而少了防"推太快滑过头"的提示（滑过头和隐变量无关，见末尾【推测】）。保留 "with the closed gripper"：判定强制要求（`open_gripper`）。
>   4. 不提重心，不提推哪个位置。H+ 臂的说明（`arm_specs/l7_offcenter_push.py` 的 `reveal`）是接在指令后面的，不用改（seed 3000 生成的 H+ README 已核对）。
> - **影响（实测）**：脚本策略不读指令。按新代码重跑 validate（n = 20，k = 5，189 s，输出写到临时目录），结果和 10-19 的 `validate.json` 完全相同（80 个 run 的全部记录、6 张缩略图都一样），所以没有覆盖原文件。用 `RobotSession` 生成 seed 3000 的 README（la，rgb，F1，k = 3，direct / indirect / hplus 各一份，没有起 HTTP 服务器，也没有 agent），和 `selftest_20261019` 里的 README 逐行 diff，只有 "## Task" 下面那一行不同。任务代码的哈希变了，所以旧 run 在 `api_batch.py contract` 下会被标出来（这些旧 run 本来就要因为 10-19 规则修改重跑）。
> - **对照臂重跑（`arms/validate.json`，10-19 代码，主表 3000–3004，实测）**：7 项 × 5 个实例全部符合预期：orig_naive F 5/5、orig_oracle S 5/5、h0_naive S 5/5、twin_naive F 5/5、twin_oracle S 5/5、cross_twin_on_orig F 5/5、cross_orig_on_twin F 5/5。每项都只推了 1 次，推动开始时夹爪开口 0.001 m（合爪），`n_open_gripper_pushes` 全为 0。成败、转角、中心偏差、触点、反馈文字和 10-12 的旧结果逐项相同（每项都是推一次，新规则不改变推一次的物理过程），seed 3000 的三张结束图也和旧图逐字节相同（没有覆盖）。naive 转角：orig 25.6 / −28.9 / −27.1 / 30.0 / −32.8°，twin −25.3 / 28.6 / 27.2 / −30.4 / 32.4°。旧文件备份为 `arms/validate.json.bak-20261019-wording`。脚本是别的会话写的 `arms/validate.py` 的副本（原文件没改），加了新字段：n_pushes、n_open_gripper_pushes、trap_fired、每次推动开始时的开口 `push_gripper_width_m`；这个副本放在本会话的 scratchpad（`wording/arms_validate.py`，临时目录，没有放进仓库）。
> - **主表 5 个实例（和 validate 走同一条代码路径，`--seed0 3000 --n 5 --k 5`，实测）**：naive 5/5 第 1 次失败（第一推转 25.5、−28.9、−26.9、30.0、−32.7°，失败码 turned）；oracle 5/5 第 1 次成功（4 条命令）；adaptive 5/5 在第 1 次尝试内修好（11 / 11 / 11 / 11 / 15 条命令）；blind 5/5 一直失败；所有推动都是合爪推的。和下面 10-19 的记录相同，主表种子不用换。
> - **备份**：`lffbench/tasks/l7_offcenter_push.py.bak-20261019-wording`、`task_cards/l7_offcenter_push.md.bak-20261019-wording`、`results/l7_offcenter_push/arms/validate.json.bak-20261019-wording`。

> **2026-10-19 改（用户定）：不再限制每次尝试的推动次数。** 原来的"每次尝试只能推一次"是为了不让 agent 在一次尝试里边推边纠偏，等于把尝试内的试错硬挪到尝试间（违反 `DESIGN_INSIGHTS.md` §1"尝试内和尝试间的试错都算"）；做法照 `l7_key_turn` 10-16 的先例（commit 609c33f）。论文里这个任务现在归 L5（物理误判），代码里 `level` 仍是 "L7"。下面是 10-19 的新规则和实测；本节以下的旧内容写于 10-07，凡是和"只推一次"有关的地方都标了"10-19 前"。
>
> **新规则**
> - 指令（direct）：`Slowly push the long brown box straight forward (away from the robot) with the closed gripper so that it ends up completely inside the green rectangle, without turning.`（删掉了 "Use a single push: …" 那一句，其余不变；indirect 版同样处理）。**这是 10-19 第一版的措辞；同一天又删掉了 "straight forward" 和 "without turning"，见最上面一节。**
> - 推几次都行。**成功** = 尝试结束（settle 30 步）时盒子四个角都在框内、没翻倒（upright cos > 0.95），且**所有推动都是合爪推的**（新增，见"漏洞检查"）。旧的 `several_pushes` 失败没有了。
> - "一次推动"沿用原定义：某条 move_to 让盒子移动 > 1 cm 或转 > 3°，记进 `box_moves`。每条记录现在还有：推完后的绝对转角 `yaw_deg`、本次转角 `turned_deg`、夹爪在盒子上的触点（盒子坐标 `touch_u`/`touch_v`，`face` = back / front / left_end / right_end）、`gripper_open`（这条 move 开始时手指张开 > 1 cm）、推完后的状态 `inside / over_turned / overshot / sideways / tipped`、`bad`、`new_trap`。
>
> **错误事件（只记录，不结束尝试）**
> - 框能容下的最大转角按几何算：盒子半宽 a = 4.5、半长 b = 11 cm，框半宽 6.5、半长 13 cm，转 θ 后外廓半宽 a cosθ + b sinθ ≤ 6.5 → **θ ≤ 10.9°**（`FIT_ANGLE`，盒子正好居中时；旧卡写的"约 9°"不准）。
> - 每次推完判一次：`over_turned` = |转角| > 10.9°（放在哪里都进不了框）；`overshot` = 有角越过框的远边；`sideways` = 有角越过框的侧边；`tipped` = upright cos < 0.5。**一次推动是 `bad`（错误事件）当且仅当推完后盒子 over_turned / overshot / sideways / tipped 之一**，也就是"再往前直推也进不了框"。
> - 和任务描述的差别（需要用户确认）：描述里写的是"转角超过框能容下的角度**或者没进框**"。我把"没进框"具体化为"越过远边或侧边"；**盒子只是还没推到（还在框后面、方向和侧向都没问题）不算错误**，否则分几次推到位的 agent 每一推都会记错误（实测 grok 独立 run 先推了 1.7 cm、1.2 cm 两小下，按字面都要算错误）。
> - outcome 新字段：`n_pushes`（旧名 `n_box_moving_commands` 保留，值相同）、`n_bad_pushes`、`n_turned_pushes`（bad 里因为转过 10.9° 的）、`n_trap_events`（从"好状态"——在框内或只是没推到——变成 bad 的推动次数，即陷阱新触发了几次；纠偏途中盒子一直是 bad 的推动不重复计）、`trap_fired` = n_bad_pushes > 0、`recovered_in_attempt` = trap_fired 且成功、`first_push`（第一推的完整记录）、`first_push_yaw_deg`（第一推后的转角，论文"第一次交互是否成功"用）、`first_interaction_ok`（第一推不是 bad）、`n_open_gripper_pushes`、`irrecoverable` / `irrecoverable_why`（条件 → 首次出现的控制步）、`n_move_commands`、`sim_s`。
> - `irrecoverable`（只记录，不结束尝试）：翻倒（某条 move 后 upright cos < 0.5，或结束 settle 后 < 0.95）；掉下桌面（中心低于桌面 2 cm 或出桌面范围 |x| > 0.5、|y| > 0.6）；出了可达范围（盒子中心 world x > 0.13 或 |y| > 0.40）。可达阈值是实测的：推高度（抓取点离桌 2 cm）下夹爪能稳定到 world x = 0.10（误差 < 1 cm），再远就停在 0.13–0.17 并往下塌；把没转的盒子从前面往回推，中心在 0.125（远面 0.17）还推得动，中心在 0.135 时手指会爬上盒子或把它推得更远；转 25°、中心 0.113（远角 0.20）的盒子也还推得回来。所以用中心而不是远角判（10-19 第一版用远角 > 0.12，太保守，已改）。框的远边在 world x = −0.055 … 0.055，所以要推过头 7.5 cm 以上才会被记成出可达范围。
>
> **漏洞检查（实测，2026-10-19）**
> - **张开夹爪推（漏洞，已补）**：张开的两指在背面形成相距约 8 cm 的两个接触点，推正中时只要 |e| 不超过约 7 cm 盒子就直走。验证集 20 个 + 主表 5 个实例，**张爪推正中一次 22/25 直接成功**（只有 e = −7.3、−7.0、−7.5 cm 的 1005、3001、3004 转了 7–16°）；合爪 yaw 0 是 0/25，合爪 yaw 90（单指接触）也是 0/25。指令里本来就写了 "with the closed gripper"，按 §2.1"判定比指令松、能钻空子的属于漏洞"，判定里加了一条：**只要有一次推动是张着手指推的（move 开始时开口 > 1 cm；合爪空夹时开口 0.002 m，捏住盒角时 ≤ 0.007 m），整次尝试判失败**；如果盒子本来在框里，失败码是 `open_gripper`。注意复位后手指是半开的（0.041 m），agent 不合爪直接推也算违规。这条漏洞 10-19 前就存在（只推一次时也能用），旧日志里没有任何成功的 attempt 用过张爪（grok 带历史第 3 次用过两次，那次本来就失败）。
> - **夹起来搬 / 夹住拖**：夹不住。盒子 9 cm 深、22 cm 长，夹爪最大开口 8 cm；夹爪只能竖直朝下（move_eef 只有 yaw），没法从 4 cm 高的方向夹。实测 yaw 0 / 30 / 45 / 60 / 90、夹正中 / 端头 / 盒角、抓取点离桌 2 cm 和 3.5 cm：`holding` 全是 False，抬起时盒子不离开桌面；夹住盒角往前"拖"10 cm 只会把盒子推转（25°），等于一次偏心推。**不是漏洞，不需要加判定。**
> - **压住盒顶往前拖**：盒子和桌面、盒子和手指的摩擦系数都是 0.5（盒子几何体的 priority 更高），压住顶面拖 10 cm，盒子基本不动（3 个实例 × 3 种压深）；压得很深（2 cm）时盒子会被翻起来立在侧面（记 `tipped`、irrecoverable）。不是漏洞。
> - **从侧面推**：推端面只能让盒子沿长边平移（推力线沿长轴，过重心，不转），对"往前推进框"没有帮助；脚本序列"先推左端面 4 cm 再在重心推到位"结果是侧向出框（sideways，失败）。不是漏洞。
> - **多次小推**：不理会重心、一直推正中，每推一段都会继续转（脚本序列：推到位后再推正中 2 cm，转角 30° → 38°）。小推本身不是漏洞；真正的变化是 agent 可以**先小推试探**（见下面"纠正代价"），这是放开后的预期行为，不靠规则堵。
>
> **纠正代价（实测，脚本 adaptive 和两个变体，验证种子 1000–1019 + 主表 3000–3004，命令按 la 接口计：一条 move_eef = 1 条，合爪跟第一条 move 一起发）**
> - 按常识推正中、推到位（4 条命令）后，盒子转 17–41°，横在框上。能修好的做法：**绕到前面推领先的（轻的）那一端往回推**（盒子转回来、几乎不前移），再在重端一侧从后面把盒子推进框。脚本 adaptive 第 1 次尝试内修好 19/20：在推正中之后**多 7–15 条命令（中位数 7、平均 8.9；即多 2–4 次推动）**，整次尝试 11–19 条命令、9.5–18.6 s 仿真时间，都在每次尝试 40 条 / 120 s 的现有预算内。修不好的 1 个（seed 1010，框离得最远，12.2 cm）：前推时手指爬上盒子（停在目标 7 cm 外、抬高 3 cm），之后每条 move 都要跑满 12–20 s，14 条命令就用完 120 s 运动预算；把预算放到 200 条也修不好（47 条命令、612 s）。第 2 次尝试在学到的接触点推一次就成功。
> - **只从后面纠偏（推落后的一端往前）修不好**：这种推法转回来很慢、同时把盒子往前带，0/25 修好（20 个验证实例 + 5 个主表实例），其中 11 个最后出了可达范围（中心 x > 0.13），其余推过框（失败码 position / turned）。Sonnet f1b 第 1 次就是这么纠的（又推了 5 次，接触点都在背面），盒子最后在框前方 20 cm、转 35°。
> - **先小推试探就几乎没有代价**：第一推只在正中推 3 cm，盒子只转 6.5–10.9°（都 ≤ 10.9°，不记错误事件），已经能看出哪端重；之后前推转回 + 在重端推到位，**25/25 在第 1 次尝试内成功、每次 11 条命令、零错误事件**（按上面的定义算 clean success）。
> - **结论**：从常识第一推（推到位）修复要 7–15 条命令，中位数 7，**达不到 R3 的"修复要 10 条以上命令"**，而且修复方法（绕到前面推轻端）不直观——只从后面纠偏的做法 100% 失败。不算"极其便宜"（不是 1–2 条命令），但对会先试探或会想到前推的 agent，第 1 次尝试就能成功，信号会像 key_turn 一样变成 CS（不触发陷阱直接成功）对 IR（触发后当场修好），外加 `first_push_yaw_deg`。
> - **可选的加难办法（只是建议，没有改场景，留给用户定）**：
>   1. 框收紧：边距 2 cm → 1 cm，框能容下的转角从 10.9° 降到约 5.3°，3 cm 的试探推（6.5–10.9°）也会记成错误事件；代价是前后 / 左右定位容差只剩 ±1 cm，可能测成放置精度（参考 §2.3 nudge_row 的教训），需要先看真实模型的推距误差。
>   2. 把框放远：`target_dist` 从 8–13 cm 改到例如 12–17 cm，常识第一推后领先端更靠近可达边缘，从前面推回时手指会爬上盒子（seed 1010 就是这样，实测前推在可达边缘不可靠），推正中一次就接近不可恢复；风险是 oracle 的推距也变长、可能伤到 oracle ≈ 100%，要重跑 validate 确认。
>   3. 框后面做成"台阶/桌沿"（例如框放在一块低平台的边上，推过头就掉下去，记 irrecoverable），让"推过头"真正不可恢复。**不要用挡墙**：盒子顶到墙上会自己摆正，反而变简单【推断，未测】。
>   4. 加大偏心或降低摩擦让转得更快，使 3 cm 试探也转过框的容差【推断，未测】。
>
> **validate 新旧对比（n = 20，k = 5，`python scripts/validate.py --task l7_offcenter_push --n 20 --k 5`，2026-10-19 实测，185 s）**
>
> | 策略 | 10-12（只推一次）@1 / @2 / @3 / @5 | 10-19（不限次数）@1 / @2 / @3 / @5 | 说明 |
> |---|---|---|---|
> | oracle（推在重心 c = e） | 100 / 100 / 100 / 100 | 100 / 100 / 100 / 100 | 第一推转角 ≤ 0.1°，从不触发陷阱，每次 4 条命令 |
> | naive（推正中一次就 finish，模仿 Sonnet） | 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 | 第一推转 17.0–41.4°，20/20 触发陷阱 |
> | adaptive（10-12：跨尝试按转向挪接触点；10-19：尝试内纠正，见上） | 0 / 100 / 100 / 100 | **95** / 100 / 100 / 100 | 第 1 次尝试内成功 19/20（全部是 recovered_in_attempt）；成功的第 1 次尝试平均 3.37 次推动（3–5）、12.9 条命令（11–19，中位数 11）；所有 21 次尝试平均 3.19 次推动、12.5 条命令 |
> | blind（推正中 ± 0.8 cm，每次都只推一次） | 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 | 100 次尝试最小转角 10.1°，都没成功 |
>
> adaptive 现在的写法（`adaptive_attempt`）：第 1 次尝试先和 naive 一样推正中到位；之后每推一次看一次盒子（脚本是完美感知）：转角 > 3° 就绕到前面推领先端往回推（推距 = 转角 / 每厘米转角，初值 2.8°/cm，在线更新；领先端够不着就往中间挪接触点，还不行才从后面推落后端）；转角 ≤ 3° 时，没推到就从后面在估计的重心处推、推过头就从前面在重心处推回、侧向出框就推端面。重心估计和以前的跨尝试 learner 一样用转向夹逼（第一推往哪边转，重心就在那边，先挪 6 cm）。预算按 la 接口：每次尝试 40 条命令、120 s 运动时间（`agent_max_steps` 默认 40，没有改）；第 2 次尝试从之前所有推动夹逼出的接触点开始。
>
> **主表实例（`sets/main_v1.txt`：3000–3004）新规则下实测**：naive 5/5 失败（第一推转 25.5、−28.9、−26.9、30.0、−32.7°，失败码 turned，陷阱都触发）；oracle 5/5 第 1 次成功（转角 ≤ 0.1°，4 条命令）；adaptive 5/5 第 1 次尝试内修好（3 次推动，11 / 11 / 11 / 11 / 15 条命令）；blind 5/5 一直失败。主表种子不用换。
>
> **重放旧日志（`scripts/replay_la.py` + 一个按日志里的实例重放的草稿脚本，full5T twin 的 com_offset 和种子默认值不同，必须用日志里的实例）**：共 46 个 attempt。**原来判 `several_pushes` 的 5 个**：
>
> | 日志 | attempt | 原判定 | 新判定 | 推动 | bad | trap / 当场修好 | 第一推转角 | 说明 |
> |---|---|---|---|---|---|---|---|---|
> | grok47xh 独立（noharness0，s3000，F1） | 1 | F several_pushes | **S** | 10 | 7 | ✓ / ✓ | −0.8° | 先在重端附近小推 1.7 cm、正中 1.2 cm，第 3 推（正中 9.2 cm）转 25°（陷阱）；之后 5 次从前面推、1 次推端面、1 次从后面推落后端，转回到 1°，进框；33 条命令 |
> | grok47xh 带历史（s3000，F1） | 1 | F several_pushes | F position | 3 | 2 | ✓ / | 6.6°（ok） | 第一下是从侧面过来时碰到右端面；之后推偏到侧向出框（中心偏左 6 cm、偏后 5 cm） |
> | grok47xh 带历史 | 3 | F several_pushes | F tipped | 5 | 5 | ✓ / | 30.4° | 推正中转 30°；其中 2 次是张着手指推的；最后一推把盒子掀转 98°，翻倒（irrecoverable tipped）；模型自述"arm jammed" |
> | Sonnet f1b（p11820，s3000，F1） | 1 | F several_pushes | F turned | 6 | 6 | ✓ / | 30.4° | 之后 5 次都从后面推（接触点在重端一侧 1.8–7.9 cm），转角没减（35°），盒子推到框前方 20 cm |
> | gptpolicy full1（s2000，F2，v1 工具） | 1 | F several_pushes | F turned | 3 | 1 | ✓ / | 0.0°（ok） | 前两下各推 2.5 cm 没转，第 3 下推 19 cm 转 −67° |
>
> 其余 41 个 attempt 判定和失败码都不变：cla2（F2）4 个 run 的 6 个 attempt、cla4 的 3 个、f1b 的另 3 个、full5N 独立 15 个（全是 F turned，第一推转 27.3–37.0°，每个只推了一次）、full5T twin 5 个（F turned，20.6–36.5°）、grok 带历史第 2 次、gpt full1 的 8 个。原来 4 个成功的 attempt（cla2 第 3 次、cla4 第 2 次、f1b 两个 run 的第 2 次）都只推了一次，仍是 S，trap_fired = False。重放转角和日志记录差 ≤ 0.5°。f1_l7off 两个目录没有 done 事件，跳过。
>
> **接口自测（la 风格，rgb，F1，端口 9953，seed 3000，2026-10-19）**：日志在 `results/l7_offcenter_push/selftest_20261019/`。README 只有新指令（当时是 10-19 第一版措辞，仍带 "straight forward … without turning"）、通用说明和预算（40 条命令、120 s），没有"只推一次"，没有重心信息。① 推正中一次（合爪移到盒后 / 下降 / 0.05 m/s 推到位 / 抬起，4 条命令）：回复正常（`stopped_from_target` 0.2–0.4 cm），agentview 和腕部图像都能清楚看出盒子逆时针转了约 26°、横跨框的边。② 绕到前面推右（领先）端往回推（4 条）：转角 → −2°，盒子在框后 5.4 cm。③ 从后面在左（重）端一侧推 5.4 cm（4 条）：进框。`finish` → "Attempt succeeded."；outcome：n_pushes 3、n_bad_pushes 1、trap_fired、recovered_in_attempt、first_push_yaw_deg 26.0、12 条命令、13.3 s。（纠正坐标是用日志重放读仿真里的盒子位姿算的，这一步测的是接口，不是 agent。）第二个会话（同种子，k = 2）：第 1 次张爪推正中一次，盒子进框、转 0.8°，回复 "Attempt failed."（outcome failure = open_gripper），随后正确提示第 2 次开始、场景已复位；第 2 次合爪推正中 → "Attempt failed."（turned 26°）。两个服务器进程都已关掉，9953 端口已释放。
>
> **脚本序列检查（错误事件记账，2026-10-19 实测；合爪、yaw 跟盒子朝向、8 cm/s；"重心处" = 用真实 e）**
>
> | seed | 序列 | 结果 | 推动 | bad | 陷阱新触发 | trap_fired | 当场修好 | 第一推后转角 | 第一推 ok | 张爪推 | irrecoverable | 最终转角 |
> |---|---|---|---|---|---|---|---|---|---|---|---|---|
> | 3000 | 在重心处分两次推（5 cm + 余下） | S | 2 | 0 | 0 |  |  | -0.1 | True | 0 |  | -0.1 |
> | 3001 | 正中试探 3 cm，再在重心处推完 | F turned | 2 | 1 | 1 | ✓ |  | -10.5 | True | 0 |  | -10.6 |
> | 3002 | 在重心处推过头 5 cm，再从前面在重心处推回 5 cm | S | 2 | 1 | 1 | ✓ | ✓ | -0.0 | False | 0 |  | -0.1 |
> | 3003 | 正中推到位（陷阱），再推正中一次 2 cm | F turned | 2 | 2 | 1 | ✓ |  | 29.6 | False | 0 |  | 37.9 |
> | 3004 | 张开夹爪推正中一次 | F turned | 1 | 1 | 1 | ✓ |  | -12.8 | False | 1 |  | -13.2 |
> | 3000 | 压住盒顶 2 cm 拖 10 cm（翻倒） | F tipped | 1 | 1 | 1 | ✓ |  | -0.8 | False | 0 | tipped | -0.8 |
> | 3000 | 在重心处一路推 30 cm（推出可达范围） | F position | 1 | 1 | 1 | ✓ |  | -0.1 | False | 0 | out_of_reach | -0.1 |
> | 3001 | 从侧面推左端面 4 cm，再在重心处推到位 | F position | 2 | 2 | 1 | ✓ |  | -0.0 | False | 0 |  | -0.1 |
>
> 分两次推到位不记错误（第一推后只是没推到）；推过头 5 cm 记 1 次错误（overshot），从前面推回后成功，记为当场修好；正中试探 3 cm 转 10.5°（≤ 10.9°，不算错误），但之后只在重心处直推不会把那 10° 转回来，最后有角越出框边，记 1 次错误；张爪推在 e = −7.5 cm 的 3004 上也转了 12.8°（25 个实例里张爪失败的 3 个之一）；压顶拖翻倒、一路推 30 cm 出可达范围都记 irrecoverable。
>
> **已知问题 / 需要用户拍板**
> - 错误事件里"没进框"的解释（只是没推到不算错误），见上。
> - 张爪推判失败是新加的判定规则（理由见上）；如果用户更愿意不加规则，就得接受"张爪推一次 22/25 成功"。
> - （已解决，10-19 措辞修改，见最上面一节）原文：指令保留了 "without turning"。字面上"推的过程中转了、又转回来"也违反这句，但判定只看最终状态（转回来、进框就算成功），判定比指令松；按 §2.1 这算不算问题要用户定。可以改成只说终态的措辞。
> - 不满足 R3（见"纠正代价"）；信号主要看 CS / IR 和第一推转角。会先试探的 agent 在 attempt 1 就能干净成功。
> - 出可达范围的阈值（中心 x > 0.13）是近似值：没有按 y 细分（|y| = 0.3 时夹爪只能到 x ≈ 0.08–0.12），转着的盒子也只测了 25° 一种。
> - 共享文件需要跟进（我没有改，按约束只改了本任务的文件）：`sets/main_v1.txt` 头部和 `RUN_WITH_YOUR_API.md` 的 "Task changes" 要像 key_turn 那样加一句 10-19 的说明（旧 run 要重跑，`api_batch.py contract` 会因任务代码哈希变化标出来）；`DESIGN_INSIGHTS.md` §6 的任务列表；`scripts/replay_la.py` 打印的字段里加 `n_bad_pushes`、`first_push_yaw_deg`、`n_open_gripper_pushes`、`irrecoverable`（现在只打印 n_pushes / trap_fired / recovered_in_attempt）。`results/l7_offcenter_push/arms/validate.json`（h0 / twin 对照臂）是 10-12 用旧代码跑的，没有重跑（10-19 措辞修改时已用新代码重跑，全部符合预期，见最上面一节）。
> - 旧备份：`lffbench/tasks/l7_offcenter_push.py.bak-20261019-relax`、`task_cards/l7_offcenter_push.md.bak-20261019-relax`、`results/l7_offcenter_push/*.bak-20261019-relax`。

| 项 | 内容 |
|---|---|
| 类别 | failed_interaction_inference（L7，"搞清楚为什么这次接触没起作用"），`level="L7"` |
| 能力 | Perceive（看出盒子转了、往哪边转）、Reason（转向 → 哪一端重）、Utilize（下次把接触点挪向重的一端） |
| 协议 | cross：每次尝试重置到同一实例，k = 5 |
| 场景 | 长盒子 22 × 9 × 4 cm（长边沿 y，0.6 kg，μ 0.5），顶面中间一条对称的浅色胶带；正前方（+x）8–13 cm 处一个绿色目标框（盒子外廓每边加 2 cm，只是标记，不参与碰撞）。盒子位置 x ∈ [-0.20, -0.14]、y ∈ [-0.08, 0.08] 随机 |
| 规则 | **10-19 起不限推动次数（见上）。** 10-19 前：指令里写明并在 outcome 里强制：**只能推一次**——盒子只能被一条 move_to 推动（某条 move_to 让盒子移动 > 1 cm 或转 > 3° 就算一次推；超过一次直接判失败）。否则 agent 可以在一次尝试里边推边纠偏，就不再考"跨尝试学到了什么" |
| 成功 | 10-19 起：盒子四个角都在框内、没翻倒，且每次推动都是合爪推的。10-19 前：盒子四个角都在框内、没翻倒、只推了一次 |

## 任务 → 隐藏信息 → 所需能力 → 调整后的动作
- **任务**：合爪把盒子推进它前方（远离机器人那一侧）的绿框，最后整个盒子都在框内（10-19 起的措辞；10-19 前是"从盒子后面直着往前推、不能转"，而且只能推一次）。
- **隐藏信息**：盒子重心沿长边偏了 e，|e| ~ U[4.5, 7.5] cm，左右随机（运行时改 `body_ipos`）。外观对所有 e 完全一样，动手前从 RGB 看不出来。
- **先验为什么失败**：自然做法是推背面正中。桌面摩擦合力作用在重心下方，推力线不过重心 → 盒子边走边转，**重的一端落后**；转角 17–41°（验证集 20 个实例实测），框只容得下约 9°（10-19 按几何精确算是 10.9°，盒子居中时）。
- **要从失败里学到**：转的方向 → 哪端重（落后的那端）；转得多 → 偏得多。**调整后的动作**：把接触点（夹爪中心沿长边的位置）挪向重的一端，推力线过重心附近就平移。

## F2 反馈（只给相对测量，不给绝对坐标、不给 e）
失败示例（seed 1000，e = −6.3 cm，推正中）：
> Attempt failed. The box is not completely inside the rectangle. It turned 30 degrees clockwise (seen from above): its right end ended 11.2 cm further back than its left end. Its centre ended 0.3 cm forward and 2.9 cm to the right of the rectangle's centre. When the box started to move, the gripper's centre was 0.6 cm to the right of the middle of the box (measured along the box's long side).

另外两种失败文字：（10-19 前）"The box was moved by N separate move_to commands; only one push is allowed." / "The box did not move: the gripper never pushed it."
10-19 起：推了多次时加一句 "It was moved by N separate move commands; after the first one it had turned X degrees …"，最终转角改成 "It ended up turned …"；张爪推且盒子在框内时是 "The box is completely inside the rectangle, but it was pushed with the gripper open; the task asks for pushing with the closed gripper."（F2 只做消融，主协议 F1 只给成败。）
最后一句（夹爪中心在盒子上的位置）是 agent 自己动作的测量，给 RGB-only 的 agent 校准瞄准用。"开始动"用盒子位移 2 mm 判定，不用接触轮询（见下）。

## 物理标定（实测，2026-10-07）
合爪 yaw 0（两指沿 y 并排，接触面约 5 cm 宽）、8 cm/s、推 12 cm。表中是 c − e（接触点相对重心，cm）→ 最终转角：

| e | −4 | −3 | −2 | −1 | 0 | +1 | +2 | +3 | +4 | 推正中 (c=0) |
|---|---|---|---|---|---|---|---|---|---|---|
| +4.0 cm | +22° | +9° | +1° | 0 | 0 | 0 | 0 | −12° | −22° | +22° |
| −5.5 cm | +23° | +12° | +1° | 0 | 0 | 0 | 0 | −8° | −22° | −35° |
| +7.0 cm | +22° | +5° | +1° | 0 | 0 | 0 | −1° | −14° | −23° | +42° |

- 接触点在重心 ±2 cm 内就是纯平移（< 1°），±3 cm 开始转 5–14°。所以 e 下限取 4.5 cm（e = 4 cm 时推偏 1 cm 只转 8°，盲重试会撞进框里）。
- 平推时盒子走的距离 ≈ 夹爪越过背面的距离 + 0.9 cm（合拢手指前沿在抓取点前方约 0.9 cm），oracle 落点误差 ≤ 0.2 cm。
- 推的时候接触是"粘–滑"的：在控制步边界上大多数时候查不到夹爪–盒子接触（一次 12 cm 的推只在 1–2 个控制步边界上看到接触）。所以"第一次接触"改为用盒子位移判定。
- 改 `body_ipos` 要求 XML 里的 inertial 本身就偏离 body 原点（这里写 0.1 cm），否则 MuJoCo 的 sameframe 优化会忽略运行时的 ipos（上一个子代理的记录）。

## headroom 验证（10-19 前的规则：只推一次；n = 20，k = 5，实测，`python scripts/validate.py --task l7_offcenter_push --n 20 --k 5`；10-19 新表见顶部）
| 策略 | @1 | @2 | @3 | @5 |
|---|---|---|---|---|
| oracle（推在重心 c = e） | 100% | | | 100% |
| naive（推正中） | 0% | | | 0% |
| adaptive（转向 → 往落后的一端挪 6 cm；两边都见过就二分；推直了再按前后误差修推距） | 0% | 100% | 100% | 100% |
| blind（推正中 ± 0.8 cm 随机） | 0% | 0% | 0% | 0% |

## agent 接口自测（10-19 前；端口 9920，rgb，F2，k=3，2026-10-07；10-19 的自测见顶部）
README 无泄露（只有指令和通用说明）；close_gripper / move_to / done 都正常；推正中得到上面那种 F2 文字，第二次尝试正常重置。

## 已知问题 / 推测（未验证的标【推测】）
- （10-19 前）脚本 adaptive 第 2 次就 100%，因为稳定带有 ±2 cm、e 的范围只有 3 cm 宽，一步 6 cm 总能落进去。对 VLM 难点在别处：从 RGB 估盒子位置、把"顺时针 30°、右端落后 11 cm"翻译成"往右挪多少"【推测】。
- agent 若用 yaw 90（两指沿 x，接触面只有一指宽）推，稳定带会更窄【推测，未测】。
- 推得太快盒子会滑过头（μ 0.5 时 0.6 m/s 约多滑 3.7 cm，按 v²/2μg 估算【推测，未测】）；指令里写了 "slowly"。
