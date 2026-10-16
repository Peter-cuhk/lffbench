# L7 `l7_key_turn`：钥匙只能往一个方向转，转错了就换方向

代码：`lffbench/tasks/l7_key_turn.py`　结果：`results/l7_key_turn/validate.json`　视频：`videos/l7_key_turn_adaptive_s1001.mp4`（顺时针锁：第 1 次逆时针转不动，第 2 次顺时针成功）

> 2026-10-08 新增，替代 `l7_hidden_mechanism`。用户意见：滑门那种设置太阴，正常人不会去横着拉门；开锁更自然——很多人先逆时针转钥匙，转不动就知道该顺时针转，是一个很好的 learn-from-failure 例子。
>
> **2026-10-16 改（用户定）：不再限制每次尝试的转动次数。**原来的"每次尝试只能转一次"把尝试内的试错硬挪到了尝试间（违反 `DESIGN_INSIGHTS.md` 第 9 行）：已有日志里所有先逆时针转的 run（cla4 独立、grok47xh 独立、grok47xh 带历史第 1 次），都在同一次尝试里马上改成顺时针、把锁打开了，却因为"转了 2 次"判失败。现在只要锁开了就算成功；转错方向记成错误事件（`trap_fired`），当场改正记成尝试内恢复（`recovered_in_attempt`）。用新判定重放这 3 个 attempt，都改判为成功（trap_fired、recovered_in_attempt 都是 True）。

| 项 | 内容 |
|---|---|
| 类别 | failed_interaction_inference（L7，"搞清楚为什么这次接触没起作用"），`level="L7"` |
| 能力 | Perceive（看钥匙头转没转、红色锁舌缩没缩）、Reason（"只转了 3° 就卡死" ≠ "没夹紧"）、Plan（换方向） |
| 协议 | cross，k = 5 |
| 场景 | 桌上一个灰色锁盒（9 × 9 × 5 cm，fixture），顶面插着一把黄铜钥匙，扁平钥匙头竖着露在外面（宽 3.6、厚 1.6、高 3 cm），夹爪从上往下夹住钥匙头、转手腕就能转钥匙。锁盒正面（+x，朝前置相机）伸出一截红色锁舌，开锁后缩回去。锁盒位置 x ∈ [-0.15, -0.02]、y ∈ [-0.15, 0.15]，绕竖直轴转 ±15° |
| 规则 | 转动次数不限（2026-10-16 起；之前是每次尝试只能转一次）。一条 move_to 若（开始时夹着钥匙且指令 yaw 变化 ≥ 5°）或（钥匙实际转了 > 3°）就记为一次转动，写进 `turns` |
| 成功 | 钥匙曾转到开锁位置（往正确方向 ≥ 80°，止挡在 90°；到位后锁舌缩回并保持），转几次都行 |
| 错误事件 | 一次转动若往锁住的方向转（夹着钥匙时看手腕指令方向，没夹着时看钥匙实际转向），且转动中钥匙到了锁住一侧的止挡（离止挡 0.5° 以内），且当时锁还没开，就记 `hit_stop`。outcome 里：`n_wrong_way` = 这样的转动有几次，`trap_fired` = 至少一次，`recovered_in_attempt` = 陷阱触发过且最后锁开了。论文的四分类：第 1 次尝试成功且没有 `trap_fired` 是 CS，有 `trap_fired` 但成功是 IR |

## 任务 → 隐藏信息 → 所需能力 → 调整后的动作
- **任务**：把锁打开（钥匙转四分之一圈）。
- **隐藏信息**：开锁方向，俯视顺时针（CW）或逆时针（CCW）。按种子奇偶交替（偶数 CCW、奇数 CW），每批实例正好一半一半。反方向只有 3° 空程，然后是硬止挡。两种锁外观完全一样。
- **先验为什么失败**：用用户的例子做先验——先逆时针转 90°。在 CW 锁上，钥匙转过 3° 空程就卡死，手腕顶住后停下（见下文"会失速的手腕"），锁舌不动。
- **要从失败里学到 / 调整后的动作**："钥匙往这边根本转不动"（而不是"夹滑了"）→ 往另一边转。转错一次就够，同一次尝试里换方向就能打开；带历史时，下一次尝试应该第一下就往对的方向转。

## F2 反馈（只给相对测量，不给隐变量）
转错方向（seed 1001，CW，脚本先验逆时针 90°；agent 接口自测里是同一段文字）：
> Attempt failed. The lock is still locked. During your turn the wrist rotated 13 deg counterclockwise (seen from above) and the key rotated 3 deg counterclockwise, then it would not turn any further and the wrist stopped there; the gripper lost hold of the key during the turn. When the gripper first closed, its centre was 0.1 cm backward and 0.0 cm to the right of the centre of the key's head and 1.5 cm below the top of the head; it was holding the key.

成功：`Attempt succeeded. The lock is unlocked.`
（2026-10-16 前这句是 "During your (first) turn"；当时 "turned in N separate moves; only one turn is allowed per attempt" 也算一种失败，现在不算了。）
其他失败文字：转了多次仍没打开（"The key was turned (or pushed round) in N separate moves. During the first one … During the last one …"，分别描述第一次和最后一次转动）、没转（"the key was never turned"）、没夹住（"…did not get hold of the key."）、转的方向对但不够（"…the key rotated 45 deg counterclockwise"，没有 "would not turn any further"）。
la / v2 接口下 move 的回报里还有 `stopped_from_target`：转错方向时是 `deg ≈ 77`（手腕没转到目标），这本身就是一条强线索。
锁盒是 fixture，harness 的 `holding_object` / `touched_object` 只看可动物体；和 l7_hidden_mechanism 一样用 `on_agent_tool` 钩子补了 `holding_key` 字段、把碰钥匙算进 `touched_object`（v2 / la 接口会丢掉 `holding*` 字段，与其他任务一致）。

## 物理与接口上的处理（原型实测，2026-10-08）
| 问题 | 现象（实测） | 处理 |
|---|---|---|
| 夹持力太小 | robosuite Panda 夹爪是位置伺服（kp 1000，上限 20 N），夹力 ≈ kp × 手指开度：7 mm 厚的钥匙头只有约 4 N，转动时钥匙头在指间拧歪、把手指撑开到 4 cm | 钥匙头加厚到 16 mm（约 8 N），像门锁内侧的旋钮那种粗把手 |
| 止挡太软 | 默认 solref/solimp 下，用力拧时钥匙冲过止挡 10–15° | `jnt_solref = 0.004`、`jnt_solimp = (0.99, 0.999, …)`，之后稳定停在 ±3° |
| 转错方向后泄露答案 | OSC 一直往目标拧，手指被钥匙头撑开；松手时手指把钥匙往"能转"的方向拨了 15–45° | `KeySkills` 模拟力矩受限的手腕：夹着钥匙、钥匙已在止挡、手腕又多拧 6° 时，不再继续转（失速），20 步后这条 move 结束；同时转动角速度限制在 90°/s |
| 失速后手臂乱甩 | agent 接口自测发现：失速后 `Skills.yaw` 还是没到达的目标角，下一条命令（开夹爪）里的 hold 又把手腕往那边拧，手臂甩出 8.5 cm | 失速时把目标 yaw 改成当前实际 yaw。修后重跑自测和验证，结果不变、手臂稳定 |
| yaw 取模 180° | 共享的 `Skills.wrap_yaw` 把 yaw 按 180° 等价取最近角：抓取时对，但"yaw=+95"会变成 −85°，钥匙往反方向转；yaw=90 恰好是平局，取到 −90 | 本任务的 `KeySkills.wrap_yaw` 按字面取角（离当前 yaw 最近的 360° 表示，裁到 ±113°）；大角度转动由 90°/s 的插值保证不会触发 π 处的轴角歧义 |

修完后原型 40 次（2 种摩擦 × 有/无抓取抖动 × 10）：正确方向 40/40 开锁，错误方向 40/40 钥匙留在 5° 以内。

## headroom 验证（n = 20，k = 5，实测，`python scripts/validate.py --task l7_key_turn --n 20 --k 5`）
20 个实例：CCW 10、CW 10（按种子奇偶）。

| 策略 | @1 | @2 | @3 | @5 |
|---|---|---|---|---|
| oracle（知道方向，转 90°） | 100% | | | 100% |
| naive（逆时针 90°） | 50% | | | 50% |
| adaptive（上一次钥匙转了 < 10° 就卡住 → 换方向；转得动但不够 → 同方向转 100°） | 50% | 100% | 100% | 100% |
| blind（逆时针 90 ± 8°，抓取位置/角度小抖动，无历史） | 50% | 50% | 50% | 50% |

CW 的 10 个实例上：naive 0/10，adaptive 第 2 次 10/10，blind 5 次都是 0/10。

**和验收线的偏差（已知、有意）**：naive@1 = 50%，高于 ≤ 25% 的验收线。方向只有两种，任何固定先验都只会在一半实例上失败；如果把方向偏向某一边来压低 naive，就变成在赌 agent 的先验恰好是哪边。所以按"均匀 + 配平"设计，learning 的度量只看 attempt 1 失败的那一半 run：第 2 次是否换了方向（succ@2 | fail@1）、同错复犯率。（主表 5 个实例 3001–3009 全是 CW，naive 在上面全失败。）

2026-10-16 放开转动次数后重跑同一命令，四种脚本策略的数字不变（它们每次尝试都只转一次）。

## 尝试内多次转动的脚本检查（2026-10-16，实测）
夹住钥匙后按顺序转（正数 = 俯视逆时针），然后松开、抬起：

| seed（方向） | 转动序列 | 结果 | n_turns | n_wrong_way | trap_fired | recovered_in_attempt |
|---|---|---|---|---|---|---|
| 3001（CW） | +90, −100 | S | 2 | 1 | ✓ | ✓ |
| 3003（CW） | +90, +90, −100 | S | 3 | 2 | ✓ | ✓ |
| 3005（CW） | +90 | F still_locked | 1 | 1 | ✓ | |
| 3007（CW） | +90, +90 | F still_locked | 2 | 2 | ✓ | |
| 3009（CW） | −90 | S | 1 | 0 | | |
| 3000（CCW） | +90 | S | 1 | 0 | | |
| 3002（CCW） | +90, −95（开锁后又转回 0°） | S | 2 | 0 | | |
| 1001（CW） | +90, −100 | S | 2 | 1 | ✓ | ✓ |

已经顶在止挡上再往对的方向转，不算 `hit_stop`（3003 第 3 次）；锁开了以后再往回转也不算（3002）。用新判定重放已有 agent 日志：cla4 独立、grok47xh 独立、grok47xh 带历史第 1 次，3 个原来判 `several_turns` 的 attempt 都改判 S（n_wrong_way = 1，recovered_in_attempt = True）；cla4 带历史、grok47xh 带历史第 2 次原来就是 S，现在不变（trap_fired = False）。

## agent 接口自测（la 风格，rgb，F2，k=3，seed 1001 = CW，2026-10-08）
README 只有指令和通用说明，没有方向信息。第 1 次逆时针转（`move_eef yaw=80.6`）→ 回报 `stopped_from_target deg 76.6`，结束后得到上面那段 F2；第 2 次 `yaw=-99.4` → "Attempt succeeded. The lock is unlocked."。每次尝试 5 条命令。

## 已知问题 / 推测（未验证的标【推测】）
- 放开转动次数的代价（2026-10-16）：转错只花一条命令，不满足设计规则 R3（"第一次按常识的交互要有代价"），所以细心的 agent 第 1 次尝试就会成功，A@1 会接近 100%。这个任务的信号改为：CS（第一下就往对的方向转）对 IR（转错后当场改正），以及 `n_wrong_way` ≥ 2（反复往错的方向拧）。主表实例全是 CW 锁，CS 实际测的是模型的先验是不是先顺时针。原来设想的 within 变体（不限转动次数）现在就是这个任务本身。
- F2 里有 "the gripper lost hold of the key during the turn"（失速时一只指垫离开了钥匙头，`_check_grasp` 判为没夹住）。这是真实测量，但会给 agent 一个"可能是夹滑了"的错误解释——正是 l7_hidden_mechanism 里模型掉进去的坑（把"没动"归因于夹不紧）。关键线索是 "rotated 3 deg … then it would not turn any further"。【推测】一部分模型会在同方向重试。
- 钥匙头在 agentview 里很小（320 px 渲染里整个锁盒目测约 45 px 宽），转没转主要靠腕部相机和锁舌看。【推测】RGB-only agent 的主要难点之一是估计钥匙头的高度（顶面在桌上 8.4 cm）。
- 失速是 `KeySkills` 里的软件行为（模拟力矩受限的手腕），不是 OSC 本身的特性；只在夹着钥匙、钥匙在止挡时生效，不影响其他动作。
