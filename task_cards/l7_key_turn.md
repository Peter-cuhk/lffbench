# L7 `l7_key_turn`：钥匙只能往一个方向转，转错了就换方向

代码：`lffbench/tasks/l7_key_turn.py`　结果：`results/l7_key_turn/validate.json`　视频：`videos/l7_key_turn_adaptive_s1001.mp4`（顺时针锁：第 1 次逆时针转不动，第 2 次顺时针成功）

> 2026-10-08 新增，替代 `l7_hidden_mechanism`。用户意见：滑门那种设置太阴，正常人不会去横着拉门；开锁更自然——很多人先逆时针转钥匙，转不动就知道该顺时针转，是一个很好的 learn-from-failure 例子。

| 项 | 内容 |
|---|---|
| 类别 | failed_interaction_inference（L7，"搞清楚为什么这次接触没起作用"），`level="L7"` |
| 能力 | Perceive（看钥匙头转没转、红色锁舌缩没缩）、Reason（"只转了 3° 就卡死" ≠ "没夹紧"）、Plan（换方向） |
| 协议 | cross，k = 5 |
| 场景 | 桌上一个灰色锁盒（9 × 9 × 5 cm，fixture），顶面插着一把黄铜钥匙，扁平钥匙头竖着露在外面（宽 3.6、厚 1.6、高 3 cm），夹爪从上往下夹住钥匙头、转手腕就能转钥匙。锁盒正面（+x，朝前置相机）伸出一截红色锁舌，开锁后缩回去。锁盒位置 x ∈ [-0.15, -0.02]、y ∈ [-0.15, 0.15]，绕竖直轴转 ±15° |
| 规则 | 指令里写明、outcome 里强制：**每次尝试只能转一次**——夹住钥匙，持握状态下只发一条转手腕的 move 命令，然后松开。判定：一条 move_to 若（开始时夹着钥匙且指令 yaw 变化 ≥ 5°）或（钥匙实际转了 > 3°）就算一次转动；超过一次判失败 |
| 成功 | 钥匙曾转到开锁位置（往正确方向 ≥ 80°，止挡在 90°；到位后锁舌缩回并保持），且只转了一次 |

## 任务 → 隐藏信息 → 所需能力 → 调整后的动作
- **任务**：把锁打开（钥匙转四分之一圈）。
- **隐藏信息**：开锁方向，俯视顺时针（CW）或逆时针（CCW）。按种子奇偶交替（偶数 CCW、奇数 CW），每批实例正好一半一半。反方向只有 3° 空程，然后是硬止挡。两种锁外观完全一样。
- **先验为什么失败**：用用户的例子做先验——先逆时针转 90°。在 CW 锁上，钥匙转过 3° 空程就卡死，手腕顶住后停下（见下文"会失速的手腕"），锁舌不动。
- **要从失败里学到 / 调整后的动作**："钥匙往这边根本转不动"（而不是"夹滑了"）→ 往另一边转。一次失败就够，第 2 次必成。

## F2 反馈（只给相对测量，不给隐变量）
转错方向（seed 1001，CW，脚本先验逆时针 90°；agent 接口自测里是同一段文字）：
> Attempt failed. The lock is still locked. During your (first) turn the wrist rotated 13 deg counterclockwise (seen from above) and the key rotated 3 deg counterclockwise, then it would not turn any further and the wrist stopped there; the gripper lost hold of the key during the turn. When the gripper first closed, its centre was 0.1 cm backward and 0.0 cm to the right of the centre of the key's head and 1.5 cm below the top of the head; it was holding the key.

成功：`Attempt succeeded. The lock is unlocked.`
其他失败文字：转了多次（"The key was turned (or pushed round) in N separate moves; only one turn is allowed per attempt…"）、没转（"the key was never turned"）、没夹住（"…did not get hold of the key."）、转的方向对但不够（"…the key rotated 45 deg counterclockwise"，没有 "would not turn any further"）。
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

**和验收线的偏差（已知、有意）**：naive@1 = 50%，高于 ≤ 25% 的验收线。方向只有两种，任何固定先验都只会在一半实例上失败；如果把方向偏向某一边来压低 naive，就变成在赌 agent 的先验恰好是哪边。所以按"均匀 + 配平"设计，learning 的度量只看 attempt 1 失败的那一半 run：第 2 次是否换了方向（succ@2 | fail@1）、同错复犯率。

## agent 接口自测（la 风格，rgb，F2，k=3，seed 1001 = CW，2026-10-08）
README 只有指令和通用说明，没有方向信息。第 1 次逆时针转（`move_eef yaw=80.6`）→ 回报 `stopped_from_target deg 76.6`，结束后得到上面那段 F2；第 2 次 `yaw=-99.4` → "Attempt succeeded. The lock is unlocked."。每次尝试 5 条命令。

## 已知问题 / 推测（未验证的标【推测】）
- "只能转一次"是人为规则。真人转不动会立刻往回转，那是同一次尝试内的 learn-from-failure；不加规则的话，细心的 agent 第 1 次尝试就会成功（转错→换向），任务就测不到跨尝试的学习。可以另做一个 within 变体（不限转动次数，看同一次尝试里转错之后有没有换方向、有没有反复往错的方向拧）——未实现。
- F2 里有 "the gripper lost hold of the key during the turn"（失速时一只指垫离开了钥匙头，`_check_grasp` 判为没夹住）。这是真实测量，但会给 agent 一个"可能是夹滑了"的错误解释——正是 l7_hidden_mechanism 里模型掉进去的坑（把"没动"归因于夹不紧）。关键线索是 "rotated 3 deg … then it would not turn any further"。【推测】一部分模型会在同方向重试。
- 钥匙头在 agentview 里很小（320 px 渲染里整个锁盒目测约 45 px 宽），转没转主要靠腕部相机和锁舌看。【推测】RGB-only agent 的主要难点之一是估计钥匙头的高度（顶面在桌上 8.4 cm）。
- 失速是 `KeySkills` 里的软件行为（模拟力矩受限的手腕），不是 OSC 本身的特性；只在夹着钥匙、钥匙在止挡时生效，不影响其他动作。
