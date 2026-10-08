# L1-C `l1_yaw_bias`：手腕转角有标定偏差时，把长条放进窄槽

> 状态：已验证（2026-10-07）。标"实测"的数字都来自当前代码（`lffbench/tasks/l1_yaw.py`）。

## 一句话（任务 → 隐藏信息 → 所需能力 → 调整后的动作）
- **任务**：把桌上的蓝色长条拿起来，放进一个只比它宽一点的灰色窄槽里。
- **隐藏信息**：手腕转角有固定标定偏差 β（15–30°，方向随机）：指令 yaw θ 实际执行成 θ + β，本体读数却说转到了 θ。
- **所需能力**：从"长条斜着搭在槽上"的失败画面 / F2 角度测量里看出转了多少度、往哪边（Perceive）；把偏差带到下一次（Utilize）。
- **调整后的动作**：同一种抓放方法，所有指令 yaw 整体减去 β（抓和放都要减）。

| 项 | 内容 |
|---|---|
| 类别 / 等级 | system_identification / L1 动作微调（方法和理解都对，只是机器人自己的手腕转角有偏差） |
| 能力 | Perceive, Utilize |
| 协议 | cross：每次尝试重置到同一初始状态、同一偏差 β，k = 5 |
| 指令（直呼） | "Pick up the blue bar and put it into the holder." |
| 指令（间接） | "Pick up the long blue stick and drop it into the narrow grey trough so that it lies inside." |
| 场景 | 蓝色长条（生成物体，14 × 1.8 × 1.8 cm，70 g，长轴 yaw ±45°）平放在桌上；灰色窄槽（fixture，内口 16 × 3.2 cm、壁高 3 cm、壁厚 0.6 cm，长轴 yaw ±45°）；干扰物 `cream_cheese`、`butter`。长条 x ∈ [−0.16, 0.04]、y ∈ [−0.20, 0.20]；槽离长条 ≥ 22 cm；干扰物彼此和离长条 / 槽 ≥ 15 cm，离"起始位→长条""长条→槽"两条直线 ≥ 10 cm |
| 机器人 | 原装 Panda 夹爪（8 cm 行程），不改 |
| 隐变量 | 固定手腕 yaw 偏差 β，\|β\| ~ U[15°, 30°]，符号随机：指令 yaw θ 实际执行成 θ + β。实现：把 Skills 的"yaw 0 姿态" `R0` 转 β（`YawSkills`）；agent harness 的 yaw 读数也是相对 `R0` 算的，所以读数 = 指令值，β 不出现在任何本体读数里。重置后手腕先转到机器人自己认为的 yaw 0（真实 yaw = β），首个读数就是 0 |
| 成功判据 | 长条中心在槽内口矩形里、中心低于槽沿、长条平躺（竖直轴 cos > 0.95）、不在夹爪里 |

## 为什么先验会失败
先验 = 完美感知（长条、槽的真实位置和朝向）+ 标准动作（在长条中心上方、yaw 对齐长条使两指横跨长条 → 降到夹爪点离桌面 1.05 cm → 合拢 → 升起 → 移到槽中心上方、yaw 对齐槽长轴 → 降到长条底面高过槽沿 1.2 cm 处松开），但不知道 β。

- 指令"横跨长条"时，夹爪实际斜了 β；合拢的手指把长条扳正到与钳口垂直（实测：扳转角 = β），于是长条在手里相对机器人"以为的方向"转了 β。到槽上方"对齐"槽长轴时，长条实际与槽差 β。长条横跨槽口的投影 14 sin β + 1.8 cos β 在 β = 15° 时已有 5.4 cm，比槽外宽 4.4 cm 还大，松手后长条两端搭在槽壁顶上，斜躺在槽上面。
- 实测：20/20 个 naive 都是这种失败（`missed_place`），松手时长条与槽的夹角 = β（误差 ≤ 0.2°）。
- 失败在图像里看得见：agentview 里长条斜搭在槽上；腕部相机里槽和长条明显成一个角度。

## 要从失败里学到什么
F2 报告两个角度：(1) 合拢时钳口相对"横跨长条的正确朝向"转了多少度、往哪边（逆时针 / 顺时针，从上往下看）；(2) 松手时长条长轴相对槽长轴转了多少度。脚本先验下两者都等于 β。agent 一次失败后就能估计 β，下一次所有指令 yaw 都减去 β。agent 如果放的时候本来就没对齐槽，(2) 给出的就是"没对齐 + β"的残差，照样是正确的修正量。

转向约定：从上往下看逆时针 = 从 forward（+x）转向机器人左边（+y）= yaw 增大。

## F2 反馈（实测文本，seed 1000，β = −19.4°；`outcome()["detail"]`）
第 1 次（naive，失败）：
> Attempt failed. When the jaws closed on the blue bar, they were rotated 19 deg clockwise seen from above (from forward towards the robot's right, -yaw) from square across the bar, and the closing fingers turned the bar by 19 deg. When the gripper opened above the holder, the bar's long axis was rotated 19 deg clockwise seen from above (from forward towards the robot's right, -yaw) from the holder's long axis, and the bar's centre was 0.1 cm backward (-x, toward the robot) of the holder's centre. The bar came down across the holder's walls instead of dropping in; it is now lying on top of the holder, its long axis rotated 20 deg clockwise seen from above (from forward towards the robot's right, -yaw) from the holder's long axis.

第 2 次（adaptive 把指令 yaw 加 19.35°，成功）：
> Attempt succeeded. The blue bar is inside the holder. When the gripper opened, the bar's long axis was not rotated (within 0.5 deg) from the holder's long axis.

间接版把物体名换成描述词，裸名词 bar 也换成 stick（"...the long blue stick ... the narrow grey trough's long axis ..."）。

合拢时夹爪不在长条正上方时（agent 式错误），F2 改说"the grasp point was … of the blue bar's centre, i.e. not over the bar"，不会说成"夹在长条上"（实测文本，见"已知问题"）。反馈里只有相对测量（角度、相对槽中心 / 长条中心的偏移，方向用机器人视角词），没有物体绝对坐标，也没有 β 本身。

结构化字段：`grasp_misalign`（合拢时钳口真实 yaw − 长条 yaw，弧度，模 π 到 ±90°；长条 yaw 取手指碰到之前的值）、`bar_turned_in_grasp`、`grasp_xy_offset`、`grasp_over_bar`、`release_misalign`（松手时长条 yaw − 槽 yaw）、`release_offset`（松手时长条中心 − 槽中心，xy）、`bar_end_misalign_deg`、`failure`。

## 参照策略
| 策略 | 做法 |
|---|---|
| oracle | 所有指令 yaw 减去真实 β |
| naive | 指令 yaw = 物体真实朝向（修正 0） |
| adaptive | 维护 yaw 修正：每次失败后把 F2 的 `release_misalign` 累加进去（长条没被放下过时用 `grasp_misalign`），下一次所有指令 yaw 减去它 |
| blind | 不看历史；每次重试给 yaw 修正加 N(0, 4°) 的随机量 |

## headroom 验证（`scripts/validate.py --task l1_yaw_bias --n 20 --k 5`，seed 1000–1019，实测）
| 策略 | succ@1 | @2 | @3 | @4 | @5 |
|---|---|---|---|---|---|
| oracle | 100 | 100 | 100 | 100 | 100 |
| naive | 0 | 0 | 0 | 0 | 0 |
| adaptive | 0 | 100 | 100 | 100 | 100 |
| blind | 0 | 0 | 0 | 0 | 0 |

- naive 20/20 都是长条斜搭在槽上（`missed_place`），松手夹角 = β（偏差 ≤ 0.2°）。
- adaptive 20/20 在第 2 次成功；blind（每次 N(0, 4°) 抖动）0/20。
- 用时 360 s（单进程）。结果：`results/l1_yaw_bias/validate.json`，初始 / naive 结束画面 `results/l1_yaw_bias/*.png`。

## 物理标定（实测，2026-10-06/07）
- 对齐容差（4 个布局 seed 1000–1003，残差 = β + 修正）：−5°…+5° 全部成功（4/4）；−6° 3/4、+6° 4/4；±8° 各 1/4；±10° 0/4。几何上 14 sin θ + 1.8 cos θ ≤ 3.2 cm 给出约 ±5.6°，实测与之一致。所以 β ≥ 15° 离容差很远，blind 的 N(0, 4°) 抖动碰不上。
- 脚本 adaptive 第 2 次的 β 估计误差：平均 0.03°、最大 0.06°（20 个实例）。
- 长条摩擦：用 Panda 指垫默认 μ = 2 时，斜着合拢的钳口用指垫角把长条"卡"在斜的姿态，合拢时只扳过 3–13°（β = 15–25°），之后在提起 / 搬运中继续转，扳转量随 β 和接触细节变化。把长条摩擦设成 0.5 并给 geom priority（所有接触都用 0.5，包括与指垫的接触）后，合拢时长条被完全扳正（扳转角 = β）。
- 关节限位：Panda 第 7 关节在真实 yaw 约 −137° 处到限位（实测：β = −30° 时指令 −113°，只到真实 −137°、读数 −107°，读数和指令对不上，等于泄露 β）。所以 `YawSkills.wrap_yaw` 把 ±113° 的窗口加在**真实** yaw 上（指令 + β），在窗口里选离当前最近的模 180° 等价角。

## 设计决定与理由
1. **失败出现在"放"而不是"抓"。** 钳口转了 20° 照样能夹起长条（原装 8 cm 行程，长条 1.8 cm 宽）；转角误差的后果是长条在手里歪了，放进窄槽时才暴露。这对应"扁物体插进窄槽时被卡"。也考虑过"短行程夹爪 + 跨短边抓宽盒子，斜了手指落在盒顶抓空"，但那和 l1_bias_place 的失败方式（手指压在盒顶）完全一样，只是隐变量从平移换成转角，所以没选。
2. **长条摩擦 0.5。** 见"物理标定"：μ = 2 时松手时的夹角只有 β 的一部分，F2 里两个角度对不上，脚本学习者要 3–4 次才收敛。改成 0.5 后合拢即扳正，两个角度都等于 β。
3. **转角偏差通过 `R0` 实现**，而不是在任务里另做一层 yaw 映射。agent harness（`agent/tools.py`）报告的 yaw 是 `eef_mat @ sk.R0.T`，把 `R0` 换成"机器人以为的 yaw 0"后，控制和读数自动一致，不用改任何共享文件。
4. **重置后先把手腕转到机器人自己的 yaw 0。** 否则开局读数是 −β，直接泄露。
5. **β 范围 15–30°，长条 / 槽朝向 ±45°。** 脚本用到的指令 yaw 在 ±45° 内、真实 yaw 在 ±75° 内；agent 给的任意 yaw 由真实 yaw 窗口兜底（设计决定 / 物理标定里的关节限位）。
6. **槽是 fixture。** 长条砸上去不会把槽推动，成功判据可以用槽的固定位姿。
7. **`holding()` 对长条改用"两根手指都接触"**，和成功判据一致；agent 的 `holding_object` 走这个判断。

## 已知问题 / 效度威胁
实测：
- naive 的失败方式 20/20 都是长条搭在槽上（`missed_place`），松手夹角 = β；blind 0/20。
- 开局图像（seed 1004，β = 0 / +25° / −25° 对比渲染）：agentview 里手的朝向差别能看出来但不明显；腕部图像整体转了 β（物体朝向、桌面木纹方向都跟着转）。单看一张图看不出 β（物体本身朝向随机），但如果 agent 拿 agentview 里槽的方向和腕部图里槽的方向对比，或者拿"yaw 0 时手指沿世界 y 合拢"的定义去对 agentview 里手的朝向，动手前就可能估出 β。
- 夹爪不在长条上方时的 F2（seed 1000，测试脚本手动合拢，偏 8 cm）："When the jaws closed, the grasp point was 8.0 cm to the robot's left (+y) and 0.4 cm backward (-x, toward the robot) of the blue bar's centre, i.e. not over the bar, and the jaws were rotated 19 deg clockwise ... from square across the bar, but the bar was not picked up."

推测 / 未验证（没用真实 VLM 测过）：
- **放之前的视觉检查能当场避免失败。** 夹着长条移到槽上方后，腕部图里长条和槽明显成 15–30° 角；谨慎的 agent 会在松手前按图像转正，第一次就成功。这是"动作中感知"能力，第 1 次成功率要单独报告。
- **同一次尝试内的恢复。** 长条搭在槽壁上后，agent 可以在同一次尝试里重新抓起来转一下再放；cross 协议不禁止，attempt-1 成功率会包含它。
- 先验 agent 很可能不会"让 yaw 对齐槽长轴"而是保持抓取时的 yaw 直接放——那样放下时的夹角是 (槽 yaw − 长条 yaw) + β，失败原因混进了"没对齐槽"，F2 给的夹角仍然是正确的残差，学习信号不变。
- 跨实例记忆（mismatched）下 β 的符号随机、大小不同，错配历史会给出错误的修正方向，可用来检验增益是否来自历史内容。
- 当前只做 cross 协议。within 版本（搭在槽上后原地重抓再放）没单独实现。

## 交付
- 代码 `lffbench/tasks/l1_yaw.py`；验证 `results/l1_yaw_bias/validate.json`；演示视频 `videos/l1_yaw_bias_adaptive_s1000.mp4`（第 1 次长条斜搭在槽上 → F2 → 第 2 次指令 yaw 加 19.35° 成功；第 1 次移到槽上方时腕部画面里已能看到槽和长条成角）。
- agent 接口自测（`start_robot_session.sh … rgb`，seed 1000）：开局 `eef_yaw_deg` = 0.0，README 和 `status` 里没有偏差相关字样；move_to / close_gripper / done 正常，F2 文本正常。
