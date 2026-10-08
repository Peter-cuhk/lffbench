# L1-B `l1_height_bias`：机械臂竖直方向有标定偏差时，从木块顶上拿起薄瓷片放到盘子里

> 状态：已验证（2026-10-07）。标"实测"的数字都来自当前代码（`lffbench/tasks/l1_height.py`）。

## 一句话（任务 → 隐藏信息 → 所需能力 → 调整后的动作）
- **任务**：把木块顶上的红色薄瓷片（1 cm 厚）拿起来放到盘子上。
- **隐藏信息**：机械臂竖直方向有固定标定偏差 dz（2–3.5 cm，向上）：夹爪实际停在指令高度上方 dz，本体读数却说"到了"。
- **所需能力**：看出第一次是"手指在瓷片上方空合"（不是没对准、不是夹不住），并从 F2 的高度测量里算出偏差（Perceive）；把偏差带到下一次（Utilize）。
- **调整后的动作**：同一种抓法，所有指令高度整体下移 dz（抓和放都要减）。

| 项 | 内容 |
|---|---|
| 类别 / 等级 | system_identification / L1 动作微调（方法和理解都对，只是机器人自己的竖直执行有偏差） |
| 能力 | Perceive, Utilize |
| 协议 | cross：每次尝试重置到同一初始状态、同一偏差 dz，k = 5 |
| 指令（直呼） | "Pick up the red tile from the top of the wooden block and put it on the plate." |
| 指令（间接） | "Pick up the thin flat red slab lying on the large wooden pedestal and put it on the round white dish." |
| 场景 | 木色方块（fixture，顶面 15 × 17 cm，高 6–10 cm 随机）上平放一块红色薄瓷片（生成物体，8.0 × 4.5 × **1.0 cm**，72 g，yaw ±40°，相对方块中心 ±2 cm）；桌上 LIBERO `plate`（直径 13.75 cm）、干扰物 `butter`、`chocolate_pudding`。方块中心 x ∈ [−0.10, 0.02]、y ∈ [−0.17, 0.17]；盘子离方块 ≥ 24 cm 且 \|Δy\| ≥ 12 cm（不躲在方块后面）；干扰物离方块 ≥ 16 cm、离盘子 ≥ 12 cm、离"起始位→瓷片""瓷片→盘子"两条直线 ≥ 9 cm |
| 机器人 | 原装 Panda 夹爪（8 cm 行程），不改 |
| 隐变量 | 固定竖直偏差 dz ~ U[2.0, 3.5] cm，**只向上**：夹爪实际停在指令高度上方 dz（`Skills(bias=(0, 0, dz))`）。agent 看不到 dz；回报给 agent 的末端位置在指令系（真值 − bias；harness 的 `proprio()` 和本任务 `HeightSkills.move_to` 的报告都是） |
| 成功判据 | 瓷片中心到盘子中心水平距离 < 4 cm、瓷片中心低于盘沿 + 1.5 cm、盘子直立在桌上、瓷片不在夹爪里 |

## 为什么先验会失败
先验 = 完美感知（瓷片、方块顶面、盘子的真实位置）+ 标准动作（yaw 对齐瓷片长边、跨 4.5 cm 短边从上往下抓；夹爪点放在方块顶面上方 1.05 cm，即指尖刚好离开方块顶面约 1 mm → 合拢 → 升起 → 移到盘子上方、瓷片底面高过盘沿 1.2 cm 处松开），但不知道 dz。

- 夹爪实际停在指令高度 + dz。瓷片只有 1.0 cm 厚，手指要和瓷片重叠约 4 mm 才夹得住（实测，见"物理标定"）；dz ≥ 2 cm 时指尖整体高过瓷片顶面 1.3–2.8 cm（实测，20/20 个 naive），夹爪在瓷片上方空合。
- dz 下限取 2 cm 是为了让"瞄得更低"的先验也失败：即使 agent 把夹爪点指令到方块顶面那么低（指尖插进方块约 1 cm），实际指尖也在方块顶面上方 ≥ 1.05 cm，不低于瓷片顶面，照样空合（按实测容差推算）。
- 失败在图像里看得见：腕部相机里两根手指在瓷片正上方合到一起、瓷片还在下面；升起后瓷片仍在方块上。agent 的执行报告里也有 `gripper_width_m` ≈ 0 和 `holding_object: false`。

## 要从失败里学到什么
F2 报告"夹爪合拢时，夹爪点（两指尖中点）在瓷片顶面上方多少 cm，手指最低点在瓷片顶面上方多少 cm"。agent 知道自己本来打算把夹爪点放在哪，两者之差就是 dz（或者：指令高度 − 报告的高差 = 瓷片顶面在机器人自己坐标系里的高度，下一次直接按这个高度抓）。一次失败后就能估计出 dz，下一次把抓和放的所有指令高度都减去它。注意 RGB-only 模式下 agent 本来就不知道方块精确多高，F2 给的是相对瓷片的高差，所以"感知误差 + dz"会被一起校正。

## F2 反馈（实测文本，seed 1000，dz = 2.57 cm；`outcome()["detail"]`）
第 1 次（naive，失败）：
> Attempt failed. When the jaws closed, the grasp point (midway between the fingertips) was 2.5 cm above the top face of the red tile and the lowest points of the fingertips were 1.6 cm above it (the tile is 1.0 cm thick), and the tile was not picked up. The fingers closed in the air above the tile without touching it. The tile is still on the wooden block.

第 2 次（adaptive 把指令高度下移 2.46 cm，成功）：
> Attempt succeeded. The red tile is on the plate. When the jaws closed, the grasp point (midway between the fingertips) was 0.4 cm above the top face of the red tile and the lowest points of the fingertips were 0.5 cm below it (the tile is 1.0 cm thick).

间接版（`feedback(..., indirect=True)`）把物体名换成描述词，裸名词 tile 也换成 slab，例如 "...2.5 cm above the top face of the thin red slab ... The slab is still on the large wooden pedestal."

反馈里只有相对测量（高差、相对盘子中心的偏移，方向用机器人视角词），没有物体绝对坐标，也没有 dz 本身。

结构化字段（脚本 adaptive 用的就是这些）：`grasp_height`（合拢时夹爪点 − 瓷片顶面，米；顶面取手指碰到瓷片之前的位置）、`fingertip_gap`（手指最低点 − 瓷片顶面）、`grasp_xy_offset`、`touched_at_close`、`release_offset`、`failure ∈ {missed_grasp, dropped, missed_place, not_released, None}`。

## 参照策略
| 策略 | 做法 |
|---|---|
| oracle | 所有指令高度减去真实 dz |
| naive | 指令高度 = 按真实几何算的标准高度（修正 0） |
| adaptive | 维护高度修正：每次失败后把 F2 的 `grasp_height` 减去自己本来打算的高度（1.05 − 1.0 = 0.05 cm），累加进修正，下一次所有指令高度减去它 |
| blind | 不看历史；每次重试给高度修正加 N(0, 5 mm) 的随机量 |

## headroom 验证（`scripts/validate.py --task l1_height_bias --n 20 --k 5`，seed 1000–1019，实测）
| 策略 | succ@1 | @2 | @3 | @4 | @5 |
|---|---|---|---|---|---|
| oracle | 100 | 100 | 100 | 100 | 100 |
| naive | 0 | 0 | 0 | 0 | 0 |
| adaptive | 0 | 100 | 100 | 100 | 100 |
| blind | 0 | 0 | 0 | 0 | 0 |

- naive 20/20 都是空合（`missed_grasp`），合拢时手指最低点在瓷片顶面上方 1.3–2.8 cm。
- adaptive 20/20 在第 2 次成功（一次失败就够）；blind（每次 N(0, 5 mm) 抖动）0/20。
- 用时 276 s（单进程）。结果：`results/l1_height_bias/validate.json`，初始 / naive 结束画面 `results/l1_height_bias/*.png`。

## 物理标定（实测，2026-10-06/07）
- 竖直容差（6 个布局 seed 1000–1005，残差 = 实际夹爪点高度 − 标准高度 1.05 cm）：残差 −2.0 / −1.0 / 0 / +0.4 cm 全部成功（6/6）；+0.6 cm 1/6；+0.7、+0.8、+0.9、+1.0、+1.2、+1.5 cm 全部抓空（0/6）。即指尖要比瓷片顶面低约 4 mm 以上才夹得住；向下 2 cm（指尖压进方块顶面）仍然成功，因为 OSC 是柔顺的。
- 脚本 adaptive 第 2 次的 dz 估计误差：平均 2.1 mm、最大 2.5 mm（20 个实例；来自合拢时 OSC 的跟踪误差），远小于 4 mm 的余量。
- 标准高度下 robosuite 的 `_check_grasp` 能识别瓷片被夹住（3/3）；但偏低（只夹住手指网格下部）时它会漏报，所以 `HeightSkills.holding()` 对瓷片改用"两根手指都接触瓷片"，agent 的 `holding_object` 也走这个判断。

## 设计决定与理由
1. **瓷片放在木块顶上，而不是桌面上。** agent 的 `move_to` 会把指令目标裁到 z ≥ 桌面 + 0.5 cm（`agent/tools.py` 的 `WS_LO`）。瓷片放桌上时，要补偿 2–3.5 cm 的向上偏差，指令就得低于桌面，接口不让，任务变成不可解。放在 6–10 cm 高的木块上，补偿后的指令仍在工作空间内。木块是 fixture（不会被推动），高度每个实例随机，防止 agent 记住绝对高度。
2. **偏差只向上。** 向下的偏差不会失败（见"物理标定"：−2 cm 仍成功）。所以随机的只有大小，方向固定；agent 不知道这一点，也不知道大小。
3. **dz 范围从 1.5–3 cm 改成 2–3.5 cm（2026-10-07）。** 实测容差只有 +0.4 cm（不是早先记的 0.7 cm）。1.5 cm 时，一个把夹爪点瞄在瓷片中间高度或方块顶面的 agent 先验，第一次可能碰巧成功；2 cm 起这些先验也会失败。
4. **瓷片厚 1.0 cm。** 越薄，竖直容差越小。用户原话里的"卡片"（1–3 mm）用 Panda 平行夹爪从平面上夹不起来，所以取"薄瓷片 / 杯垫"量级。
5. **不改夹爪。** 竖直方向的失败和横向开口无关，用原装 8 cm 行程；失败只来自高度。这和 l1_bias_place（短行程夹爪 + 水平偏差）区分开。
6. **F2 只给相对瓷片顶面的高差**，不给 dz、不给绝对坐标。agent 要自己把"我指令的高度"和"实际高差"对起来。

## 已知问题 / 效度威胁
实测：
- naive 的失败方式在 20/20 个实例里都是"空合"（`missed_grasp`，手指最低点在瓷片顶面上方 1.3–2.8 cm）；blind 的 ±5 mm 抖动一次都没碰上（0/20）。
- 正面 agentview 里手掌挡住瓷片上方，1–3 cm 的竖直间隙不算显眼；腕部相机看到的是"手指在瓷片上方合拢、瓷片还在"。F0/F1 条件下 agent 能知道"没夹住、高了"，大小要靠试。
- agent 接口自测（`start_robot_session.sh … rgb`，seed 1000）：README 和 `status` 里没有偏差相关字样；手动按"夹爪点比方块顶面高 0.2 cm"指令（比脚本 naive 更低的先验），F2 回报夹爪点在瓷片顶面上方 1.8 cm、手指最低点高 0.9 cm，抓空——更低的先验同样失败。
- 夹爪不在瓷片上方时 F2 不会误说"在瓷片上方空合"（测试脚本，seed 1000，前移 30 cm 合拢）："…horizontally the grasp point was 0.7 cm to the robot's left (+y) and 22.9 cm forward (+x, away from the robot) of the tile's centre, i.e. not over the tile, and the tile was not picked up. The gripper was not over the tile when it closed." 水平偏差 ≥ 1 cm 时都会附上这一句。

推测 / 未验证（没用真实 VLM 测过）：
- **接触探测能绕过偏差。** agent 可以张着爪、跨着瓷片一点点往下走，直到 `touched_object` 变成 true（指尖碰到方块顶面），再在那儿合拢——这样第一次就能成功，不需要失败。这是一种主动感知能力，第 1 次成功率要单独报告。
- **同一次尝试内的恢复。** cross 协议下一次尝试里 agent 可以多次开合：空合后看到 `gripper_width_m` ≈ 0，原地往下再抓一次。这仍然是"从失败里学"，但发生在第 1 次尝试内，attempt-1 成功率会包含它。
- **RGB-only 下方块高度要从图像估计**，估计误差和 dz 同量级（厘米级）。高估时不管 dz 都会失败，低估时会部分抵消 dz；F2 校正的是两者之和，学习信号不变，但"第一次为什么失败"的诊断可能被归因到感知。
- 偏差只向上：跨实例记忆（mismatched）下 agent 可能学到"往下补"的通用先验。
- 本体读数是"真值 − dz"，所以重置时的末端读数 z 随实例的 dz 变（真实 home 位姿固定）。单个实例里看不出来；跨实例记忆下，比较开局读数原则上能看出 dz 的差别（l1_bias_place 的 xy 也有同样的问题）。

## 交付
- 代码 `lffbench/tasks/l1_height.py`；验证 `results/l1_height_bias/validate.json`；演示视频 `videos/l1_height_bias_adaptive_s1000.mp4`（第 1 次空合 → F2 → 第 2 次下移 2.46 cm 成功）。
