# L4-B `l4_diagonal_fit`：比盒子还长的木条要斜着放进去

> **2026-10-08 停用（用户决定"扔掉"），不进 cla4 及以后的评测。** 原因：cla2 里 Sonnet 4/4 个 run 第一次就成功（独立单次 3/3）。"直接斜放"是稳妥做法：顺放放得下的木条，斜放一定也放得下，所以不需要知道"木条比盒子长多少"就能成功，失败从来不会发生。代码和本卡保留作记录（`lffbench/tasks/l4_diagonal.py` 仍会被注册，只是不再列入任务表）。

> 2026-10-07。代码 `lffbench/tasks/l4_diagonal.py`（上一个子代理写了大半，本次补完：去掉 F2 里的绝对坐标和绝对朝向、关掉碰撞 mid-phase、加角度窗口和执行误差扫描、跑验证）。"实测"= 本机跑出来的；"推测"单独标出。

| 项 | 内容 |
|---|---|
| 类别 / 等级 | geometry_inference / L4 几何适配（通过尝试判断几何约束） |
| 能力 | Perceive（看出木条横架在盒沿上、没落到底）、Reason（"沿长边长了 X cm"→ 木条比盒子内长还长 → 只能转向对角线；由 X 算出木条长度和该转的角度） |
| 协议 | cross：每次尝试重置到同一实例，k = 5 |
| 指令 | 直接："Put the wooden bar into the box so that it lies flat on the bottom of the box." 间接："Put the long brown stick into the grey open tray so that it lies flat on the tray's floor." |
| 场景 | 一个灰色敞口盒（内尺寸 a × b，a = 11.5–13.0 cm，a − b = 0.4–1.9 cm，即接近正方形；壁高 3 cm、壁厚 6 mm），盒子朝向 ±45° 随机，位置 x ∈ [−0.22, −0.13]、y ∈ [−0.10, 0.10]；一根棕色木条 L × 1.2 × 1.4 cm 平躺在盒子前方（x ∈ [−0.01, 0.05]、y ∈ [−0.13, 0.13]，朝向随机），L = a + 1.0–1.9 cm（12.6–14.9 cm，比盒子内长长 7.8–15.1%）。盒子和木条尺寸每个实例重画（改 `sim.model` 里的 geom，一个 env 服务所有实例） |
| 隐藏信息 | 具体尺寸：木条比盒子内长只长 8–15%，顺放一定放不进；而对角线方向（离盒子长边 38.9–43.8°）放得进。一张图上不容易判断"顺放差多少、斜放够不够" |
| 成功判据 | 木条投影的四个角都在开口内（容差 2 mm）、木条底面离盒底 < 5 mm、木条倾斜 < 6°、夹爪已松开、盒子倾斜 < 10° |

## 任务 → 隐藏信息 → 所需能力 → 调整后的动作
- **任务**：把木条平放到盒子底上。
- **隐藏信息**：木条比盒子内长更长（差 1.0–1.9 cm），但比对角线短；对角线方向留有 1.1–1.9 cm 的总间隙。
- **所需能力**：从"架在盒沿上、沿盒子长边长了 X cm"里推出木条长度 ≈ 盒子内长 + X，意识到顺着任何一条边都放不进，只能转向对角线，并算出（或估出）角度。
- **调整后的动作**：同样的抓—搬—放，只是放之前把手腕转到离盒子长边约 40°（两个方向都行）。

## 为什么按先验做会失败
先验（naive）："整齐地放"——木条长边对准盒子长边。木条比盒子内长长 1.0–1.9 cm，所以 20/20 都横架在盒沿上，离盒底 3.0 cm（实测）。盲目重试（同样对齐，角度 ±4°、位置 ±3 mm 抖动）结果一样：20/20 × 5 次全失败。

## 要从失败里学到什么
F2 给出木条相对盒子的朝向和超出量。对齐时"沿长边长了 X cm"直接给出 L ≈ a + X；然后找让两个方向都放得下的角度（≈ 40°）。如果第二次角度偏了，反馈会给出这时沿哪条边超了多少，可据此往另一边修。

## F2 反馈示例（脚本跑出来的原文，只有相对测量，没有坐标和绝对朝向）
- 顺放（naive）："The bar did not go in: it is lying across the top of the box on its rim, 3.0 cm above the box floor. The bar lies along the box's long side (within 2 degrees). At that angle the bar's footprint is 1.2 cm longer than the inside of the box along the box's long side."
- 转得太多："The bar did not go in: it is lying across the top of the box on its rim, 3.0 cm above the box floor. The bar is turned about 53 degrees clockwise (towards the right), seen from above, from the box's long side. At that angle the bar's footprint is 0.6 cm longer than the inside of the box along the box's short side."
- 角度刚好擦边、一头掉进去："The bar did not go in: it is tilted about 12 degrees, with one end down inside the box and the other end on the rim. The bar is turned about 53 degrees counterclockwise (towards the left), seen from above, from the box's long side. At that angle the bar's footprint is smaller than the opening (spare 3.1 cm along the long side, 0.2 cm along the short side), but it caught on the rim (bar centre within 0.5 cm of the box's centre)."
- 没动木条就结束（seed 1000，自测时实测）："The bar is not in the box: it is lying on the table, its centre 23.6 cm forward and 2.9 cm to the left of the box's centre."（方向词按机器人视角）
- 成功："The bar is lying flat on the bottom of the box."

结构化字段（脚本 adaptive 用）：`feedback = {rel_deg（木条与盒子长边的夹角，折到 0–90°）, rel_signed_deg, over_long_cm, over_short_cm, spare_long_cm, spare_short_cm, centre_offset_cm}`，`failure ∈ {rim, off, table, held, None}`。

## 脚本参照策略
| 策略 | 做法 |
|---|---|
| oracle | 知道 a、b、L：转到使两个方向间隙中较小者最大的角度（38.9–43.8°），符号取离木条当前朝向近的那个 |
| naive | 长边对齐（rel_yaw = 0） |
| adaptive | 只通过带噪声的感知知道盒子内尺寸和木条宽度（σ = 3 mm / 1.5 mm），不知道木条长度。每条"在角度 d 沿长边 / 短边长了（或富余）X cm"都给出一个 L 的方程（L cos d + W sin d = a + X 或 L sin d + W cos d = b + X），取最大的估计（保守），转到该长度下间隙最大的角度 |
| blind | 不看历史：仍然对齐，角度加 N(0, 4°)、位置加 N(0, 3 mm) |
| random_yaw（敏感性对照） | 第 1 次同 naive，之后每次随机一个朝向（−90°～90° 均匀），不看历史 |

执行（所有策略相同）：在木条中点横跨夹住（夹爪点离桌面 1.05 cm）→ 抬到 15 cm → 到盒子上方转腕 → 闭环把木条中心伺服到盒子中心、木条底面高出盒沿 1 cm → 松手。只用 move_to 和开合夹爪。

## headroom 验证（`python scripts/validate.py --task l4_diagonal_fit --n 20 --k 5 --kinds oracle,naive,adaptive,blind,random_yaw`，seed 1000–1019，实测，用时 458 s）
| 策略 | @1 | @2 | @3 | @4 | @5 |
|---|---|---|---|---|---|
| oracle | 100 | 100 | 100 | 100 | 100 |
| naive | 0 | 0 | 0 | 0 | 0 |
| adaptive | 0 | **100** | 100 | 100 | 100 |
| blind | 0 | 0 | 0 | 0 | 0 |
| random_yaw | 0 | 25 | 35 | 50 | 50 |

（succ@k 按 run 计，%。验收线：oracle@1 ≥ 95 ✓，naive@1 ≤ 30 ✓，adaptive@5 ≥ 90 ✓，blind@5 远低于 adaptive ✓。）
- adaptive 20/20 都是第 2 次成功：一次"沿长边长了 X cm"就足够估出 L 并转到对的角度。
- random_yaw 说明：不推理、只是换着角度乱试，5 次内也只有一半能碰上（每次命中概率约 2 × 21° / 180° ≈ 23%）。

## 物理可行性和标定（实测，2026-10-07，`results/l4_diagonal_fit/`）
- `physics_check.json`（12 个实例 × 6 种放法）：顺放（0°）和横放（90°）全部架在盒沿上（离盒底 3.0 cm），最佳角 ±θ 全部成功，最佳角 ±6° 全部成功；0 个不符。
- `window_check.json`（6 个实例，木条居中，最佳角 −16°～+16° 每 2° 一次）：成功角度区间宽 18–28°（最佳角 −8～−14° 到 +8～+12°），和几何间隙 ≥ 0 的区间一致（间隙 −0.1 mm 时也有一次成功，属于 2 mm 判据容差内）。
- `robustness_check.json`（6 个实例 × 11 种执行误差，都在最佳角附近）：中心沿长边 / 短边偏 4 mm、7 mm，角度偏 ±4°、±8°，从盒沿上方 3 cm 松手，以及偏 4 mm + 偏 4° + 3 cm 松手，66/66 全成功。对 agent 来说，角度大致对（±8°）、对准盒子中心（±7 mm）就够了。
- 碰撞 mid-phase：本任务运行时改盒子和木条的 geom 尺寸，MuJoCo 编译时算好的包围盒层次（`bvh_aabb`）不会跟着变。在 l4_nest_order 上这导致手指穿过缩小后的杯壁；这里我对比了开 / 关 mid-phase 的角度窗口扫描（3 个实例 × 17 个角度），结果完全一样，但为保险还是关掉了（`apply_instance` 里 `disable_midphase`）。

## agent 接口自测（实测，2026-10-07，端口 9861，`start_robot_session.sh l4_diagonal_fit 1000 full F2 9861 direct 3 selftest rgb 1`）
- README 只有通用说明和指令原文，没有尺寸、角度、"对角线"之类的提示。`./robot status / move_to / close_gripper / open_gripper / done` 都正常，报告里 `holding_object` 在夹起木条后为 true。
- 第 1 次只移动、合爪就 `done`：反馈的文字分支选错了（木条躺在桌上却说"partly on the box's rim"，判断"在桌上"的条件写错了），已修，修后输出见上面的示例。
- 第 2 次用 `move_to` 开环（不伺服）抓起木条、对齐盒子长边放下：反馈和脚本一致（"lying across the top of the box on its rim, 3.0 cm above the box floor … 1.2 cm longer … along the box's long side"）。
- 第 3 次同样开环、手腕转到离长边 −41°：成功（"The bar is lying flat on the bottom of the box."）。说明不需要脚本里的闭环伺服，agent 用工具就能做成。
- 服务器已 kill，`runs/claude/selftest_l4_diagonal_fit_*` 和沙箱已删。

## 已知问题 / 效度威胁
实测：
- naive / blind 都是 0%，第 2 次的 adaptive 是 100%：一次失败的信息就足够，这个任务主要测"能不能从'长了 X cm'想到转向对角线"，不测多步收敛。
- 自测里发现并修了一个反馈分支错误（木条在桌上被说成"partly on the rim"），只影响 agent 没把木条放到盒子上的情况；脚本策略从不出现这种情况，所以验证数字不受影响。
- F2 里"盒子长边"是相对盒子自己说的；盒子接近正方形（长短边只差 0.4–1.9 cm），agent 从图上可能分不清哪边是长边。对齐那次的反馈会说"lies along the box's long side"，agent 由此知道自己对准的是长边。

推测（未验证）：
- 木条比盒子长 8–15%，细心的 agent 从图像上可能一开始就看出来、直接斜放，第 1 次成功率要单独报告。
- 木条只有 1.2 cm 宽、1.4 cm 高，agent 下探到 z ≈ 0.91 去夹，工作空间下限是 0.905（`agent/tools.py`），余量小；夹偏了木条可能在手里转，松手时角度会变。成功角度窗口有 18–28°，大概能吸收。
