# L5 `l5_slippery_lift`：提起又重又滑的金属棒，并保持水平

> 状态（2026-10-07）：已实现并通过 headroom 验证（n = 30）。代码 `lffbench/tasks/l5_slippery_lift.py`。

| 项 | 内容 |
|---|---|
| 类别 / 等级 | dynamics_adaptation / L5（"搞清楚这个物体怎么动"） |
| 能力 | Perceive（看出棒往哪头歪）、Reason（哪头沉 → 重心在抓点哪一侧）、Utilize（下一次把抓点往重的一头挪） |
| 协议 | cross（每次尝试重置到同一实例、同一隐变量），k = 5 |
| 指令（direct） | Pick up the metal bar and hold it level (tilted less than 8 degrees), with its lowest point at least 10 cm above the table. Keep holding it when you finish. |
| 指令（indirect） | Pick up the long grey rod lying on the table and hold it horizontal (…)… |
| 动作 | 只用 move_to / close_gripper：张爪 → 横跨棒子（yaw 90°，两指沿 x 合拢）下到棒中高 → 合爪 → 提 15 cm |

## 任务 → 隐藏信息 → 所需能力 → 调整后的动作
- **任务**：桌上横放一根 4 × 28 × 4 cm 的灰色金属棒（长轴沿机器人左右方向 y），要求提起来并保持水平（倾角 ≤ 8°、最低点离桌 ≥ 10 cm），done 之后仍夹着。
- **隐藏信息**：棒里藏了一块配重（占总质量 60%），重心偏离几何中点 e，|e| ∈ [2.5, 7] cm，偏左或偏右随机；总质量 0.8–1.2 kg、指–棒摩擦 μ 0.40–0.60（"又重又滑"）。棒外观完全均匀，动手前从 RGB 上看不出重心在哪。
- **为什么先验会失败**：先验是在棒的中点横抓。重心偏几厘米时，提起来的瞬间重力矩超过两个指垫能承受的摩擦力矩，棒在指间转动，重的一头垂下去，最后斜挂 30–60°。慢慢提没用（力矩是静态的），换更快的速度也没用。
- **要从失败里学到**：哪一头沉 → 重心在抓点的哪一侧；倾角越大大致说明偏得越远（实测：抓中点时 e≈2.6 cm → 31°，e≈4 cm → 34–40°，e≈5–6 cm → 45–55°，e≈6.6 cm → 56–61°），但到 2 cm 以上就趋于饱和，所以具体挪多少要靠两三次尝试夹逼。
- **调整后的动作**：把抓点往沉的那头挪，直到离隐藏重心 ≲ 1.7 cm（实测窗口）。

## F2 反馈示例（实测文本）
- 失败："Attempt failed. The bar did not stay level: it rotated in the gripper while it was being lifted and now hangs tilted by 53 degrees, its right end lower than its left end. You grasped the bar 0.1 cm to the right (-y) of its midpoint. Its lowest point is 3 cm above the table."
- 成功："Attempt succeeded. The bar is held level (tilt 0 degrees), 15 cm above the table. You grasped the bar 6.1 cm to the right (-y) of its midpoint."
- 其他失败分支：棒滑脱掉回桌面（dropped）、根本没夹住（no_grasp）、水平但不够高（too_low）。
- 反馈只给相对测量（倾角、哪头低、抓点相对棒中点偏多少、往机器人哪个方向），不给绝对坐标，不给重心位置、质量、摩擦的真值。

## 物理标定（实测，2026-10-07，脚本在 scratchpad `l5dyn/bar_win.py`、`bar_lift.py`）
- 质量 0.8/1.2 kg × μ 0.4/0.6，提 15 cm、0.15 m/s：|抓点 − 重心| ≤ 1.0 cm 倾角 ≤ 2°；1.5 cm 时 0–41°（临界）；≥ 2 cm 多为 30–50°。保持 3 s 倾角不再变化（不蠕变）。
- μ = 0.8 时倾角对偏移不单调（例如偏 3 cm 只歪 2°），所以 μ 上限取 0.6。μ = 0.3 + 1.3 kg 时即使抓在重心也整根滑脱，所以 μ 下限取 0.4、质量上限 1.2 kg。
- 隐藏重心用"显式 inertial + 运行时改 body_ipos/body_inertia"实现；编译时 inertial 的 pos 故意不为零（否则 MuJoCo 的 sameframe 优化会忽略 ipos）。
- 上一个子代理的"快提会滑、慢提能成"纯平移打滑方案已实测不成立（见下方"放弃的方案"），本任务换成了转动打滑（力矩），和速度无关。

## headroom 验证（n = 30，k = 5，脚本参照策略，实测）
`python scripts/validate.py --task l5_slippery_lift --n 30 --k 5` → `results/l5_slippery_lift/validate.json`

| 策略 | @1 | @2 | @3 | @5 |
|---|---|---|---|---|
| oracle（抓在真实重心） | 100% | | | 100% |
| naive（抓几何中点） | 0% | | | 0% |
| adaptive（按哪头低夹逼抓点：单侧有界先挪 3 cm，两侧有界取中点，边界留 0.8 cm 余量） | 0% | 47% | 100% | 100% |
| blind（中点 ± 1 cm 随机重试） | 0% | 3% | 3% | 10% |

## agent 接口自测（实测，2026-10-07，端口 9880，rgb 模式）
`start_robot_session.sh l5_slippery_lift 1000 full F2 9880 direct 3 selftest rgb 1`：README 只有通用说明和任务指令，没有重心/质量/摩擦等隐变量；
第 1 次在棒中点横抓（yaw 90）提 15 cm → 反馈 "…hangs tilted by 54 degrees, its right end lower than its left end. You grasped the bar 0.2 cm to the right (-y) of its midpoint. Its lowest point is 3 cm above the table."，agentview 图里能清楚看到右端下垂；
第 2 次抓点右移 5.2 cm → "Attempt succeeded. The bar is held level (tilt 0 degrees), 14 cm above the table. …"。无报错。
小问题：棒歪着挂在指间时，move_to 报告里的 `holding_object` 可能是 false（robosuite 的抓取判据要求两个指垫都接触），任务的 outcome 用的是 1.5 s 保持后的判定，不受影响。

## 演示视频
`videos/l5_slippery_lift_adaptive_s1000.mp4`（seed 1000：e = −5.2 cm；第 1 次抓中点歪 53°，第 2 次右移 3.1 cm 仍歪 40°，第 3 次右移 6.1 cm 水平提起）。

## 已知问题 / 威胁（推测，未验证的标【推测】）
- naive@1 = 0%：所有实例的 |e| ≥ 2.5 cm，抓中点必失败。这是有意的（隐变量总是起作用）；如果希望有少量"第一次就成"的实例，可以把 e 下限降到 1 cm。
- agent 在一次尝试内可以"提一下、看歪向、放下重抓"，这属于尝试内学习，会抬高 attempt-1 成功率；cross 协议下这是合法行为，评测时应单独统计。【推测】强 agent 会这么做。
- 倾角大小对偏移量只有粗信息（2 cm 以上饱和），agent 需要 2–3 次夹逼；脚本 adaptive 3 次内全部成功。
- 夹爪 yaw 必须横跨棒子（90°）；若 agent 沿棒长方向合爪会夹不住，属于 agent 的感知/规划失误，不是本任务的隐变量。
- 和 l7_offcenter_push 的隐变量同类（重心沿长边偏移），但交互方式不同（提起时的转动打滑 vs 推动时的平面转动）。

## 放弃的方案（上一个子代理，2026-10-06 实测）
"快提会滑、慢提能成"的纯平移打滑：5×5×12 cm、1 kg 方盒，从顶部往下 3 cm 夹住竖直提 12 cm，隐变量 κ = 2·F·μ/(m·g)。

| κ | 0.03 | 0.08 | 0.15 | 0.25 | 0.35 | 0.45 | 0.6 | 最快 | (m/s；数字 = 盒在指间下滑 cm，X = 掉落) |
|---|---|---|---|---|---|---|---|---|---|
| 0.98 | X | 1.2 | X | X | X | X | X | X | 位置 (-0.10, 0) |
| 1.02 | 0 | 0.1 | 0.5 | X | X | X | X | X | |
| 1.06 | 0 | 0 | 0.1 | 0.2 | 2.2 | 3.1 | 3.4 | X | |
| 1.10 | 0 | 0 | 0.2 | 1.1 | 0.1 | 0.4 | 0.5 | 0.9 | |
| ≥1.15 | 0 | 0 | 0 | 0 | 0 | ≤0.2 | ≤0.2 | ≤0.2 | |
| 1.06 @ (0.05, 0.18) | 0 | 0 | 0 | 0.2 | 1.6 | X | X | X | 换个位置阈值就变 |

结论：OSC 最快提升时末端峰值加速度只有约 4.4 m/s²（0.45 g），"快提滑、慢提不滑"只在 κ ∈ [1.0, 1.1] 这一条 10% 的窄带里出现；带内结果不单调，且随桌面位置变化。水平搬运（0.05–0.6 m/s，κ 1.1–3）完全不滑。所以该方案对微小扰动极敏感，不可用。
