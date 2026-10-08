# L7 `l7_offcenter_push`：把重心偏的长盒子一次推进目标框

代码：`lffbench/tasks/l7_offcenter_push.py`　结果：`results/l7_offcenter_push/validate.json`　视频：`videos/l7_offcenter_push_adaptive_s1000.mp4`

| 项 | 内容 |
|---|---|
| 类别 | failed_interaction_inference（L7，"搞清楚为什么这次接触没起作用"），`level="L7"` |
| 能力 | Perceive（看出盒子转了、往哪边转）、Reason（转向 → 哪一端重）、Utilize（下次把接触点挪向重的一端） |
| 协议 | cross：每次尝试重置到同一实例，k = 5 |
| 场景 | 长盒子 22 × 9 × 4 cm（长边沿 y，0.6 kg，μ 0.5），顶面中间一条对称的浅色胶带；正前方（+x）8–13 cm 处一个绿色目标框（盒子外廓每边加 2 cm，只是标记，不参与碰撞）。盒子位置 x ∈ [-0.20, -0.14]、y ∈ [-0.08, 0.08] 随机 |
| 规则 | 指令里写明并在 outcome 里强制：**只能推一次**——盒子只能被一条 move_to 推动（某条 move_to 让盒子移动 > 1 cm 或转 > 3° 就算一次推；超过一次直接判失败）。否则 agent 可以在一次尝试里边推边纠偏，就不再考"跨尝试学到了什么" |
| 成功 | 盒子四个角都在框内、没翻倒、只推了一次 |

## 任务 → 隐藏信息 → 所需能力 → 调整后的动作
- **任务**：合爪，从盒子后面把它直着往前推进框里，不能转。
- **隐藏信息**：盒子重心沿长边偏了 e，|e| ~ U[4.5, 7.5] cm，左右随机（运行时改 `body_ipos`）。外观对所有 e 完全一样，动手前从 RGB 看不出来。
- **先验为什么失败**：自然做法是推背面正中。桌面摩擦合力作用在重心下方，推力线不过重心 → 盒子边走边转，**重的一端落后**；转角 17–41°（验证集 20 个实例实测），框只容得下约 9°。
- **要从失败里学到**：转的方向 → 哪端重（落后的那端）；转得多 → 偏得多。**调整后的动作**：把接触点（夹爪中心沿长边的位置）挪向重的一端，推力线过重心附近就平移。

## F2 反馈（只给相对测量，不给绝对坐标、不给 e）
失败示例（seed 1000，e = −6.3 cm，推正中）：
> Attempt failed. The box is not completely inside the rectangle. It turned 30 degrees clockwise (seen from above): its right end ended 11.2 cm further back than its left end. Its centre ended 0.3 cm forward and 2.9 cm to the right of the rectangle's centre. When the box started to move, the gripper's centre was 0.6 cm to the right of the middle of the box (measured along the box's long side).

另外两种失败文字："The box was moved by N separate move_to commands; only one push is allowed." / "The box did not move: the gripper never pushed it."
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

## headroom 验证（n = 20，k = 5，实测，`python scripts/validate.py --task l7_offcenter_push --n 20 --k 5`）
| 策略 | @1 | @2 | @3 | @5 |
|---|---|---|---|---|
| oracle（推在重心 c = e） | 100% | | | 100% |
| naive（推正中） | 0% | | | 0% |
| adaptive（转向 → 往落后的一端挪 6 cm；两边都见过就二分；推直了再按前后误差修推距） | 0% | 100% | 100% | 100% |
| blind（推正中 ± 0.8 cm 随机） | 0% | 0% | 0% | 0% |

## agent 接口自测（端口 9920，rgb，F2，k=3，2026-10-07）
README 无泄露（只有指令和通用说明）；close_gripper / move_to / done 都正常；推正中得到上面那种 F2 文字，第二次尝试正常重置。

## 已知问题 / 推测（未验证的标【推测】）
- 脚本 adaptive 第 2 次就 100%，因为稳定带有 ±2 cm、e 的范围只有 3 cm 宽，一步 6 cm 总能落进去。对 VLM 难点在别处：从 RGB 估盒子位置、把"顺时针 30°、右端落后 11 cm"翻译成"往右挪多少"【推测】。
- agent 若用 yaw 90（两指沿 x，接触面只有一指宽）推，稳定带会更窄【推测，未测】。
- 推得太快盒子会滑过头（μ 0.5 时 0.6 m/s 约多滑 3.7 cm，按 v²/2μg 估算【推测，未测】）；指令里写了 "slowly"。
