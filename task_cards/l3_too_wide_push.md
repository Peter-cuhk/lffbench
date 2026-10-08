# L3-B `l3_too_wide_push`：抓不起来的宽盒子推进方框

> 代码：`lffbench/tasks/l3_wide.py`。验证：`results/l3_too_wide_push/validate.json`。视频：`videos/l3_too_wide_push_adaptive_s1000.mp4`。2026-10-07 完成。

| 项 | 内容 |
|---|---|
| 等级 / 类别 | L3 更换操作方式（`strategy_switching`） |
| 能力 | Perceive（看出手指落在盒顶、盒子没动）、Reason（为什么抓不起：盒子比夹爪最大开口宽，换方向只会更宽）、Plan（放弃抓取，改成合爪推，规划推的方向、距离和次数） |
| 协议 | cross，每次尝试重置到同一实例，k = 5 |
| 指令 | 直接："Put the brown box completely inside the green square marked on the table." 间接："Put the cardboard-coloured package completely inside the green outlined area on the table." |
| 场景 | 棕色盒子：窄边 8.3–8.9 cm、长边 11–13 cm、高 4 cm，0.40 kg，滑动摩擦 0.6（priority 覆盖桌面的 1.0）。绿色方框是画在桌上的标记（无碰撞），内边 18 × 18 cm |
| 随机 | 盒子尺寸；盒子位置；朝向（长边大致沿 x 或沿 y，±10°）；方框在盒子的前 / 后 / 左 / 右 16–22 cm 外，横向再偏 ±3 cm |
| 成功判据 | 盒子 8 个角的水平投影都在方框内边以内，盒子静止在桌面上（最低角离桌面 < 1 cm），夹爪没有拿着它。不要求保持原来的面朝下 |

## 任务 → 隐藏信息 → 所需能力 → 调整后的动作
- **任务**：把盒子放进方框。先验做法是抓起来放进去。
- **隐藏信息**：盒子的窄边比夹爪最大开口（两指垫内侧 7.9 cm，实测）宽 4–10 mm。这个差别在 512 px 图像上看不出来，agent 也没被告知夹爪开口多大。
- **所需能力**：从"手指落在盒顶、没夹住"和 F2 里的尺寸对比推断出：这个盒子在任何方向都抓不起来（长边更宽），抓取这条路走不通。
- **调整后的动作**：换成非抓取操作。合上夹爪，降到盒子侧面（指尖离桌面约 1 cm），用 move_to 慢速平移把盒子推进方框。为了不让盒子转，推的方向最好垂直于盒子的一个面、对准面中心；横向偏差再补一次推。agent 没有 push 原语，这些都只能用 close_gripper + move_to 拼出来（自测里 5 条命令就推成功了，见下）。

## 为什么按先验做会失败（实测）
- 夹爪几何（仿真里测）：全开时指垫内侧 ±3.91 cm（开口 7.9 cm），手指外缘 ±6.6 cm；合拢后两指外缘 ±2.73 cm；指尖在 site 下方 0.95 cm；手掌底面在 site 上方 3.0 cm、沿合拢方向 ±10 cm。
- 盒子顶面是平的，手指落在顶面上不会像夹圆柱那样靠斜面摩擦夹住（同一几何下，8.6 cm 粗的圆柱瓶身是能被夹起的，见 `l3_under_table_tool` 卡片里的放弃记录）。
- agent 式无保护抓取（`Skills.grasp_at`：move_to 直接往下压到 site 高 0.5 / 2 / 3 cm，xy 加 σ = 1 cm、yaw 加 σ = 0.1 rad 的扰动，共 120 次，seed 2000–2019 和 3000–3019）：**0 次抬起**；2 次把盒子压翻成侧立；25 次把盒子挤动超过 1 cm（最多 14 cm）。侧立不算作弊：判据只看是否在方框里，侧立后盒子 4 cm 厚、能抓，这也是一种"换操作方式"，只是 agent 很难有意做到。

## F2 反馈（脚本跑出来的原文）
- 第一次直接抓（naive，seed 1000）："The fingers came down on top of the box instead of on either side of it: along the gripper's closing direction the box measures 8.4 cm, wider than the gripper's maximum opening (7.9 cm). The box was not grasped. The box did not move; it sticks out 19.6 cm beyond the square's right edge."
- 换成沿长边抓（seed 1003）："... along the gripper's closing direction the box measures 12.1 cm, wider than the gripper's maximum opening (7.9 cm). The box was not grasped. The box did not move; it sticks out 16.2 cm beyond the square's left edge."
- 推得不够（seed 1003，单次直线推、停在差 6 cm 处）："The box was moved 12.8 cm but is not completely inside the square: it sticks out 3.0 cm beyond the square's left edge; its centre is 0.4 cm backward and 7.3 cm left of the square's centre. It turned by 12 deg (counter-clockwise seen from above)."
- 推的作用线偏离盒子中心 3.5 cm（seed 1003）："The box was moved 15.3 cm but is not completely inside the square: it sticks out 3.7 cm beyond the square's backward edge, 3.5 cm beyond the square's left edge; its centre is 5.3 cm backward and 5.3 cm left of the square's centre. It turned by 33 deg (clockwise seen from above)."
- 成功："The box lies completely inside the square."

反馈只给相对量（离方框边多少、往哪个机器人方向偏、转了多少度），不给绝对坐标，不提"推"。文字由任务里的逐步事件监视器生成（包了一层 `env.step`），所以 harness 里 agent 自己调用 move_to 做的尝试也能得到同样的反馈。"手指落在盒顶"只在夹爪张开（开口 > 3 cm）时记录，合爪推的时候手指碰到盒顶不会被误报成抓取失败。

## 脚本参照策略
- oracle：直接推。闭环：每次观测盒子位姿，沿盒子自身的轴（垂直于一个面）推剩余误差较大的那个分量，最多 4 次，误差 < 0.8 cm 停。推的时候夹爪合拢、yaw 取推的方向（两指并排横在推进方向上，正面宽 5.5 cm），site 高 2 cm，速度 8 cm/s。
- naive：在盒子中心跨窄边抓（yaw 对准盒子），下降碰到顶面就停（接触感知），合爪、抬起；抬起来了才去方框里放。
- adaptive：第 1 次同 naive；读到 `too_wide` 后换成 oracle 的推法；如果推完仍在框外，按反馈里的中心偏差把目标点反向偏一半再推。
- blind：每次都直接抓，加扰动（dx、dy ~ N(0, 0.8 cm)，dyaw ~ N(0, 0.15 rad)），不看历史。

## headroom 验证（`scripts/validate.py --task l3_too_wide_push --n 20 --k 5`，seed 1000–1019，实测）

| 策略 | @1 | @2 | @5 |
|---|---|---|---|
| oracle | 100 | 100 | 100 |
| naive | 0 | 0 | 0 |
| adaptive | 0 | 100 | 100 |
| blind | 0 | 0 | 0 |

- 失败模式统计：naive 20/20、blind 100/100 次尝试都是 `too_wide`；adaptive 第 2 次全部成功。耗时 219 s。
- 额外 oracle 抽查：seed 2000–2029 共 30 个新实例，30/30 成功。
- 第一版有 1/20 oracle 失败（seed 1015）：方框在盒子后面（往机器人方向推），手要伸到盒子远侧 x ≈ 0.09，到了机械臂在桌面高度的够取极限，手停住、盒子没动。现在采样时要求"往后推"的实例盒子远侧面在 x ≤ −0.01，上面的数字是改后的。

## agent 接口自测（端口 9840，2026-10-07）
`start_robot_session.sh l3_too_wide_push 1000 full F2 9840 direct 3 selftest rgb 1`。README 只有通用说明和任务句，没有盒子尺寸、夹爪开口或"推"的提示。第 1 次：张开、yaw 98° 对准窄边、move_to 下压到 z = 0.92、合爪、抬起、done，得到上面 naive 那段 F2；下压时手在盒顶上滑了 1.5 cm。第 2 次：close_gripper → move_to 到盒子右侧 (−0.025, −0.13, 1.0) yaw 90 → 降到 z = 0.92 → 以 0.08 m/s 平移到 y = 0.125 → 抬起 → done，成功。全程无报错，会话和目录已删除。

## 已知问题 / 效度威胁
实测：
- 推的结果对作用线很敏感：偏离盒子中心 3.5 cm 推 15 cm，盒子转 33°（上面的 F2 例子）。agent 用 RGB 估计盒子中心有误差，可能要多推几次，40 条命令/次的预算够用（自测一次推成用了 6 条）。
- 方框 18 cm 对 11–13 cm 的盒子比较宽松，主要考的是"换方式"，不是推的精度。

推测（未验证）：
- 盒子宽于开口 4–10 mm，强模型如果知道 Panda 开口约 8 cm、又能从图上估出盒子约 8.5 cm，可能第 1 次就直接推。和 l3_wall_flush 一样，建议把 attempt-1 成功率和 succ@k、ΔICL 分开报告。
- F2 直接给出"盒子沿合拢方向 8.4 cm、比最大开口 7.9 cm 宽"，信息量很足；F1 条件下 agent 只能从图像（手指停在盒顶）自己看出原因，难度会高很多，值得单独比较。
