# L3-A `l3_wall_flush`：贴墙的扁长盒子放到盘子上

| 项 | 内容 |
|---|---|
| 等级 | L3 更换操作方式（默认抓法在这个状态下物理上做不到，要先换成非抓取的预操作） |
| 能力 | Perceive（看出手指落在墙顶，不是盒子上）、Reason（为什么抓不起：贴墙、横跨长边又张不开）、Plan（先沿墙推出墙端，再横跨宽度抓） |
| 协议 | `l3_wall_flush`：cross，每次尝试重置到同一实例，k = 5。`l3_wall_flush_within`：同一 episode 原地继续，每次尝试从上一次留下的状态开始，k = 5 |
| 指令 | 直接："Put the long box on the plate." 间接："Put the flat blue block onto the round dish." |
| 场景 | 蓝色扁盒 16 × 6 × 3.5 cm，0.50 kg，长边沿 y（从机器人看是左右方向）。木色矮墙是固定 fixture：厚 3 cm、高 4.5 cm、长 24–30 cm，盒子一条长边贴着它。LIBERO 原生 `plate`（直径约 13.7 cm）放在盒子不靠墙的那一侧，距盒子 16–18 cm |
| 随机 / 隐变量 | 墙在盒子远侧（+x）还是近侧（−x），各约一半；墙长；墙的 y 位置；盒子沿墙的位置（整条盒子都在墙边，两端各至少多出 2 cm 墙）；盒–墙间隙 0–3 mm（相机分辨不出）；盘子位置 |
| 成功判据 | 盒子中心离盘子中心水平距离 < 5 cm，盒底高于桌面 4 mm 以上，盒子平放（\|cos\| > 0.9），和盘子有接触，且夹爪已经松开 |

## 为什么按先验做会失败
Panda 两指全开时指垫内侧相距 7.9 cm（实测）。盒子只有 6 cm 的宽度能横跨着抓，16 cm 的长边张不开。横跨宽度抓，必须有一根手指下到盒子和墙之间；手指加指垫需要约 1.2 cm 空隙，而这里间隙只有 0–3 mm。所以靠墙那根手指会落在墙顶，指尖停在桌面上方 4.5 cm。指垫从指尖往上 0.5 cm 才开始，于是整个指垫都在盒顶（3.5 cm）以上，夹爪合上只夹到空气。原样重试、加小抖动，结果都一样。

## 要从失败里学到什么
这个状态下这种抓法不可行，要换操作方式：先用合拢的夹爪顶住盒子靠墙端的端面，把盒子沿墙推出离它最近的墙端（让盒子中心越过墙端约 4.5 cm），然后再横跨宽度抓起来，放到盘子上。推的时候手掌底面比墙顶高约 7 mm，所以可以贴着墙推。

## F2 反馈示例（脚本跑出来的原文）
- 第一次直接抓（naive）："The gripper could not get around the box: the finger on the wall side (far (+x) side) came down on top of the wall, 4.5 cm above the table, which is higher than the top of the box (3.5 cm), so the box was not grasped or lifted. Where the gripper came down, the box is flush against the wall (gap 2 mm)."
- 推得不够远："The box was pushed 4.9 cm along the wall (towards -y). The gripper could not get around the box: … The wall still continues 5.7 cm beyond the grasp point in the -y direction."
- 转 90° 横跨长边抓："The fingers came down on top of the box: along the gripper's closing direction the box measures 16 cm, more than the 7.9 cm maximum opening. The box was not lifted."
- 抓起来了但没放好："The box was lifted but ended on the table, 9.3 cm from the plate centre." / "The box is still in the gripper; it was not released on the plate."

反馈文字由任务里的逐步事件监视器生成：它包了一层 `env.step`，记录手指落在哪个顶面上（墙 / 盒子），以及落下那一刻的间隙、夹爪离墙端多远、盒子有没有被举起、沿墙被推了多远。所以 harness 里由 agent 自己调用工具做的尝试，也能拿到同样的 F2，不依赖脚本的 `execute`。`outcome` 还返回 `terminal`（盒子掉下桌），within 协议用它判断能否继续。

## 脚本参照策略
- oracle：直接用两步策略。先 `push_along_wall` 推向较近的墙端，余量 4.5 cm；再 `grasp_and_place`。
- naive：在盒子原位横跨宽度抓（夹爪 yaw 对准盒子短轴），下降时碰到顶面就停。
- adaptive：第 1 次同 naive。读到 `wall_blocked`（或 `box_top`）后换成两步策略，推向较近的墙端；墙端位置从图像上能读出来，脚本用 `observe()` 代替感知。如果推完仍是 `wall_blocked`，就按反馈里"墙还剩多长"加 2 cm 再推。
- blind：每次都直接抓，加抖动（dx、dy ~ N(0, 0.8 cm)，dyaw ~ N(0, 0.12 rad)），不看历史。

## headroom 验证（`scripts/validate.py --n 20 --k 5`，seed 1000–1019，实测）
（数字见下方"最终验收"一节）

## 物理可行性和标定（实测，2026-10-06）
- 夹爪几何（仿真里测的）：全开时指垫内侧 ±3.9 cm，手指外缘 ±6.6 cm，指尖在夹爪 site 下方 0.96 cm。手掌底面在 site 上方 3.0 cm，沿合拢方向伸出 ±10 cm。
- 墙高的取值依据：墙必须比盒子高，这样手指停在墙顶时指垫在盒顶以上（4.5 vs 3.5 cm，留 1.5 cm 余量）；又必须比贴地推时的手掌底面低（site 2.2 cm + 3.0 cm = 5.2 cm），这样才能贴着墙推。厚度 3 cm 是为了让张开的那根手指（离中心 3.9–6.6 cm）整根落在墙顶的平面上。
- 最早试过墙沿 x 方向（朝向远离机器人）。推到 +x 端以后盒子在 x≈0.10，到了臂展边缘：1/2 个 oracle 因为够不到而搬运失败。所以改成墙沿 y，所有抓取点 x ∈ [−0.20, −0.04]，盘子 x ≤ 0.06。
- 下降方式：如果不带接触停止，直接 `move_to` 压向墙顶，OSC（kp 300）的柔顺会让夹爪侧滑 2–7 cm（8/8 个 seed 都有）。所以脚本抓取用"碰到顶面就停"的下降（`_guarded_descent`）。
- agent 式的无保护下降（共享 `Skills.grasp_at`，等价于 harness 里 move_to 往下压再合爪）：墙面摩擦 1.0、盒子 0.13 kg 的初版中，60 个实例里 0 个举起盒子，但有 10 个把盒子掀成侧立（10 个都是墙在远侧的实例）。侧立后盒子只有 3.5 cm 厚、离墙 4.8 cm，是一个意外的、能抓的状态，会让瞎试也成功。把墙面摩擦提到 2.0、盒子加重到 0.5 kg 后，这 8 个易翻 seed 0/8 翻倒；60 个 seed 的结果见下节。
- 扭腕问题：当前姿态是 yaw −π/2 时，再要 +π/2（转 180°），OSC 会把手腕拧到关节限位，之后永远到不了目标。within 协议下第一次跑 adaptive 时遇到过 2/20（seed 1007、1009）。现在任务里 yaw 一律取 mod π 等价角里离当前 yaw 最近的那个；共享 `Skills.move_to` 后来也加了同样的处理（coordinator 改的）。共享 `Skills.push` 往 −y 推时 yaw = −π，修之前推完再抓 0/30 成功，修之后的数字见下节。

## 已知问题 / 效度威胁
实测：
- naive 失败在 agentview 里不算显眼：画面上只是夹爪悬在盒子上方、盒子没动。在腕部相机和 sideview 里看得很清楚，能看到手指压在墙顶、盒子紧贴着墙（`results/l3_wall_flush/naive_contact_*.png`，三联图是 agentview / 腕部 / sideview）。F0 条件下 agent 需要看腕部图。
- 成功判据要求盒子平放。盒子侧立放在盘子上不算成功；指令里没写"平放"。
- 盘子直径 13.7 cm，盒子 16 cm，成功时盒子两端各伸出盘沿约 1 cm。

推测（未验证）：
- 陷阱在初始图像里原则上能预见：盒子看得出贴着墙。看不出来的只是那 0–3 mm 的间隙，而这对结果没有影响，任何小于 1.2 cm 的间隙都抓不进去。所以强模型可能第 1 次就先推再抓，这时第 1 次成功率测的是预见能力，不是从失败里学习。建议第 1 次成功率和 succ@k、ΔICL 分开报告。这一点还没用真实 VLM 测过。
- adaptive 用 `observe()` 读到的墙端位置是精确值，真实 agent 要从图像估计。估计误差在 4.5 cm 余量以内应该没事；超出时，"推得不够远"的反馈会给出墙还剩多长。这个闭环脚本测过（2/2），没用 agent 测过。
- 其他可行的做法没测：按住盒顶把它从墙边拖开、先推一部分再抓伸出墙端的那一截、或者故意把盒子立起来再抓。最后一种会被"平放"判据判为失败（除非再放平）。
- harness 的 push 工具默认高度是 TABLE_Z + 0.02，这时手掌底面离墙顶约 5 mm，余量很小。如果 agent 给的推高度更高，手掌就会刮到墙顶。
