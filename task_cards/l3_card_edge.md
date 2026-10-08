# L3 `l3_card_edge`：薄卡片平放夹不起来，先滑到板边悬空再夹

代码：`lffbench/tasks/l3_card_edge.py`　结果：`results/l3_card_edge/validate.json`　视频：`videos/l3_card_edge_adaptive_s1000.mp4`（第 1 次直接夹失败，第 2 次滑到板边再夹，放进盒子）

> 2026-10-08 新增。用户的失败分级 L3 原例："卡片贴着桌面捏不起来 → 先把卡片滑到桌沿，再夹起来"（失败类型：要换夹取的位置）。之前的 `l3_flat_card_edge` 因为卡片能直接夹起而放弃（见 `l3_under_table_tool.md` 文末）；这次查清了原因，换了卡片厚度后成立。

| 项 | 内容 |
|---|---|
| 类别 | strategy_switching（L3，"发现这种做法根本行不通"），`level="L3"` |
| 能力 | Perceive（看卡片有没有被夹起、悬出多少）、Reason（平放时根本夹不住 → 换做法）、Plan（压住滑到板边 → 夹悬空部分 → 放进盒子） |
| 协议 | cross，k = 5 |
| 场景 | 桌上一块木板（22 × 26 × 5 cm，fixture，μ 0.3），板上平放一张红色薄卡片（9.0 × 5.6 × 0.2 cm，10 g，μ 0.3），长边大致沿 x 或沿 y（±10°），离板边至少 3 cm（长边方向）/ 2 cm（短边方向）。板旁桌上一个灰色盒子（内 16 × 16 cm、高 5 cm，fixture），在板的左边或右边，和板边隔 6 cm。木板中心 x ∈ [-0.14, -0.06]、y ∈ [-0.08, 0.08] |
| 成功 | 卡片在盒子里（中心在盒内、低于盒沿、没被夹着） |
| 为什么用木板代替桌沿 | 真桌沿够不到：近边离机器人基座 16 cm，侧边在 \|y\| = 0.6 m，超出工作空间（\|y\| ≤ 0.45）；而且 harness 不允许抓取点低于桌面 +5 mm，手指伸不到桌面以下。5 cm 高的木板起的就是"桌沿"的作用 |

## 任务 → 隐藏信息 → 所需能力 → 调整后的动作
- **任务**：把卡片放进盒子。
- **隐藏信息 / 结构陷阱**：卡片太薄（2 mm），平放时两指尖只能贴着板面碰到它 2 mm 的边。从图像上看得出卡片薄，但看不出"薄到夹不住"。
- **先验为什么失败**：从上往下、跨短边夹卡片中间。指尖落在卡片两侧的板面上，合拢时一接触到 2 mm 的边，接触法向就翻成竖直，两指尖骑上卡片、在卡片上方合拢（实测见下）。原样小幅重试结果一样。
- **要从失败里学到 / 调整后的动作**：卡片平放时根本夹不起来 → 换做法：先用合拢的指尖压住卡片滑动（或贴着板面推卡片的端头），让它的一端悬出板边约 3 cm（2.5–4 cm 都行，超过一半长度会翻下去）；再跨短边夹悬空的那一段，指尖放到卡片以下，手指就夹在卡片完整的边上，能提起来。

## F2 反馈（只给相对测量，机器人视角方向词）
直接夹（seed 1000，脚本先验；agent 接口自测里是同一段文字）：
> Attempt failed. The card was never lifted; it still lies flat on the board. When the gripper closed on the card lying flat on the board, its fingertips were on either side of the card, at board level (-0.2 cm from the fingertips to the top of the card); the fingers closed fully, sliding over the top of the card instead of gripping its edges.

滑了但悬出太少、夹的位置还在板上方（开发时 seed 1000 的一次失败）：
> Attempt failed. The card was never lifted; it still lies flat on the board (it was slid 10.6 cm and stuck out up to 2.8 cm beyond the board's edge). When the gripper closed, the card stuck out 2.8 cm beyond the board's right edge and the gripper's centre was 1.4 cm beyond the board's right edge (-0.4 cm from the fingertips to the top of the card); the fingers closed fully, sliding over the top of the card instead of gripping its edges.

成功：`Attempt succeeded. The card is in the bin.`
其他失败文字：卡片掉到桌上（"it fell off the board and lies flat on the table"）、提起来了但没进盒子（"The card was lifted X cm above the board but is not in the bin: it lies on the table, its centre …"）、仍夹在手里、从没在卡片处合过夹爪。
"夹取尝试"只统计指尖离卡片顶面 ≤ 2 cm、合拢前张开 > 2 cm 的那几次合爪（压卡片前在空中合爪不算）；反馈描述最后一次。

## 物理标定（原型 + 任务内实测，2026-10-08）
**为什么 2 mm 夹不起、4 mm 能夹起**：Panda 手指碰撞外壳的内侧面是平的，一直延伸到指尖（finger.stl 凸包 18 个顶点，内侧面 y ≈ 0 从指尖到根部），指垫从指尖往上 4.85 mm 才开始。所以夹平放的卡片时只有外壳的指尖棱接触卡片边。MuJoCo 的凸体碰撞取最小穿透方向：指尖压到卡片边上时，水平穿透一超过卡片厚度，法向就从水平翻成竖直（调试时逐步打印接触：第 3 步法向 [-1, 0, 0.02]，第 6 步变成 [0, 0, 1]、穿透 1.2 mm），手指骑上去。卡片 4 mm 时竖直重叠够大，不翻，所以能夹起——这和旧设计的结论一致。

| 测试 | 结果 |
|---|---|
| 平放直接夹：厚 1.5 / 2 / 3 mm，抓取点高度 −6…+3 mm，xy 抖动 5 mm，yaw 抖动 0.1 rad | 0/12、0/12、0/12 提起 |
| 悬出板边夹（跨短边，指尖在卡片下 0–1 cm）：悬出 2.5 / 3.5 cm × 3 种高度 × 3 种厚度 | 18/18 提起 11.5 cm |
| 悬出 4.5 cm（> 原型卡片半长 4.3 cm，重心已在板边外） | 0/9 提起（推测是夹之前卡片已翻下板，没有逐帧确认） |
| 压住滑动：压深 2 / 4 / 8 mm × 速度 0.05 / 0.15 m/s，目标滑 12.5 cm | 6/6 到位（误差 +0.1…+0.7 cm），随后夹悬空段 6/6 成功 |
| 贴板面推卡片端头（合拢的手指，指尖在板面 / 压下 2 mm），扫 10 cm | 16/16 推动 7.3–8.7 cm |
| 同上，指尖离板面 3 mm | 7/8 完全没动（指尖从卡片上方掠过），1/8 推动 2.0 cm |

- 单次滑动会多走 0.1–0.7 cm；朝机器人一侧（backward）的板边滑时更多，seed 1003 一次多走到悬出 4.2 cm，夹的时候卡片翻下了板。所以脚本 oracle 分两段滑（先滑到差 1 cm，再看一眼滑完），目标悬出 3 cm，夹在板边外 2.2 cm 处（手指要整个越过板边：卡片斜 10° 时手指的角会碰到板边）。
- 卡片和板的摩擦 0.3，手指外壳 1.0、指垫 2.0（MuJoCo 取两者较大值），所以压住滑动时卡片跟着手指走。

## headroom 验证（n = 20，k = 5，实测，`python scripts/validate.py --task l3_card_edge --n 20 --k 5`）
20 个实例：卡片长边沿 y 11 个、沿 x 9 个；oracle 用的板边 backward 9、right 6、left 5，其中 9 个正好是盒子那一侧的板边（9/9 成功）。

| 策略 | @1 | @2 | @3 | @5 |
|---|---|---|---|---|
| oracle（沿长边分两段滑到悬出 3 cm，夹板边外 2.2 cm 处，放进盒子） | 100% | | | 100% |
| naive（从上往下跨短边夹卡片中心） | 0% | | | 0% |
| adaptive（"平放时没提起来" → 滑到板边再夹；掉下板 → 少悬 1 cm；悬出 < 4 cm 仍夹不住 → 多悬 1 cm） | 0% | 100% | 100% | 100% |
| blind（直接夹，xy ± 6 mm、yaw ± 0.12 rad、高度 −6…+3 mm 随机，无历史） | 0% | 0% | 0% | 0% |

失败模式：naive 20/20、blind 100/100 都是 `not_lifted`（卡片还平放在板上），blind 100 次里卡片最高只离板 0.16 cm。

## agent 接口自测（la 风格，rgb，F2，k=3，seed 1000，2026-10-08）
README 只有指令和通用说明。第 1 次：`move_eef` 到卡片上方 → 下到卡片高度并合爪 → 抬起 → 上面那段"sliding over the top of the card"的 F2。第 2 次只用 `move_eef`（8 条）：合爪悬在卡片上方 → 压下（指尖低于卡片顶面 4 mm）→ 以 0.08 m/s 平移到让卡片悬出右边板边 3 cm → 抬起并张开 → 移到悬空段上方 → 下到板面以下 3 mm 并合爪 → 抬起 → 移到盒子上方并张开 → "Attempt succeeded. The card is in the bin."

## 已知问题 / 推测（未验证的标【推测】）
- "夹不起"来自仿真接触模型（法向翻转让指尖骑上卡片），不是摩擦或夹力不足。现实中薄卡片夹不起的原因相近（指尖够不到卡片的边、插不到卡片下面），但厚度阈值未必相同。仿真里 4 mm 能夹、2 mm 不能，所以卡片做成 2 mm。
- 第 1 次失败的 F2 写了 "the fingers closed fully, sliding over the top of the card instead of gripping its edges"，是对接触结果的描述，没说"该滑到板边"。【推测】强模型看到"薄卡片 + 木板"可能第 1 次就想到滑到边上，attempt-1 成功率要和 succ@k、ΔICL 分开报。
- 合爪前张开 ≤ 2 cm 的合爪不算夹取尝试；如果 agent 先半合再去夹，F2 里会说 "never closed with its fingertips down at the card"。【推测】少见。
- 只有木板的四条边可用；盒子那一侧的板边外 6 cm 才是盒子外壁，手指放得下（验证集里 oracle 有 9 个实例用的正是盒子那一侧的板边，9/9 成功）。
