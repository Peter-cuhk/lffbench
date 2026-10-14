# L3 `l3_card_shelf`：搁板上一头藏配重的薄卡片，推错一端就翻落到桌上

代码：`lffbench/tasks/l3_card_shelf.py`　结果：`results/l3_card_shelf/validate.json`　图：`results/l3_card_shelf/visibility_naive_top_oracle_bottom_s3000_3001_3003.png`（上排 naive 结束时，下排 oracle 结束时）

> 2026-10-12 新增，是饱和的 `l3_card_edge` 的加深版（设计见 `research_notes/P1P2_加深与常识任务设计.md` §1、§3.2）。`l3_card_edge` 保留不动，方便复现。
> 饱和原因：Sonnet 5.5（cla4）空夹一次后，在同一次尝试里把卡片推出板边 2–3 cm 再夹，H、N 两个 run 都是第 1 次就成功。空夹只花 1 条命令，场景也不变。
> 加深思路：让第一次按常识的交互（把最近的一端推出约 3 cm）在重端实例上付出不可逆的代价。卡片翻落到桌上就夹不起来，这次尝试作废。

| 项 | 内容 |
|---|---|
| 类别 | strategy_switching（L3），`level="L3"` |
| 能力 | Perceive（看见卡片翻落、推出多少）、Reason（才推出约 2 cm 就翻 → 那一端重）、Plan（换推另一端，推得更远） |
| 协议 | cross，k = 3（`max_attempts = 3`），`agent_max_steps = 40` |
| 指令 | "Pick up the thin red card lying on the wooden shelf and put it into the grey bin."（只把 board 改成 shelf，不提配重）；indirect："Put the flat red rectangle that lies on the raised wooden platform into the grey container." |
| 场景 | 一张木搁板（fixture）：顶板 22 × 26 × 1 cm，4 条 2 × 2 cm 的腿缩在顶板四角内，顶面离桌 10 cm，μ 0.3。搁板中心 x ∈ [-0.14, -0.06]、y ∈ [-0.08, 0.08]。板上平放一张红色薄卡片：9.0 × 5.6 × 0.2 cm，10 g，μ 0.3，尺寸、质量、摩擦都和 l3_card_edge 相同。卡片长边沿 x 或 y（±8°）。近端离它所指的板边 2.5–4 cm，远端离对边 9–14 cm，两条长边离侧边都 ≥ 5 cm。沿 x 放的卡片，近端一律是 forward（远离机器人）那端，原因见"已知问题"。灰色盒子（内 16 × 16 cm、高 5 cm，fixture）放在搁板左边或右边的桌上，与板边隔 6 cm |
| 成功 | 卡片在盒子里：中心在盒内、低于盒沿、没被夹着 |

## 任务 → 隐藏信息 → 所需能力 → 调整后的动作
- **任务**：把卡片放进盒子。
- **隐藏信息**：卡片 10 g，其中 8 g 是藏在一端的配重，配重中心离卡片中点 3.5 cm。重心因此偏向重端 COM_E = 2.8 cm。两端外观完全一样。每 20 个连续 seed 里有 14 个的重端是**近端**（`(seed*7) % 20 < 14`），其余 6 个的重端是远端。
- **先验为什么失败**：先空夹一次（2 mm 卡片平放时夹不起来，同 v1），再把最近的一端分几下推出板边约 3 cm（每推一下就抬手看一眼），然后夹悬空段。如果推的是重端，推出约 1.8 cm 时重心就越过板边。手指一抬，卡片就翻下搁板，0.4 s 内平躺在桌上。桌上的平卡片夹不起来，也够不到任何桌边，这次尝试作废。翻落在 agentview 里很明显：搁板上空了，卡片躺在搁板前方或侧面的桌上。
- **要学到的东西**：才推出约 2 cm 就翻了，说明那一端重（"推太远了"是更便宜但不对的解释）。下次推另一端：轻端推出 1.8–3.8 cm 都能夹起；远端离板边 9–14 cm，要滑得更远。另一个办法是只把重端推出 1.3–1.5 cm、紧贴板边夹，但窗口很窄（实测 12/36）。

## 物理标定（实测，2026-10-12）
**翻落阈值**（卡片直接摆成悬出 h，静置 2 s；seed 3000 的布局，卡片沿 x）：

| 重心偏移 | 重端朝外：最大不翻 / 最小会翻 | 说明 |
|---|---|---|
| 2.5 cm | 2.0 / 2.1 cm | |
| 2.67 cm | 1.8 / 1.9 cm | 20 g 配重 + 10 g 卡片时，配重放到最端头能达到的上限 |
| **2.8 cm（采用）** | **1.7 / 1.8 cm** | |
| 3.0 cm | 1.5 / 1.6 cm | |
| 轻端朝外（2.8 cm） | 7.0 cm 不翻，7.5 cm 翻 | |
| 均匀卡片 | 约 4.0–4.5 cm（这一段结果不单调） | |

**其他物理实测：**

| 测试 | 结果 |
|---|---|
| 平放直接夹，10 g 卡片（5 个 seed × 4 次，xy 抖动 4 mm、高度 −6…+3 mm） | 0/20 提起，指尖都从卡片上滑过，合拢后宽度 0.7–0.85 cm |
| 平放直接夹，30 g 卡片（惯量被 `_fmt` 舍入，见"已知问题"） | 4/16 被捏住提起（宽度 5.2 cm）→ 改回 10 g |
| Sonnet 式推重端（validate 前 40 个 seed 里 28 个重端在近端的实例，按压拖动和推端头两种推法各 1 次） | 56/56 翻落；全部平躺在桌上（倾角 0°）；落地后原地再夹 0/56 |
| 翻落过程（5 个实例逐帧记录） | 从开始翻到平躺在桌上 ≤ 0.4 s；中途 agent 可能拍到卡片挂在板边倾斜约 45° 的一帧 |
| 重端朝向盒子那一侧（y 卡片，近端对着盒子） | 落在搁板和盒子之间 6 cm 宽的空隙里，平躺，12/12；没有落进盒子，也没有斜靠在盒壁上 |
| oracle ±5 mm（抓取点 xy ±5 mm、高度 ±5 mm、推出量 ±5 mm、按压点 ±3 mm；12 个 seed × 2 次） | 24/24 成功 |
| 轻端推出量窗口（4 个 seed） | 推出 1.8–3.8 cm：11/12 成功；4.4–4.7 cm：1/4 成功 |
| 重端"少推、贴边夹"（4 个重端在近端的 seed，推出 1.0–1.5 cm，抓取点在板边外 0.8 / 1.0 / 1.2 cm） | 12/36 成功，只在推出 1.3–1.5 cm 时成功；其中 1 个 seed 0/9 |
| 沿短边推（4 个 seed，悬出 1.0–2.65 cm；3 种夹法：手指沿长边、照常跨短边、45° 夹角） | 只推不夹时 0/16 翻落；夹 0/48 提起；有 18/48 次夹的时候把卡片碰下了板 |
| 手和腿的碰撞（validate 全部 122 次尝试 + 鲁棒性 24 次 + 均匀卡片对照 20 次） | 机器人和腿接触 0 次 |

**对照（实测，seed 1000–1019，单次尝试）：**
- 同一个 Sonnet 式 naive 放到 **v1 `l3_card_edge`**（均匀卡片、实心板）：**20/20 成功**。
- 同一个 naive 放到本任务，但卡片**均匀**（COM ≈ 0）：**20/20 成功**。
- 结论：失败只来自隐藏的配重，搁板本身不改变难度。

## headroom 验证（n = 20，k = 3，实测，`python3 scripts/validate.py --task l3_card_shelf --n 20 --k 3`）
20 个实例：
- 卡片沿 y 11 个、沿 x 9 个；
- 近端 forward 9 个、right 6 个、left 5 个；其中 5 个 y 卡片的近端正对着盒子；
- 重端在近端 14 个，在远端 6 个。

| 策略 | @1 | @2 | @3 | 说明 |
|---|---|---|---|---|
| oracle（知道哪端轻：把轻端分两段推出 3 cm，在板边外 2.2 cm 处夹，放进盒子） | 100% | 100% | 100% | 用到的板边：backward 8、right 8、left 3、forward 1 |
| naive（Sonnet 式：空夹 → 最近的一端推到板边，再每次推 1 cm 到约 3 cm，每推一下抬手看一眼 → 夹） | 30% | 30% | 30% | 第 1 次尝试翻落 14/20；重端在近端 0/14，轻端在近端 6/6 |
| adaptive（第 1 次 = naive；翻落 → 下一次推另一端；推出了但没夹住 → 多推 1 cm） | 30% | 100% | 100% | 第 2 次尝试 14/14 成功 |
| blind（naive 加噪声：推出 2.5–3.5 cm，按压点 ±5 mm，抓取点 ±5 mm，不看历史） | 30% | 30% | 30% | 重端实例 42/42 次尝试全部翻落，翻落前推出了 1.9–3.2 cm |

- **重端在近端 vs 轻端在近端：** naive@1 为 0/14 vs 6/6；陷阱触发率 14/14 vs 0/6。
- naive 记录到的翻落前推出量是 2.7–2.8 cm：按压拖动每推一下会多走 0.5–0.8 cm，卡片被手指压着还没翻，手指一抬才翻。
- 带 agent 接口的自测（见下）里，按 1 cm 一步推：0.1 → 0.9 → 1.7 cm 都没翻，下一步就翻了。

## 事件与 outcome（只进服务器日志，agent 只收到 F1）
- `events`：每条带 sim step，类型如下：
  - `end_over_edge`（part = heavy / light，某一端第一次推出 ≥ 0.5 cm）；
  - `side_over_edge`（长边推出 ≥ 0.5 cm）；
  - `tipped_off`（从没被提起就掉下搁板；记录是哪部分、哪条边、翻前推出多少）；
  - `lifted`；`released`；`in_bin`；
  - `hand_hit_leg`（手或臂碰到腿）。
- R6 字段：`tipped_off`、`trap_fired`（翻落的是重端）、`grasp_tries`，以及 `end_out`（重端 / 轻端 / 长边各自的最大推出量）、`slid_edge`、`leg_hits`、`card_final`。
- `mode` / `failure`：`success`、`fell_off`、`not_lifted`、`dropped`、`held`。
- F2 文字示例（seed 1000，naive）：
  > The card was never lifted: after one end of it had been slid 1.9 cm beyond the shelf's left edge, it tipped over that edge and fell off the shelf; it lies flat on the table. When the gripper closed on the card lying flat on the shelf, its fingertips were on either side of the card, at shelf level (-0.2 cm from the fingertips to the top of the card); the fingers closed fully, sliding over the top of the card instead of gripping its edges.

## 失败看得见吗（实测，自己看过图）
- **x 卡片向 forward 翻**：卡片平躺在搁板正前方的桌上，agentview 里是一大块红色，很显眼（seed 3000、3001、1003）。
- **y 卡片向侧面翻**：卡片落在搁板左侧或右侧的桌上，有一部分被近处的腿挡住，但红色清楚可见（seed 1000、1001）。近端对着盒子的，卡片落在盒子和搁板之间的空隙里，同样可见（seed 3003）。
- **两种情况的共同点**：搁板顶面空了。
- **oracle 结束时**：卡片在盒子里。

## 选定的 pilot seed（实测）
seed ≥ 3000 中前 3 个"naive 失败且 oracle 成功"的：**3000**（x 卡片，近端 forward，盒子在右）、**3001**（x 卡片，近端 forward，盒子在左）、**3003**（y 卡片，近端 right，盒子在右，即近端对着盒子）。seed 3002 是轻端在近端，naive 成功。

## agent 接口自测（2026-10-12）
- `claude_batch.py start --batch smoke_l3_card_shelf --only harness --k 3 --perception rgb --style la --feedback F1 --port0 12620 --tasks l3_card_shelf:3000` 正常起服务，README 正确：指令、3 次尝试、每次 40 条命令、120 s 仿真运动。之后已 `stop`。
- 用 `./robot` 发命令这一步被本机的权限分类器拦下，没有执行。改为在进程内直接调用同一个 `RobotSession.handle()`：参数相同，la / cli / rgb / F1，seed 3000，只是不经过 HTTP。
  - 第 1 次：`status` → 空夹 → 按压往 forward 推，每次约 1 cm 后抬手。推出 0.1 → 0.9 → 1.7 cm 都没翻，下一下就翻了。`finish` → "Attempt failed."；日志里的 outcome 是 `fell_off`，事件 `tipped_off(heavy, forward)`。
  - 第 2 次：推 backward（轻端）两下，推出 2.6 cm → 在板边外 2.2 cm 处夹（宽度 0.056）→ 放进盒子 → "Attempt succeeded."。
  - 第 1 次用了约 21 条命令，第 2 次 16 条，都在 40 条的预算内。

## 已知问题 / 推测（未验证的标【推断】）
- **Sonnet 推哪一端不确定。** cla4 里 H 推的是更远的 forward 边（7.8 cm 对 5.8 cm），N 推的是更近的一端。所以本任务把 x 卡片的近端统一设成 forward：最近的一端同时也是远离机器人的一端。y 卡片的近端是 2.5–4 cm 对 13–14 cm，差距很大。【推断】如果 agent 随机挑一端推，第 1 次翻落率约 50%，不是 70%。
- **为什么 x 卡片不让近端朝 backward**（实测 seed 1003 的旧版布局）：卡片从 backward 边翻落后，落在搁板后面的桌上，被 10 cm 高的顶板挡住，agentview 看不见，只能看出"搁板上空了"。现在只有一种情况还会这样：重端在远端的 x 卡片（约 15% 的实例），agent 主动去推远端。
- **重端"少推、贴边夹"的窄窗口。** 推出 1.3–1.5 cm，紧贴板边外 0.8–1.2 cm 夹，实测 12/36 成功。自测里推出 1.7 cm 也没翻。【推断】小步推、每步都看的 agent 有可能停在 1.5 cm 左右然后夹成功，这一部分第 1 次成功不算"学到了"。
- **沿短边推是死路**：夹 0/48，与配重无关。v1 也一样，不是新问题。只要卡片还在板上就能恢复：推回来再推一端。
- **翻落中途的那一帧。** 如果抬手命令刚好在翻落开始时结束，图上卡片是挂在板边、倾斜约 45° 的样子。下一条有运动的命令执行完时，卡片已经平躺在桌上了（≤ 0.4 s）。【推断】agent 来不及在半空接住它。
- **卡片质量必须是 10 g，惯量要用科学计数法写。** `objects._fmt` 只保留 5 位小数，会把 10–30 g 卡片的惯量舍成约 0：10 g 时 XML 直接编译失败；30 g 时舍入后的惯量让平放直接夹 4/16 次把卡片捏起来。代码里的惯量用 `%.6e` 写出。【推断】其他手写小惯量 inertial 的任务也可能有同样问题；建议共享代码里提供一个不舍入的格式化函数（我没改 `objects.py`）。
- **重心在 XML 里写死**（`<inertial pos="0.028 0 0">`），哪端重由卡片的 yaw 决定，运行时不改 `body_ipos`，所以不需要 `refit_bounds`。已确认 `body_sameframe = 3`，子树重心在重端一侧 2.8 cm 处。
- **【推断】Sonnet 的实际第 1 次失败率还没测**（本次不跑 LLM 评测）。设计备忘录的估计是 50–70%。
