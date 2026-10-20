# LH `lh_shelf_droop`（v2）：把四个罐子放上墙边搁板——支架的卡扣是松的，第三个罐子放在搁板中间后搁板翻折下去，前面放好的罐子滑到桌上

代码：`lffbench/tasks/lh_shelf_droop.py`　结果：`results/lh_shelf_droop/validate.json`、`controls.json`、`pilot_seeds.json`、`risk_sweep.json`、缩略图、冒烟测试截图、演示视频 `demo_adaptive_s3000.mp4`　v1 代码（未进仓库）：`/tmp/claude-0/-mnt-cpfs/2aa0daff-4c37-4ec7-b3e7-10b838c28688/scratchpad/lh_shelf_droop/lh_shelf_droop_v1.py`　原型与中间数据：同一 scratchpad 目录（临时；`risk_sweep.py`、`rsum.py`、`vsum.py`、`p3_run.py`、`pilot.py`、`smoke/drive.py`）

> 2026-10-16 新增，同日从 v1 改成 v2（v1 的失败有 20% 在 `finish` 之后才发生，agent 看不到；见"物理修正"）。设计意图（总管）：长程任务，自损要过一会儿才发生。上线标准同 lh_nudge_row / lh_bystander：模仿 Sonnet 的 naive 碰坏约 65–75%、adaptive 全部修好、失败在画面里看得出来。v2 为了"看得见"，把"放下后到罐子滑落"的延迟从 v1 的 3–5 s 缩短到约 1–3 s（中位 1.25 s 验证 / 1.55 s 扫描）。

| 项 | 内容 |
|---|---|
| 类别 | state_restoration，`level="L2"` |
| 能力 | Perceive（看出搁板翻下去了、罐子躺在桌上）、Reason（不是"放得不稳/松手太高/手臂碰的"，而是离墙远的载荷压翻了支架）、Restore（尝试内按把手提起躺倒的罐子、扶正、放回靠墙处）、Plan（下一次一开始就靠墙放） |
| 协议 | cross，k = 3，`agent_max_steps = 120`（LA 预算 120 条命令 / 360 s 仿真运动） |
| 指令（直接版） | "Put the four jars (red, green, blue, yellow) on the shelf. At the end all four must be standing upright on the shelf board, and the gripper must be open and clear of the jars and the shelf. (Each jar has a round knob on its lid; held by the knob, a jar hangs straight down.)" |
| 指令（间接版） | "Put the four little square jars with knobbed lids up on the wooden shelf by the wall. When you are done, all of them have to stand upright on the shelf board, and the gripper must be open and away from the jars and the shelf. (Held by its round knob, a jar hangs straight down.)" |
| 成功（终态判定） | 机械臂静止 ≥ 15 s 仿真时间后（搁板或罐子还在动就延长到最多 30 s）：四个罐子都立在搁板上（罐底在板面上、在板的轮廓内、轴线与板面法线夹角 ≤ 10°、没被夹着），搁板下垂 ≤ 15°，搁板只被罐子碰着（不许垫东西撑板），夹爪不碰罐子和搁板，场景静止 |

## 场景（代码里的数，实测）
- **墙板**：沿 x 方向，厚 2.5 cm、长 38 cm，在机器人右侧或左侧（随机；种子 1000–1019 里右 15、左 5），墙面 |y| = 25–27 cm，中心 x ∈ [−0.17, −0.12]，高出搁板面 4.5 cm。
- **搁板**：34 cm（沿墙）× 20 cm（进深）× 1.2 cm，浅木色，面高出桌面 11 cm；铰链在墙面（轴沿 x，在板厚中间），下垂硬限位 25°。agentview 相机在桌子远端（+x）朝机器人看，从搁板端面看过去，所以下垂在画面里是板子的转动（前沿往下）。
- **4 个方罐**（红、绿、蓝、黄）：罐身 5 × 5 × 6 cm、白盖、1.2 cm 颈上一个 2.2 cm 深色球形把手；总高 10.2 cm，0.24–0.26 kg，重心离罐底 2.6 cm。在桌子另一侧（|y| 13–17 cm）排成一行，间距 9 cm，颜色顺序随机。罐子–搁板摩擦 0.27–0.33（每个罐子各抽一个）。
- **扶起躺倒的罐子**：把手是球（扭转摩擦 0.0005，condim 4），从上面夹住躺倒罐子的把手往上提，罐子在两指间转动、竖直挂下来（同 lh_bystander 的瓶盖）；指令里写了这一点。

## 隐变量：支架
- **松**（每 20 个连续种子 14 个，`(seed × 7) mod 20 < 14`）：预载 0.62–0.68 N·m（种子 1000–1019 实际 0.626–0.679）；**紧**（6/20）：2.5 N·m，任何放法都达不到。两种外观完全一样。
- **v2 机制**（实测）：预载 = 弹簧 + 卡扣（0.12 N·m）。罐子对铰链的载荷力矩低于预载时，板顶在限位上保持水平；超过后卡扣在下垂的前 3° 内松脱，只剩软弹簧（预载之外 0.25 N·m/rad）。罐子重心在铰链轴上方 3.2 cm，板一下垂重心就往外移（每个罐子 +0.08 N·m/rad），所以板上有 3–4 个罐子时没有中间平衡点：阻尼 0.45–0.60 N·m·s/rad 下约 1 s 折到 25° 限位，罐子从前沿滑到桌上、侧躺。罐子都滑掉后板子弹回水平（agent 看到的是"板子是平的，罐子在桌上"，见冒烟测试截图）。
- 卡扣是按板子角度施加的关节力矩（`_bracket()`，每个 MuJoCo 子步写 `qfrc_applied`，做法同 lh_drawer_cabinet 的卡扣）。
- 力矩（罐重 2.45 N × 重心到铰链的水平距离；预载 0.62–0.68）：

| 放法（罐心离墙面） | 力矩 N·m | 松支架 |
|---|---|---|
| 4 个靠墙（3.5 cm，留 1 cm 缝） | 0.34 | 不动 |
| 4 个 6 cm | 0.59 | 不动 |
| 4 个 7 cm | 0.69 | 翻 |
| 中间（10.4 cm）1 / 2 / 3 个 | 0.25 / 0.51 / 0.76 | 不动 / 不动 / **翻** |
| 前沿（16.4 cm）2 个 | 0.80 | 翻 |

- 这个量放上去之前测不出来（R2）：两种支架外观相同；只有加载才暴露。

## 为什么常识会失败
- 常识做法（naive，模仿 Sonnet 在其他任务里的谨慎习惯，**本任务还没有真模型小试，"放在中间"是推断**）：从上面夹把手，抬到 25 cm 搬运，把罐子放在搁板**进深的中间**，从远端开始一个一个放，放完看一眼刚放的那个，finish 前不回头检查。
- 第 3 个罐子放下后，载荷力矩 0.76 > 预载：验证里 naive 13/14 次板子在 0.1–0.55 s 内过 8°，0.85–2.05 s 后前两个罐子和第 3 个一起滑下前沿、侧躺在桌上——这时夹爪刚松开、正在抬起或已经去拿第 4 个（另 1 次见已知问题 s1015）。naive 只把"刚放的那个"重新放回中间（它只检查刚放的），再放第 4 个（两个罐子 0.51 < 预载，不再翻），finish 时桌上躺着 2 个罐子（14/14 松实例都是这样，实测）。
- **R3（第一次常识交互的代价）**：3 个罐子掉下来侧躺，每个要按把手提起、等它挂直、再放回去（约 8 条命令），而且放回中间会再翻（对照 naive_check_mid）。
- **R4（同一失败外观的解释）**：放得不稳 / 松手太高砸翻了 / 手臂碰到了 / 搁板太滑——最省事的这几种都是错的；对的是"离墙远的载荷压翻了支架"。翻板发生在松手之后约 0.3 s，正好像是"我松手松坏了"。

## 要学到什么
- **尝试内**：放完每个罐子后回头看整个搁板（不只看刚放的那个），发现罐子躺在搁板前沿外的桌上 → 按把手提起扶正 → 放回**靠墙**处；已经在板上、离墙远的罐子也挪到墙边。
- **跨尝试**：从一开始就靠墙放（罐心离墙 ≤ 6 cm，即罐子背面离墙 ≤ 3.5 cm）。

## 判定（`outcome()`）
- finish 后机械臂保持不动、仿真 ≥ 15 s（板子或罐子还在动就延长，最多 30 s）再判，所以"最后一个罐子放下立刻 finish"抢不过支架。
- 成功条件见上表。搁板限位从 v1 的 12° 放宽到 15°：v2 里板子要么水平、要么折到 25°，不会停在中间带着站立的罐子（罐子在 ~14.5° 开始滑）；验证（108 次尝试）和 v2 风险扫描（224 次）里 0 次 `board_drooped`，这个条件实际上不再起作用，只防"把板子撑住/罐子卡在下垂板上"之类的情况。
- `failure`：`none / held / propped / touching / slid_off / knocked_off / regressed / not_on_shelf / board_drooped / not_static`。
- F2 `detail` 例（只用于消融，主表是 F1）：
  > The red jar lies on its side on the table, 3 cm out from the shelf's front edge. It had stood on the shelf; it slid off when the shelf board drooped under the load. The blue jar lies on its side on the table, 2 cm out from the shelf's front edge. It had stood on the shelf; it slid off when the shelf board drooped under the load. During the attempt the shelf board drooped 1 time(s) (down to 25 degrees) and 3 time(s) a jar slid off; 1 time(s) a jar was put back.

## 事件监视器（只进服务器日志，agent 看不到）
- `DroopSkills._act` 每个控制步调用 `monitor_step`；每个 MuJoCo 子步收集机器人–罐子、罐子–罐子、搁板–非罐子的接触。
- 阶段（每个罐子一个，`red / green / blue / yellow`）：`achieved / lost / restored`（0.5 s 去抖）；`lost` 带 `cause`：`slid_off`（板子在下垂中或 3 s 内下垂过 5°）/ `robot` / `jar:<颜色>` / `picked_up` / `other`；`restored` 带 `after`。
- 搁板：`sag_start`（> 1.5°）、`sag`（> 8°，= `trap_fired`）、`sag_end`（回到 < 0.5° 并保持 1 s），带"距让力矩超过预载的那次放置多久"、当时的力矩和预载；`jar_fell`（距它自己放下多久、距翻板多久、是否在最终静置里）。
- `outcome()` 返回 `success, stages, events, n_regressions, n_lost, n_restored, n_fell, n_sag, trap_fired, sag_visible, recovered_in_attempt, first_sag_dt_s, first_sag_start_dt_s, first_sag_after, first_fall_dt_s, fell_before_finish, board_deg_at_finish, off_at_finish, max_droop_before_finish_deg, visible_before_finish, fell_jars, failure, …, detail`。`*_at_finish` / `visible_before_finish` 记录 agent 最后一张图能看到什么（finish 不返回图像）：`visible_before_finish` = finish 前板子过了 8° 或已有罐子离开搁板。v2 改动：板子翻下又弹回很快，原来"`in_sag` 时丢失才算 slid_off"会把滑落记成 `other`，改成"3 s 内板子下垂过 5°"；`sag_end` 要求回平 1 s。

## 脚本策略（完美感知 = 仿真状态；只用 move_to + 开/合爪）
- 共用动作：开爪 → 抬到搬运高度 → 平移到把手上方 → 竖直下降到把手 → 合爪（躺倒的罐子：手指横跨罐轴，先慢慢提到挂直、等摆动停下）→ 抬到 25 cm → 平移到搁板上的目标点 → 竖直下降到罐底离板面 `h_rel` → 开爪 → 抬起。目标点按"板子水平"计算。
- **naive**：罐心离墙 10.4 cm（进深中间）± 1 cm（每实例的偏差）± 0.6 cm（每个罐子），松手高度 1–4 cm（按种子），yaw 0，速度 0.35 m/s，每个罐子约 8 s 仿真时间，从离机器人最远的罐子开始；放完只看刚放的那个，不在板上就重放一次；finish 前不回头。
- **oracle**：靠墙（3.5 cm），无误差，松手 0.5 cm，每个罐子后回头看、最后检查。
- **adaptive**：第 1 次 = naive + 每放一个就回头看：有罐子掉了或板子 > 3°，从此改放靠墙（4.3 cm，误差同 naive），并修复（板上离墙远的挪到墙边，掉下的扶起放进空位），最后再检查；第 2 次起一开始就放 4.3 cm、同样回头看。
- **blind**：第 1 次 = naive；之后随机：进深 U(6, 15) cm、顺序随机、搬运高度 24–28 cm、松手高度 0.5–4 cm、速度 0.12–0.3；不看历史、不回头。
- 对照：**naive_front**（放前沿 16.4 cm）、**naive_back**（naive 的速度和误差，但放 4.3 cm）、**naive_check_mid**（naive + finish 前检查修复，但放回中间）、**naive_fast**（最快速度、放完立刻 finish）、**switch**（第 1 次 naive，之后只换靠墙、不回头修）、**oracle_noise**（oracle ± 5 mm）。

## headroom 验证（v2，n = 20，k = 3，实测，`python scripts/validate.py --task lh_shelf_droop --n 20 --k 3`，种子 1000–1019，1584 s）
20 个实例：松 14、紧 6；墙在右 15、左 5。

| 策略 | @1 | @2 | @3 | 第 1 次陷阱触发 | 备注 |
|---|---|---|---|---|---|
| oracle | **100%** | 100% | 100% | 0/20 | 板子最大下垂 0° |
| naive（模仿 Sonnet：放进深中间） | **30%** | — | — | **14/20 = 70%**（松 14/14，紧 0/6） | 失败 14/14 都是 `slid_off`；13/14 由第 3 个罐子触发，1/14（s1015）由第 2 个触发（4 cm 高松手砸的，见已知问题）；每次掉 2–3 个，finish 时桌上躺 2 个；14/14 在 finish 前就掉了（`fell_before_finish`） |
| adaptive | **100%** | 100% | 100% | 14/20 | 14 个触发的实例**全部在第 1 次内修好**：39 次 `slid_off` 丢失、39 次恢复；修完的罐子离墙 2.9–5.4 cm，力矩 0.35–0.45 N·m |
| blind | 30% | 30% | 30% | 14/20 | 松实例 3 次全失败（42 次）：随机进深 6.3–14.9 cm 都翻（6.3 cm 也翻，是 ±0.6 cm 误差和松手冲击） |

**可见性（风险 (a)，实测）**：验证里 56 次失败尝试，`sag_visible == False` 0 次，`fell_before_finish == False` 1 次（blind s1010 第 3 次：finish 时板子已经下垂 18.7°、罐子正在滑），两者都 False 0 次。

对照（实测，`p3_run.py`，种子 1000–1009（松 7、紧 3），`results/lh_shelf_droop/controls.json`）：
| 对照 | @1 | @2 | @3 | 松实例 | 说明 |
|---|---|---|---|---|---|
| naive_back（naive 的速度、误差、松手高度，但放 4.3 cm） | **100%** | | | 7/7 成功 | 0 触发：同样的笨拙动作，只要靠墙就不翻 |
| oracle_noise（oracle ± 5 mm） | 100% | | | 7/7 | |
| naive_front（放前沿 16.4 cm） | 30% | | | 0/7 | 第 2 个罐子就翻，4 个全掉（7/7），finish 前看得见 |
| naive_check_mid（finish 前检查并修复，但放回中间） | 30% | | | 0/7 | 7/7 修复时又翻：四个全掉，finish 时桌上躺 3 个——光"回头修"不够，要放到墙边 |
| naive_fast（最快速度，放完立刻 finish） | 30% | | | 0/7 | 7/7 失败且在 finish 前就掉了：快也抢不过翻板 |
| switch（第 1 次 naive，之后只换靠墙、不回头修） | 30% | **100%** | 100% | 第 2 次 7/7 | 跨尝试学习就够 |

## 时间测量（实测）
- **Sonnet 的节奏**（lh_nudge_row 小试 `runs/claude/np2_lh_nudge_row/`，3 个 run、254 条 `move_eef`，scratchpad `timing.py` / `tail.py`）：每条 `move_eef` 仿真时间中位 0.8 s（p10 0.32，p90 1.43）；两次松手之间中位 5.1 s（p10 3.2，p90 8.3）；最后一次松手后通常只再发 1–2 条命令（0.5–1.6 s）就 finish。`observe` / `status` 不推进仿真时间，没有 wait 命令。
- **脚本 naive**：每个罐子约 8 s（比 Sonnet 慢）；最后一个罐子落到板上后约 0.8–1.1 s 就 finish（比 Sonnet 快，是可见性的保守下限）。
- **v2 翻板时间**（从让力矩超过预载的那个罐子落到板上算）：验证 70 次触发，过 8° 中位 0.3 s（p10 0.15，p90 0.76，最大 1.55）；第一个罐子滑落中位 1.25 s（p10 0.95，p90 2.6；最长 8.65 s 是两罐慢翻）。风险扫描 v2 的 144 次触发：过 8° 中位 0.4 s（p90 0.8，最大 2.8）；第一个罐子滑落中位 1.55 s（p10 1.05，p90 2.9）。
- **v1 对比**：v1 是一阶滞后（τ 4.3–5.7 s），第 3 个罐子之后 3–5 s 才滑落，正好在拿第 4 个罐子时——但如果超载的是第 4 个罐子，这个延迟就落在 finish 之后（见下）。

## 物理修正（v1 → v2，实测，`results/lh_shelf_droop/risk_sweep.json`）
扫描：松实例，naive 的速度和误差，每次 1 个尝试；`u<r>` = 4 个罐子都放在离墙 r cm（每罐 ± 0.6 cm、松手 1–4 cm）；`rnd` = 每罐进深 U(6, 12) cm、顺序随机；`c<r>` = 小心放（± 0.3 cm、松手 0.5 cm）。"看不见" = 失败，且 finish 前板子没过 8°、也没有罐子离开搁板。

| | v1（预载 0.45–0.49、弹簧 0.6 N·m/rad、阻尼 2.6–3.4，无卡扣） | v2（定版） |
|---|---|---|
| 尝试 / 失败 | 92 / 74（9 个松种子，到结论清楚为止） | 224 / 144（14 个松种子 × 16 种放法） |
| 失败里看不见的 | **15 / 74 = 20%**；finish 前没有罐子掉下的 30 / 74 = 41% | **1 / 144 = 0.7%**（s1004 `u6`：刚好超一点，过 8° 用了 1.65 s，罐子 3.65 s 后才滑；脚本 0.75 s 就 finish 了） |
| 4 个放 4 / 5 cm | 9/9、9/9 成功 | 14/14、14/14 成功 |
| 4 个放 6 cm（naive 误差） | **0/9 成功**，全看不见：板子在最终静置里慢慢垂到 14°（5 次 `board_drooped`）或罐子滑掉（4 次）；finish 时板子只有 1.5–2.6° | 10/14 成功；4 次失败里 3 次看得见 |
| 4 个放 7 cm | 0/9 成功，6/9 看不见（finish 时 3.9–8.4°） | 0/14 成功，全看得见 |
| 小心放 4 / 5 / 6 / 7 cm | — | 14/14、14/14、14/14、**0/14**（7 cm 全部翻、看得见） |
| 中间 9–12 cm、`rnd` | 全失败，看得见 | 全失败，看得见 |

- **原因（v1）**：弹簧 + 重心外移让板子在 6–8 cm 这一段只有第 4 个罐子才超载，而一阶滞后要几秒才垂到能看出来的角度，正好落在 finish 之后；另外 4 个罐子放 6 cm（罐子背面离墙 3.5 cm，算"靠墙"）在 v1 里也会失败——风险 (b) 成立，而且失败本身看不见。
- **考虑过的改法**：(1) 降预载让两个中间罐子就超载——靠墙区更窄（v1 已经 6 cm 失败），和 (b) 冲突，不用；(2) 只缩短 τ——中间那段仍然停在 8–14° 的半下垂状态（罐子站着、板子斜着），判不清；(3) **定版**：预载提到 0.62–0.68（4 个罐子 ≤ 6 cm 安全，3 个中间罐子 0.76 仍超载），加一个 3° 内松脱的卡扣、把弹簧降到 0.25 N·m/rad、阻尼降到 0.45–0.60：一旦超载就在 ~1 s 内折到底，不存在半下垂的稳定状态。
- **风险 (b) 的结论**：小心放在罐心离墙 4–6 cm（背面离墙 1.5–3.5 cm）都成功；7 cm（背面离墙 4.5 cm）失败，但现在是看得见的 `slid_off`，不是看不见的 `board_drooped`。搁板限位放宽到 15°（见"判定"），v2 里没有它能拦住的情况。
- 监视器同时修了：快速翻板后 `slid_off` 的归因、`sag_end` 去抖；新增 finish 时刻字段。

## pilot 种子（实测，scratchpad `pilot.py`：从 3000 起逐个跑 naive 和 oracle 各 1 次，取 naive 失败且陷阱触发、oracle 成功的前 3 个；`results/lh_shelf_droop/pilot_seeds.json`）
- 3000（松，墙在右，预载 0.632）：naive 掉下绿、红、黄，第一个 1.05 s 后滑落，oracle 成功；3001（松，墙在左，0.663）：掉下蓝、绿、黄，1.35 s，oracle 成功；3002 是紧支架（naive 成功，不算）；3003（松，墙在右，0.661）：掉下蓝、绿、红，1.05 s，oracle 成功。
- **pilot 种子 = 3000, 3001, 3003**。终态图：`results/lh_shelf_droop/pilot_300{0,1,3}_{init,naive,oracle}_agent.png`（naive / oracle 图左 agentview、右腕部）。

## agent 接口冒烟测试（la 风格，rgb，F1，k = 3，seed 3000，端口 15650，2026-10-16，实测）
`LFF_STYLE=la LFF_RUNS_DIR=<scratchpad>/runs scripts/start_robot_session.sh lh_shelf_droop 3000 full F1 15650 direct 3 smoke rgb 1`，不调用模型，用 `smoke/drive.py` 按 `api_player.call_server` 的方式发 HTTP 命令（JSON POST + `X-Token`）。日志：scratchpad `runs/smoke_lh_shelf_droop_s3000_full_F1_direct_k3_1016-042303_p15650/`（`events.jsonl`、`rollout.mp4`、快照 `snapshots/a1_s028`）。
- sandbox `README.md`：指令是上面的直接版（含把手那句），3 次尝试、每次 120 条命令 / 360 s 仿真运动、100 次工具调用，坐标是机器人基座系；内容正确。
- **第 1 次（naive 式，37 条命令）**：每个罐子 9 条 `move_eef`（上方开爪 → 降到把手上 3 cm → 降到把手 → 合爪 → 抬到桌面上方 25 cm（基座系 z = 0.238）→ 平移到搁板中间（罐心离墙 10.4 cm）→ 降到罐底离板 1 cm → 开爪 → 抬起），从远端开始：绿、红、黄、蓝。第 3 个（黄）在命令 26（开爪，仿真 15.4–16.2 s）里松开，板子 15.95 s 过 8°，命令 28（去拿蓝罐，16.45–17.95 s）期间绿、红、黄 16.75–16.85 s 滑落；蓝罐放到中间后板子没再动。看过的 512 px 图：`a1_s025`（三罐在水平板上）、`a1_s026` / `a1_s027`（板子明显斜下去，罐子跟着倾斜，腕部图也看得出罐子歪了）、`a1_s028`（板子弹回水平，三个罐子侧躺在搁板前方的桌上，把手朝外）、`a1_s037`（finish 前最后一张：板上只有蓝罐，桌上躺三个）。拷贝在 `results/lh_shelf_droop/smoke_3000_a1_s0{25,27,28,37}_*.jpg`。`finish` → 只回 **"Attempt failed."**；日志 `failure=slid_off`、`trap_fired=True`、`sag_visible=True`、`fell_before_finish=True`、`off_at_finish=[red, green, yellow]`，`sag` 归因 `after_placing=yellow, dt 0.25 s`，3 个 `jar_fell`（`dt_since_cause_s` 1.05–1.15）；回退快照 `a1_s028` 已存。
- **第 2 次（同样的动作，只把目标改成罐心离墙 4.3 cm，37 条）** → **"Attempt succeeded."**，四个罐子离墙 4.5–4.6 cm，力矩 0.437 < 预载 0.632，事件只有 4 个 `achieved`（`smoke_3000_a2_s038_near_wall_success_agentview.jpg`）。
- 结束后只停了这个 server（PID 603499）。

## 演示视频
`results/lh_shelf_droop/demo_adaptive_s3000.mp4`（`python scripts/make_video.py --task lh_shelf_droop --kind adaptive --seed 3000 --k 3 --out …`；adaptive 在第 1 次内翻板、扶起三个罐子放回墙边、成功）。

## 已知问题 / 推测（未验证的标【推断】）
- **还没有真模型小试**。naive 把罐子放在"进深中间"是按其他任务里 Sonnet 的谨慎习惯推的【推断】。如果模型第 1 次就把罐子往里放（"搁板上东西放靠里更稳"，罐心 ≤ 6 cm），陷阱不触发，等于一个紧实例；放在 7–8 cm 会触发（看得见）。上线前要先小试（DESIGN_INSIGHTS §2.3）。
- **延迟变短了**：v2 罐子在松手后 1–3 s 滑落，agent 在抬手或去拿下一个罐子时看到，"在做下一阶段时前面的被毁"变成"刚松手就塌了"。好处是 R4 更强（像是自己松手松坏的），代价是"延迟自损"的味道弱了。设计取舍：用户原则"陷阱和可见性冲突时优先保可见性"。
- **安全区 = 罐心离墙 ≤ 6 cm**：学到"往里放"但离墙留 4–5 cm（罐心 6.5–7.5 cm）的 agent 还会失败（看得见），可能要再试一次才放到够近。naive 的 ±0.6 cm 误差下 6 cm 也有 4/14 失败。
- **松手冲击能提前触发**：s1015（预载 0.626）naive 第 2 个罐子从 4 cm 高松手，两个中间罐子静力矩 0.56 < 预载，但冲击把板子推进了卡扣区，板子慢慢翻下去（过 8° 用 1.45 s，罐子 8 s 后才滑）。两个罐子时重心外移不够，翻得慢；三个以上都很快。物理上合理（往弱支架上砸东西），但让"安全"和松手高度有关。
- 仅剩的看不见的失败（扫描 1/144）发生在载荷刚好超过预载一点、且最后一个罐子触发、并且放完立刻 finish 的组合下。
- `sag_start` 的归因可能差一个罐子：板子在罐子落下的同一个控制步开始动时，原始"放到板上"谓词每两步才采样一次，`sag_start.after_placing` 会记成前一个罐子（冒烟里记成 red，实际是 yellow）；`sag`（> 8°，即 `trap_fired`）的归因是对的。只影响日志。
- 卡扣是按板子角度算的力矩（`qfrc_applied`），在每个子步之后更新、滞后一个子步（2 ms）；从快照重放时第一个子步用的是旧值【推断：影响可以忽略，未单独测】。
- 掉下的罐子一般躺在搁板前沿外 2–4 cm 的桌上；少数可能滚到搁板下面（F2 文本有 "under the shelf" 分支），从上面夹把手会被搁板挡住【推断：验证里 adaptive 全部修好，没遇到】。
