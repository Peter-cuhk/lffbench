# L3-C `l3_under_table_tool`：玻璃矮桌下的方块，要用棍子推出来；v2 起有一侧被看不见的挡板封住

## v2（2026-10-08，用户审阅 cla2 后改）：棍子对面那一侧有透明挡板
> 代码仍是 `lffbench/tasks/l3_tool.py`（v1 备份 `l3_tool.py.bak-20261008-pane`，v1 验证结果另存为 `results/l3_under_table_tool/validate_v1.json`）。本节之后的"v1"各节保留作记录，场景尺寸、棍子、抓取细节仍然适用。

**为什么改**：v1 的隐藏信息是"手伸不进玻璃下面"。cla2 里 Sonnet 4/4 个 run 看到矮桌边的 T 形棍就直接用棍子推，没有一个先伸手去抓，第 1 次全部成功（独立单次 3/3）。陷阱被动手前的推理绕过了，不需要从失败里学。

| 项 | v2 |
|---|---|
| 隐变量 | 矮桌**远离棍子**的那一侧（左或右），两条腿之间有一块透明亚克力挡板，4 mm 厚，从桌面一直到玻璃下表面。只有碰撞，不渲染（rgba alpha 0，没有可视副本），两个相机都看不到。另外三侧开着 |
| 布局改动 | 盒子（bin）从矮桌前方挪到挡板那一侧，离矮桌 4–6 cm；矮桌整体比 v1 远离机器人 4 cm（x ∈ [−0.12, −0.06]），这样从后面往前推时够得着 |
| 先验（naive） | 拿起棍子，从棍子所在那一侧伸进去，把方块横着推向盒子（cla2 里 Sonnet 全都这么做）。方块撞上看不见的挡板，停在桌边，仍然完全在玻璃下面，手够不着 |
| 要学到什么 | 那一侧被看不见的东西挡住了。F1 下只能从图像看出来：方块推到边上就停了，没出来，棍子也停住了。于是换方向：把棍子放到矮桌后面（机器人这一侧）朝前，从前侧推出来，再抓起放进盒子 |
| oracle | `tool_push_front`：在质心处夹起棍子，空中转腕让棍头朝前，放到矮桌后方与方块对齐；在离自由端 7 cm 处重新夹住，往前推到方块中心出前沿 5 cm；退回、放下棍子，再抓方块进盒子 |
| adaptive | 横推后方块仍在桌下 → 下一次改前推；前推没推出来 → 多推 2 cm |
| F2（只用于消融） | "The cube was pushed against a clear acrylic panel that closes the left/right side of the low table …; it cannot leave the low table on that side." 再加原有的"还差 X cm 出桌下" |
| 结构化字段 | `mode` 新增 `pane`；`cube_hit_pane`、`stick_hit_pane`、`robot_hit_pane`（可以据此统计 agent 有没有在推之前先去探挡板） |

### v2 headroom 验证
VALIDATION_L3_PLACEHOLDER

### v2 实例 seed 3000（cla3 预检 / cla4 用）
棍子在左边（side = +1），挡板在右边（pane_side = −1）。总管的 `results/_cla4_seed_check/trapcheck.py` 实测：naive 失败（`mode = pane`，方块撞挡板），oracle 成功（`results/_cla4_seed_check/l3_under_table_tool_3000.json`）。
SONNET_L3_PLACEHOLDER

### v2 已知问题 / 效度威胁
- 挡板完全不可见是有意的。真实的透明亚克力在图像里也很难看出来，但不是完全看不见；这里是一个不可感知的隐变量，只能从"方块推到边上就停了"去推断。
- 可以主动探测绕过：先用夹爪或棍子沿那一侧探一下，回复里的 `stopped_from_target` 会显示被挡住。这算尝试内的主动感知，用 `robot_hit_pane` / `stick_hit_pane` 和首次推的方向单独统计。
- 尝试内恢复：第一次横推失败后，agent 可以在同一次尝试里改成前推。按用户 10-08 的口径这也算从失败中学习，但会被算进第 1 次成功率。
- 前推要把棍子转 90° 重新摆放，比 v1 的横推难。棍子夹在自由端 3.5 cm 处（v1 的抓法）往前推时，手掌会撞到玻璃后沿（手在指尖上方 4–13 cm），所以 oracle 改夹 7 cm 处。

---

# v1 记录（2026-10-07）

> 代码：`lffbench/tasks/l3_tool.py`。验证：`results/l3_under_table_tool/validate.json`。视频：`videos/l3_under_table_tool_adaptive_s1000.mp4`。2026-10-07 完成。
> 这是 L3 第二个新任务。原计划的 `l3_flat_card_edge` 和 `l3_fallen_mug`（做成了倒下的瓶子）在本仿真里都做不成，放弃原因和实测数字见文末。

| 项 | 内容 |
|---|---|
| 等级 / 类别 | L3 更换操作方式（`strategy_switching`） |
| 能力 | Perceive（看出夹爪停在玻璃上、方块没动）、Reason（手进不去：从上面被玻璃挡住，从侧面手掌比桌下空间高）、Plan（改用工具：拿棍子、对准、推出、放下棍子、再抓方块） |
| 协议 | cross，每次尝试重置到同一实例，k = 5 |
| 指令 | 直接："Put the red cube into the gray bin." 间接："Put the small red block that is under the low glass table into the gray container." |
| 场景 | 玻璃矮桌（固定 fixture）：桌面 12 × 12 cm、厚 1 cm、半透明，下表面离桌面 7 cm，四条腿 1.6 cm 见方。红色方块 4 cm 在矮桌正下方。T 形木棍（自由物体）：柄 24 × 1.6 × 2 cm，头 5 cm 宽，躺在矮桌左侧或右侧，头朝矮桌，离矮桌边 2.5–4.5 cm。灰色 bin（内 12 × 12 cm、壁高 5 cm）在矮桌前方 |
| 随机 | 矮桌位置；方块在桌下 ±1 cm；棍子在左边还是右边、朝向 ±10°、离桌边距离；bin 位置 |
| 成功判据 | 方块中心在 bin 内框以内、低于 bin 壁顶，夹爪已松开 |

## 任务 → 隐藏信息 → 所需能力 → 调整后的动作
- **任务**：把方块放进 bin。先验做法是直接去抓。
- **隐藏信息**：机器人手的尺寸和桌下空间的关系。方块看得见（桌面是玻璃），但玻璃下表面只有 7 cm 高：从上面抓，手指停在玻璃上；从侧面低位伸进去，手掌（在指尖上方约 4–13 cm、沿合拢方向宽 20 cm）先撞到玻璃边。agent 不知道手有多大，单看图也不容易判断这 7 cm 够不够。
- **所需能力**：从"停在玻璃上 / 撞到玻璃边、方块没动"的失败里推断：用手根本够不到，换别的抓法也没用。
- **调整后的动作**：用工具。抓住棍子柄的末端（合拢方向垂直于柄），稍微抬离桌面，横向挪一下让棍头对准方块，沿柄的方向慢速平移，把方块从矮桌另一侧推出来（离桌边 ≥ 5 cm），放下棍子，再抓方块（手的长边要和桌边平行，否则手会碰到玻璃），放进 bin。全部只用 move_to 和开合夹爪。

## 为什么按先验做会失败（实测）
- agent 式直接尝试，seed 2000–2009 和 3000–3009 共 20 个实例 × 6 种：无保护抓取（`Skills.grasp_at`，site 压到桌面 +0 / +2 cm，xy 加 σ = 1 cm、yaw 加 σ = 0.3 rad 扰动），以及合爪在 site 高 2 cm 从前、后、左、右四个方向往桌下平推 28 cm。**120/120 次被挡住，方块一次也没动。**
- 从上面抓时指尖停在桌面上方约 8 cm（玻璃顶面）。从侧面推时手掌撞上玻璃边，被顶高到指尖约 7.5 cm。

## F2 反馈（脚本跑出来的原文）
- 第一次直接抓（naive，seed 1000）："The gripper came down on the glass top of the low table: its fingertips stopped 8.0 cm above the tabletop, while the cube is underneath the glass (the glass underside is 7.0 cm above the tabletop), so the gripper never got to the cube. The cube did not move; it is still under the low table."
- 合爪从侧面往里推（seed 1003）："The hand ran into the edge of the glass top of the low table (fingertips 7.5 cm above the tabletop, gripper 0.9 cm outside the edge) and could not move in under the glass. The cube did not move; it is still under the low table."
- 推出来了但没放进 bin（seed 1003，脚本故意把 bin 位置读偏 13 cm）："The cube was lifted but ended outside the bin: it is 0.1 cm forward and 13.0 cm left of the bin's centre."
- 推出来以后用 yaw 0 抓（手的长边横跨桌边，抓取失败，seed 1003）："The cube is out from under the low table but not in the bin: it is 20.0 cm backward and 11.7 cm left of the bin's centre."
- 推了但还在桌下（文字模板，验证里没出现）："The cube was moved X cm but is still under the low table (Y cm from getting completely out from under it)."
- 成功："The cube is in the bin."

只给相对量（离桌面多高、离桌边多远、离 bin 中心往哪个机器人方向偏多少），不给绝对坐标，不提棍子。文字由逐步事件监视器生成（包 `env.step`），agent 自己调 move_to 的尝试也一样。

## 脚本参照策略
- oracle：用棍子。抓柄末端往里 3.5 cm 处，抬 6 mm；按观测到的棍头位置横向平移，让棍头对准方块；沿"垂直于远侧桌边"的方向（纯 ±y）以 8 cm/s 推，推到方块中心离远侧桌边 7 cm；仍在桌下就每次再推 2.5 cm（最多 3 次）；松开棍子，抓方块（手的合拢方向平行于桌边），放进 bin。
- naive：在方块正上方张开、带接触感知往下降（碰到东西就停）、合爪、抬起；抓到才去 bin。
- adaptive：第 1 次同 naive；读到 `blocked` 换成 oracle 的用棍子推法；如果推完方块还在桌下，下一次多推 4 cm。
- blind：每次都直接抓，加扰动（dx、dy ~ N(0, 0.8 cm)，dyaw ~ N(0, 0.15 rad)），不看历史。

## headroom 验证（`scripts/validate.py --task l3_under_table_tool --n 20 --k 5`，seed 1000–1019，实测）

| 策略 | @1 | @2 | @5 |
|---|---|---|---|
| oracle | 100 | 100 | 100 |
| naive | 0 | 0 | 0 |
| adaptive | 0 | 100 | 100 |
| blind | 0 | 0 | 0 |

- 失败模式：naive 20/20、blind 100/100 次尝试都是 `blocked`；adaptive 第 2 次全部成功。耗时 249 s。
- 额外 oracle 抽查：seed 2000–2029 共 30 个新实例，30/30 成功。
- 开发中修过的两处（都是脚本参照策略的问题，不是任务设计）：(1) 拿着 30 cm 的棍子转手腕，棍子会在指间打滑转走，所以现在棍子初始就大致朝着矮桌，脚本只平移不转；(2) 沿棍子自身方向推（偏 y 轴最多 10°）时棍头横向漂移、蹭到桌腿，所以改成沿 y 推、桌腿改细（内侧间距 8.8 cm，棍头 5 cm）。第 (2) 处改之前 oracle 26/30（seed 1000–1029），改之后同一批 30/30。

## agent 接口自测（端口 9842，2026-10-07）
`start_robot_session.sh l3_under_table_tool 1000 full F2 9842 direct 3 selftest rgb 1`。README 只有通用说明和任务句（没有提棍子、桌高、手的尺寸）。初始图像里棍子、玻璃矮桌、桌下的方块、bin 都看得见。第 1 次：在方块上方张开、move_to 降到 z = 0.92、合爪、抬起、done，得到"fingertips stopped 8.3 cm above the tabletop ..."。第 2 次（13 条命令）：yaw 83.8° 抓住棍柄末端 → 抬到 z = 0.918 → 平移 0.5 cm 对准 → 以 0.08 m/s 推到 y = −0.055 → 松开 → 在方块上方 yaw 90 抓起 → 移到 bin 上方松开 → done，成功。无报错，会话和目录已删除。

## 已知问题 / 效度威胁
实测：
- harness 的 move_to 报告里 `touched_object` 不包括 fixture（`ContactTracker` 只登记 `objects_dict`，矮桌是 fixture），所以撞到玻璃时报告仍是 `touched_object: false`。agent 只能从图像和 F2 知道撞了玻璃。l3_wall_flush 的墙也是这样。这是共享文件 `agent/tools.py` 的行为，没改。
- 推出来之后，如果 agent 用 yaw 0 去抓（手的 20 cm 长边横跨桌边），手会碰到玻璃、抓不起来；F2 只说"方块出来了但不在 bin 里"，没有说手撞到了玻璃（上面 YAW0_GRASP 的例子）。
- 自测时用的是精确的实例坐标，真实 agent 要从 RGB 估计棍子、方块的位置；棍头只有 5 cm 宽，对不准时方块会被斜推、可能偏离。

推测（未验证）：
- 初始画面里放着一根棍子，强模型可能一看就猜到要用工具，第 1 次就成功；attempt-1 成功率要和 succ@k、ΔICL 分开报告。
- 如果 agent 拿着棍子转手腕（比如想把棍子转到别的方向），棍子会在指间打滑（脚本开发时实测过）。这是额外难度，未必是坏事，但可能让一些合理的计划失败。

## 放弃的两个原设计（实测）
**`l3_flat_card_edge`（贴桌面的薄卡片，先推到边上悬空再夹）**：卡片 10 × 5.5 × 0.4 cm 放在 4 cm 高的木板上。
- Panda 手指的碰撞外壳（凸包，18 个顶点）内侧面一直延伸到指尖；指垫虽然从指尖往上 4.6 mm 才开始，但外壳本身就能夹住卡片 4 mm 厚的边。agent 式抓取（指尖相对板面 −8 / −4 / −2 / 0 / +1 / +3 mm，xy σ = 6 mm，yaw σ = 0.1 rad，每档 8 个实例），按卡片真实高度判定是否抬起：7/8、8/8、7/8、6/8、5/8、6/8 抬起来了。
- 开发初期的"150 次 0 次抬起"是错的：那次用的是 robosuite 的 `_check_grasp`，它只看指垫接触，看不到外壳夹住的情况。是在 agent 接口自测时从图像里看到卡片被夹起来才发现的。
- 另外，用指尖推卡片的端面时，手指底部的倒角会骑上卡片（目标悬出 4 cm，按推的方向和指尖高度不同，实际悬出 −2.0 到 3.9 cm）。
- 结论：先验抓法在这个仿真里大多能成功，任务不成立。

**`l3_fallen_mug` → 改成倒下的瓶子（瓶身 Ø 8.3–8.9 cm、瓶颈 Ø 2.8 cm，要求立起来放进方框）**：
- 瓶身虽比开口宽，但指垫摩擦系数是 2，夹在圆柱赤道以上也能夹住：4/4 个实例瓶子被躺着抬起来了。按摩擦锥估算，要让夹不住，接触点法线要在竖直 27° 以内，瓶身要粗到约 18 cm。
- 原本打算让 agent 改抓瓶颈、靠重力把瓶子吊直，但实测抓着瓶颈提起后，瓶子只转到离竖直约 49° 就停住（指间的转动摩擦撑住了），放下会倒。
- 躺着的杯子从上往下也抓不到杯沿（只有顶部的平夹爪），所以 fallen_mug 原设计也不可行。
