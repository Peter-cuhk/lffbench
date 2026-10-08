# L7 `l7_hidden_mechanism`：打开机构未知的小柜门

> **挂起（2026-10-08）**：用户意见——滑门那种设置太阴，正常人抓住把手只会往外拉，不会去横着拉。由 `l7_key_turn`（钥匙转错方向→换方向）替代。代码保留并仍注册（旧 run 可复现），验证结果移到 `results/_parked/l7_hidden_mechanism/`。

代码：`lffbench/tasks/l7_hidden_mechanism.py`　结果：`results/_parked/l7_hidden_mechanism/validate.json`　视频：`videos/l7_hidden_mechanism_adaptive_s1000.mp4`（左铰，2 次成）、`videos/l7_hidden_mechanism_adaptive_s1001.mp4`（右滑，3 次成）

| 项 | 内容 |
|---|---|
| 类别 | failed_interaction_inference（L7，"搞清楚为什么这次接触没起作用"），`level="L7"` |
| 能力 | Perceive（看门/旋钮怎么动了）、Reason（"转了一点、旋钮往一侧偏" vs "完全没动" → 机构）、Plan（换拉的方向） |
| 协议 | cross，k = 5 |
| 场景 | 小木柜 14 × 24 × 10 cm，门朝外（背对机器人、朝前置相机），门中间一根黑色立柱旋钮（Ø1.8 cm，顶在桌面上 7.7 cm）。机器人从柜顶上方伸过去，从上往下抓旋钮。柜子位置 x ∈ [-0.15, -0.10]、y ∈ [-0.07, 0.07] 随机 |
| 规则 | 指令里写明并在 outcome 里强制：**每次尝试只能拉一下**——抓住旋钮，持握状态下只走一条直线 move_to，然后松开；推门/推旋钮也算一次。判定：一条 move_to 若（开始时握着旋钮）或（门动了 >1° / >0.3 cm）或（夹爪碰到门/旋钮且指令水平位移 ≥1.5 cm）就算一次拉；超过一次判失败 |
| 成功 | 铰链门开到 ≥ 60°，滑门滑开 ≥ 12 cm，且只拉了一次 |

## 任务 → 隐藏信息 → 所需能力 → 调整后的动作
- **任务**：把柜门开大。
- **隐藏信息**：门的机构，四选一（均匀）：左边缘铰链（HL）、右边缘铰链（HR）、向左滑（SL）、向右滑（SR）。铰链和滑轨都不建可见模型，四种外观完全一样，动手前看不出来。
- **先验为什么失败**：自然做法是抓旋钮垂直门面往外（forward）直拉 12 cm。
  - 铰链门：旋钮要沿铰链画圆，直拉只能把门带开 30–34°，旋钮被拽出指间（验证集 8 个铰链实例实测）；旋钮停在前方 5.0–5.6 cm、**往铰链一侧**偏 3.7–4.3 cm，门保持半开。
  - 滑门：完全不动（< 0.3 cm），手指从旋钮上滑脱。
- **要从失败里学到 / 调整后的动作**：
  - 旋钮"往前+往左"动了 → 左边铰链 → 朝左前方约 55° 拉 22 cm（实测 80–84°）；右边同理。
  - 旋钮完全没动 → 不是铰链，是滑门 → 横着拉 18 cm。**滑向哪边第一次失败里看不出来**（四类里唯一不可辨的 1 bit），最坏要再失败一次：往左拉也没动 → 往右。

## F2 反馈（只给相对测量，不给绝对坐标、不说机构名）
铰链门推正（seed 1000，HL）：
> Attempt failed. The door did not open wide enough. During your (first) pull the knob moved 5.5 cm forward and 4.1 cm to the left; the gripper lost hold of the knob during the pull. When the gripper first closed, its centre was 0.1 cm backward and 0.0 cm to the right of the knob's centre and 0.2 cm below the top of the knob; it was holding the knob.

滑门："… During your (first) pull the knob did not move at all (less than 0.3 cm); the gripper lost hold of the knob during the pull. …"
其他失败文字：拉了多次（"The door or knob was pulled or pushed in N separate moves; only one pull is allowed per attempt…"）、没拉（"the knob was never pulled"）、没抓住（"…did not get hold of the knob."）。
旋钮位移取"第一次拉之后静止下来"的位置（到下一次拉开始或尝试结束）。
柜子是 fixture，harness 的 `holding_object` / `touched_object` 只看可动物体；用 `on_agent_tool` 钩子给动作报告加了 `holding_knob` 字段，并把碰门算进 `touched_object`（`observe` / `status` 的本体读数里没有 `holding_knob`，那两个不经过钩子）。

## 物理标定（实测，2026-10-06/07）
单次直线拉（握点高 7.5 cm，0.1 m/s），α 为拉的方向与正前方的夹角（朝铰链一侧为正），表中为门最终开角：

| 拉法 | 结果 |
|---|---|
| α 0°，12 / 20 cm | 31–36°（旋钮被拽脱） |
| α 30°，12 / 20 cm | 44° |
| α 45°，12 / 20 cm | 50° / 62° |
| α 50°，18–24 cm，3 个柜子位置 | 70–78°（HR 一处 86°） |
| α 60°，18 cm / 24 cm，3 个柜子位置 | 67–79° / 86–98° |
| α 70°、80° | 0–109°，不稳定（手指卡门边） |
| α −45°（拉向非铰链一侧），15 cm | 21–22°，旋钮仍往铰链一侧偏 2.4–2.6 cm |
| 滑门：横拉 18 cm（正确方向） | 17.8–18.0 cm |
| 滑门：斜 60° / 45° 拉 20 cm | 4.0 / 2.3 cm |
| 滑门：反方向 / 正前方拉 | 0 |

- 握点放低到 6.2 cm 后铰链门结果明显变差（α 50–60° 只有 41–92°，左右不对称），所以脚本策略用 7.5 cm。
- **一个坑（已修）**：运行时把关节从 hinge 改成 slide 后，MuJoCo 的 `dof_invweight0` 还是编译时那个"过门中心的铰链"的值，限位约束因此软了约 200 倍——滑门被反方向拉时冲过限位 17 cm、松手后弹回（实测）。现在 `apply_instance` 按当前机构重设 `dof_invweight0`，并把限位 `solref` 设为 0.004；修后反向拉只进 0.1 cm。
- 正前方拉滑门时门会朝它的打开方向挪 ~1 mm（实测 0.1 cm），低于 F2 的 0.3 cm 阈值，图像上也看不出。

## headroom 验证（n = 20，k = 5，实测，`python scripts/validate.py --task l7_hidden_mechanism --n 20 --k 5`）
20 个实例里：SL 9、SR 3、HL 3、HR 5（按种子随机，未配平）。

| 策略 | @1 | @2 | @3 | @5 |
|---|---|---|---|---|
| oracle（知道机构：铰链 ±55°/22 cm，滑门 ±90°/18 cm） | 100% | | | 100% |
| naive（正前方拉 12 cm） | 0% | | | 0% |
| adaptive（看第一次拉时旋钮位移：前+侧 → 铰链在那侧；只侧 → 滑向那侧；没动 → 先左后右） | 0% | 85% | 100% | 100% |
| blind（正前方 ±8°、12 ± 2 cm 随机） | 0% | 0% | 0% | 0% |

adaptive @2 未成的 3 个都是右滑门（先试左、没动、第 3 次往右），符合设计。blind 在铰链门上最大开到 35.8°。

## agent 接口自测（端口 9921，rgb，F2，k=3，seed 1000 = HL，2026-10-07）
README 无泄露（只有指令和通用说明，没有 hinge/slide 的实例信息）。第 1 次正前方拉 → 上面那段 F2 文字，画面上门绕右侧（图像右 = 机器人左）半开；第 2 次朝左前 55° 拉 22 cm → "Attempt succeeded. The door is open wide."。自测时发现 `touched_object` 对 fixture 恒为 false，已用钩子补上（见上）。

## 已知问题 / 推测（未验证的标【推测】）
- 滑门方向是 1 bit 的"排除法"，不是从第一次失败推出来的；adaptive @2 因此是 85% 而不是 100%。若想四类都能一次辨认，可以把滑轨做成略斜（滑开时稍往外走），让正拉也能带出一点侧向位移——未实现、未测。
- "只能拉一下"是人为规则：真人会边拉边感受门往哪边走。允许多次试探的话，agent 在第一次尝试内就能摸清机构，任务就变成 within 型【推测】。
- 门朝外、机器人从柜顶上伸过去抓，是为了让前置相机看得见门；对 RGB-only 的 agent，估旋钮高度（7.7 cm）和位置可能是主要难点之一【推测】。
- 铰链门的最优方向依赖拉的距离（α 60°/24 cm 能到 86–98°，α 70° 以上不稳）；F2 只给旋钮位移，agent 需要自己把"往前 5.5、往左 4.1"翻译成拉的方向【推测：这是该任务对 VLM 的主要推理负担】。
