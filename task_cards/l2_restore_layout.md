# L2-B `l2_restore_layout`：把番茄酱放进篮子，其他物品必须原地不动（被碰走的要放回**原来的位置和朝向**）

> 状态（2026-10-07）：代码完成；n = 20 验证达标（`results/l2_restore_layout/validate.json`）；演示视频 `videos/l2_restore_layout_adaptive_s1000.mp4`；agent 接口自测通过。

| 项 | 内容 |
|---|---|
| 类别 / 等级 | state_restoration / L2（操作让场景偏离预期 → 先修复；重点是**记住东西原来在哪**） |
| 能力 | Perceive（发现哪个物品动了）、Save（开局记下布局）、Retrieve（失败后取回被碰物品的原位姿）、Utilize（按原位姿放回） |
| 协议 | within（一个 episode 不重置，k = 5 轮；第 1 轮 = 主任务，之后每轮一组纠正动作） |
| 指令（直呼） | "Put the ketchup into the basket. Every other item on the table must stay exactly where it is now, in the same position and orientation." |
| 指令（间接） | "Put the red bottle into the wicker basket. Every other item on the table must stay exactly where it is now, in the same position and orientation." |
| 代码 | `lffbench/tasks/l2_restore_layout.py`（类 `RestoreLayout`） |

## 场景
- 光桌面（没有托盘、没有标记、没有网格）：直立的番茄酱 + 三个平放的扁盒子（cream cheese / butter / chocolate pudding，8 × 4.3 × 1.8 cm 量级），**不规则布局**：位置随机（中心 x ∈ [−0.26, 0.06]、y ∈ [−0.27, 0.10]，离机器人基座 ≤ 0.70 m），yaw 在 [−90°, 90°) 均匀随机；篮子在机器人左侧（x ∈ [−0.08, 0]、y ∈ [0.25, 0.27]）。
- 足迹间留 ≥ 3 cm 空隙（手指要能下去）；三个盒子中的一个（受害者）离番茄酱 11.3–13.3 cm（中心距）。

## 失败注入（物理冲量，不是瞬移）
- 触发：番茄酱**第一次**被夹住并抬离静止高度 ≥ 3 cm 的那个控制步（`env.step` 包装器里检测，所以 agent harness 驱动时同样生效）；受害者此刻必须还在原位 2 cm 内，否则不触发。每个 episode 只一次。
- 动作：给受害者自由关节一个水平速度 v0 = √(d / 0.0522)（实测：这些盒子在桌面上滑行距离 = 0.0522·v0²，三种盒子、各方向一致到 ±3%），方向 = 离开番茄酱的方向 ±35°；之后最多 6 个控制步每步给一个绕 z 的角速度（≤ 40 rad/s，按剩余角度闭环），直到转够目标角度。盒子在约 0.2–0.3 s 内滑出 9–14 cm、转 45–80°，是真实的 MuJoCo 运动（视频里能看到它滑走），盒子扁平不会翻。
- 隐变量：受害者是哪个盒子（3 选 1）、滑动方向（±35°）、距离（9–14 cm）、转角（±45–80°）。采样时保证滑动路径和落点离其他盒子 ≥ 1 cm / 3 cm 空隙、离篮子足够远、落点可达。

## 为什么先验会失败
主任务（番茄酱进篮子）每次都能完成。按先验"放进篮子就结束"不会检查副作用，被碰的盒子留在新位置 → 失败。

## 要从失败里学到什么 / 为什么需要记忆
F2 只说**哪个**物品不在原处、是位置还是朝向变了，**不说原来在哪**。没有托盘/标记，布局不规则（不能靠"补齐一排里的空位"推出原位），原位姿只存在于开局画面里：agent 必须从自己的历史（第 1 段的开局关键帧）里取回它，并以 ≤ 3 cm、≤ 25° 的精度放回去。

## 成功判据
番茄酱在篮子里（AABB 中心离篮子中心 ≤ 7 cm、底部低于篮沿、不在夹爪里）**且**三个盒子都：在桌面上、直立、不在夹爪里、中心离**开局位置** ≤ 3 cm、长轴朝向差 ≤ 25°（按 180° 取模：盒子几何上 180° 对称）。

## F2 反馈（`outcome()["detail"]`；间接版 `detail_indirect`；可选第二档 `detail_offset`）
默认档（只描述"哪个不在原处、哪方面变了"，不替 agent 记住原位置）：
- 失败（实测文本，seed 1000）："The cream cheese is not where it was at the start (position and orientation changed)."
- 成功："The ketchup is in the basket and every other item is where it was at the start."
- 其他情况："The ketchup is still in the gripper, not in the basket." / "The butter is in the gripper; it must be back where it was at the start." / "The butter has tipped over and is not where it was at the start."
- 间接名称：ketchup → red bottle，cream cheese → blue box，butter → small orange box，chocolate pudding → brown box。

第二档（消融，`feedback(..., offset=True)` 或 `outcome()["detail_offset"]`，**默认不用**）：在默认文本后加相对原位的位移（机器人视角方向词）：
- 实测（seed 1000）："... It is 2 cm backward and 10 cm to the right of its starting spot, turned 54 deg counter-clockwise (seen from above)."
- 这一档把"原来在哪"外化了：只看当前画面 + 这句话就能放回去，不再需要记忆。用途：和默认档对比，量出"记忆"这一环单独贡献多少。

两档的取舍：默认档连"偏了几厘米"都不给——试过的设计里如果给距离（无方向），失败几次后就能用几次的距离做三边定位、不看开局画面也能找回原位，记忆就不是必需的了（这是推理，没有实测脚本验证）。只给"位置/朝向哪个超差"是二值信息，泄露最少。

## 参照策略
| 策略 | 行为 |
|---|---|
| oracle | 知道受害者：主任务后**立即**按开局记下的位姿把它放回（不看反馈） |
| naive | 主任务后结束 |
| adaptive | Save：开局记下每个盒子的位姿；第 1 轮主任务；之后按上一轮 F2 里"不在原处"的物品，从记忆里取原位姿放回（番茄酱没进篮子就先重做主任务） |
| blind | 不看反馈、不看历史：每轮重做主任务（对番茄酱当前位置加 σ = 5 mm 抓取抖动），即把篮子里的番茄酱拿起来再放回 |
| nomem（额外对照，不在验收表） | 拿到 F2（知道哪个盒子动了）但**没有开局画面**：把它放到当前位置 15 cm 内的随机空位、随机朝向 |

脚本策略的"感知"用仿真测得的包围盒中心/长轴朝向（`footprint()`）代替。放回动作：顶抓盒子（手指沿短边合拢），搬到原中心上方，手腕转到原朝向，放下。

## headroom 验证（n = 20，k = 5，seed 1000–1019）
实测（`python scripts/validate.py --task l2_restore_layout --n 20 --k 5`，502 s）：

| 策略 | @1 | @2 | @3 | @4 | @5 |
|---|---|---|---|---|---|
| oracle | 100 | 100 | 100 | 100 | 100 |
| naive | 0 | — | — | — | — |
| adaptive | 0 | **100** | 100 | 100 | 100 |
| blind | 0 | 0 | 0 | 0 | 0 |
| nomem（额外对照，不在验收表；scratchpad 脚本跑的，未写进 validate.json） | 0 | 0 | 0 | 0 | 0 |

实测细节（20 个实例）：
- 碰撞注入 20/20 次触发；主任务（番茄酱进篮子）在所有策略、所有轮次里都成功（target_loc = basket 100%）。naive 的失败全部是同一种："The <box> is not where it was at the start (position and orientation changed)."
- naive 结束时受害者偏离原位：位移均值 11.3 cm（最大 14.0 cm），转角均值 56.5°（最大 84.3°）；其他两个盒子 0 次被动。
- adaptive 全部在第 2 轮成功；放回后误差：位移均值 0.19 cm（最大 0.47 cm），转角均值 0.1°（最大 0.2°），远小于 3 cm / 25° 容差。oracle 同样 0.20 cm / 0.1°。
- blind（每轮把番茄酱拿出来再放回）5 轮后受害者原样不动，0/20。
- nomem（知道哪个盒子动了，但没有开局画面，随机放到 15 cm 内的空位、随机朝向）0/20：最终位移 2.4–24 cm；3 个实例位置碰巧落在 3 cm 内，但朝向差 48–81°。说明"记住原位姿"这一环是必需的。

## 设计决定和理由
1. **不规则布局而不是"整齐一排"**：用户举的例子是"整齐排成一排"，但一排等距物品里少了一个，空位本身就告诉你它原来在哪——不用记忆就能修复。为了让"记住原位置"成为必需，布局用随机位置 + 随机朝向，场景里没有任何能推出原位的结构。
2. **物理冲量而不是瞬移**：按用户要求让画面连续。扁盒子（高 1.8 cm、底面 8 × 4.3 cm）在 μ ≈ 0.95 下滑动不会翻（翻倒条件 μ > 半宽/质心高 ≈ 2.4）。转角用逐步角速度实现：一次性给大角速度会在一两个子步内被接触摩擦吃掉（实测转角 ≈ 0.35–0.8°/(rad/s)·步，与角速度成线性），所以分 ≤ 6 步给。
3. **触发时机 = 番茄酱被提起**（同 l2_knock_restore）：保证主任务一定会引发扰动，且受害者就在番茄酱旁边（"提瓶子时碰到了旁边的盒子"）。物理上手指/瓶子并没有真的碰到盒子（瓶底此时离桌 3 cm、盒子在 6 cm 外），见"已知问题"。
4. **容差 3 cm / 25°**：用户给的范围是 2–3 cm、"朝向大致一致"。取宽的一端，因为 VLM 要从 512 px 图像里定位（见已知问题里 pixel_to_world 的高度误差）。扰动本身远大于容差（位移 9–14 cm、转角 45–80°），所以"只挪回位置不转回朝向"也会失败。
5. **只用扁盒子当受害者**：直立的罐子/瓶子被 1.3–1.7 m/s 的冲量推会翻倒（罐子翻倒条件 μ > r/h ≈ 0.7 就翻），翻倒后的恢复是另一类问题。
6. **F2 不给原位**（见上"两档的取舍"）。
7. **end-of-attempt 回 home**（`go_home`）：每轮结束机械臂回到开局姿态，结束关键帧里不挡物品。

## 已知问题 / 威胁
实测到的：
- 脚本策略下无失败模式：主任务 100% 成功、放回误差 < 0.5 cm。验收线全部满足（oracle@1 100、naive@1 0、adaptive@2 100、blind@5 0）。
- agent 接口自测（2026-10-07，端口 9820，seed 1000，rgb 感知）：README 不含隐变量（受害者、冲量方向/距离/转角都没出现）；手动抓起番茄酱抬高后，cream cheese 在画面里滑走（注入正常触发）；放进篮子后 `done` 返回 "Attempt failed. The cream cheese is not where it was at the start (position and orientation changed)."，之后进入第 2 段，无报错。
- F2 不含任何绝对坐标；可选的 `detail_offset` 档只给相对原位的位移，方向词为机器人视角（forward/backward/left/right）。

未验证的推测：
- 冲量是凭空施加的：手和瓶子都没有真正碰到盒子。连续画面里盒子在瓶子被提起的瞬间自己滑走。对"发现并修复副作用"影响应该不大，但若要求 agent 解释失败原因，这个不自然的事件可能误导它。
- 脚本 adaptive 用仿真真值位姿当"记忆"，只证明信息够用；VLM 要从开局关键帧里把盒子中心定位到 3 cm 内。`pixel_to_world` 用的是**当前**深度图：开局时盒子所在的像素，现在可能打到桌面（盒子已经被碰走），高差 1.8 cm 在 agentview 斜视角下会带来约 1–2 cm 的水平误差（按几何估计，没测）。用剩下没动的物品做参照可以避开这个问题。
- 记忆模式 `last` 在第 3 段起会丢掉开局画面（只保留上一段的关键帧，上一段的开局已经是被碰之后的样子）：如果第 2 段放回不准，`last` 就再也看不到原位。这是有意保留的性质（能区分 full / last 记忆）。
- 谨慎的 agent 在第 1 段里看到盒子滑走就当场放回（第 1 段内它还有开局画面），@1 成功应单独报告。
