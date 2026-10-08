# LFF-Bench（Learn-From-Failure Bench）

测 VLM 智能体（GPT-6 Astra、Claude 等）控制机械臂时**能不能从失败里学到东西**的基准。每个任务藏一个第一次几乎必然做错的隐变量
（标定偏差、摩擦、重心、几何、机关……，L1–L7），智能体在 k 次尝试里靠失败反馈修正。

- 仿真：LIBERO（robosuite 1.4.1 + MuJoCo 3.2.3），纯 CPU + OSMesa 软渲染即可跑。
- **没有数据集**：任务、物体（MJCF）、场景（BDDL）全部由代码生成（`lffbench/tasks/`、`lffbench/objects.py`、`lffbench/bddl.py`）；
  机械臂、桌子和部分物体网格来自 LIBERO，安装时从上游下载。
- 每个任务的说明在 `task_cards/<task>.md`，headroom 验证结果在 `results/<task>/validate.json`。
- 开发者指南（加任务的规则）：`AGENT_GUIDE.md`；智能体 harness 细节：`lffbench/agent/README.md`。

## 1. 环境要求

- Linux x86_64（智能体的隔离 shell 依赖 Linux user namespace，macOS 跑不了 LA 接口）。
- `git`、[`uv`](https://docs.astral.sh/uv/)（`curl -LsSf https://astral.sh/uv/install.sh | sh`；没有 Python 3.10 时 uv 会自动下载）。
- CPU 渲染用 OSMesa：Ubuntu/Debian `sudo apt install libosmesa6`。
- root 权限（只用于把隔离 shell 装到 `/opt/lffjail`）。
- 磁盘约 2.5 GB（venv 约 1.4 GB，LIBERO 资产约 640 MB）；RPent / GPT-Policy 另需约 0.6 GB。

## 2. 安装（新机器从零开始）

```bash
git clone git@github.com:Peter-cuhk/lffbench.git && cd lffbench
bash setup/install.sh            # .venv（Python 3.10）+ third_party/LIBERO（固定 commit）+ .libero/config.yaml
sudo bash jail/setup_jail.sh     # 智能体的隔离 shell：/opt/lffjail（无网络、看不到 /home /mnt /tmp，带 numpy/scipy/Pillow/OpenCV）
cp .env.example .env             # 填 OPENAI_API_KEY；走中转 / 网关时再填 OPENAI_BASE_URL
source run_env.sh                # 以后每个新 shell 都要先 source
python scripts/check_setup.py --api   # 检查包、仿真+渲染、隔离 shell，并发一条极短的 API 请求
```

`check_setup.py` 全部 `[ok]` 就可以跑了。常见问题：

| 现象 | 处理 |
|---|---|
| `jail runs` 失败，提示 `uid_map` / `Operation not permitted` | Ubuntu ≥ 23.10 默认禁止非特权 user namespace：`sudo sysctl -w kernel.apparmor_restrict_unprivileged_userns=0`（永久生效写进 `/etc/sysctl.d/`） |
| 渲染报 OSMesa / `libOSMesa.so` 找不到 | `sudo apt install libosmesa6` |
| 有 NVIDIA GPU，想用 EGL | `export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl` 后再 `source run_env.sh`（本仓库只在 OSMesa 上验证过） |
| API 报证书错误 | 在 `.env` 里加 `SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt` |
| 中转站返回 Cloudflare 1010 | 用仓库里的驱动（走 openai SDK，带 SDK 的 User-Agent），不要自己用 urllib 直连 |

## 3. 运行

所有命令都在仓库根目录、`source run_env.sh` 之后执行。建议 `export OMP_NUM_THREADS=1`（每个仿真进程单线程，多个 run 并行时不互相抢核）。

### 3.1 列出任务 / 不调模型的验证
```bash
python -c "import pkgutil,importlib,lffbench.tasks as t; from lffbench.task_base import TASKS; [importlib.import_module('lffbench.tasks.'+m.name) for m in pkgutil.iter_modules(t.__path__)]; print('\n'.join(f'{k:26s} {v.level:3s} {v.category}' for k,v in sorted(TASKS.items())))"
python scripts/validate.py --task l5_slide_to_target --n 3 --k 5   # 脚本策略（oracle/naive/adaptive/blind）跑 headroom
```

### 3.2 GPT 当智能体（LA 接口，推荐）
LA 接口（仿 LIBERO-Agent）：每个 episode 一个工作目录，智能体有三个工具——`robot`（机器人命令）、`shell`（在隔离环境里跑命令，
可以自己写 Python 处理图像）、`view_images`（看工作目录里的图）。每个 (任务, seed) 起一个机器人服务器 + 一个 GPT 驱动进程。
```bash
python scripts/gpt_batch.py start --batch demo --model gpt-6-astra --effort high --style la --perception rgb \
    --feedback F1 --k 3 --video --tasks l5_slide_to_target:3000 l7_offcenter_push:3000
python scripts/gpt_batch.py status --batch demo              # 每个 run 的命令数、尝试结果（S/F）、是否结束
python scripts/la_report.py --runs gpt --batch demo --out demo_table.md
python scripts/gpt_batch.py stop --batch demo                # 结束后关服务器
```
- 输出在 `runs/gpt/<batch>/<run>/`：`events.jsonl`（服务器日志：每条命令、每次尝试的成败和内部测量）、`gpt_transcript.jsonl`
  （每次 API 请求 / 回复，图像存为文件引用）、`rollout.mp4`（`--video`）、`server.log`、`gpt_player.log`。
- 常用开关：`--perception rgb|rgbd|depth`（是否给深度 + 相机标定）、`--feedback F0|F1|F2`（失败后给：无 / 只给成败 / 给测量误差）、
  `--k` 尝试次数、`--nominal`（L1 任务去掉隐藏偏差的对照）、`--port0` 起始端口（每个 run 占一个本地端口）。
- 每次尝试的工具调用上限 100（LA 接口）。GPT-6 每轮约 7 s；一个 run 通常几十到一两百次调用，先小批量试。
- 种子约定：1000–1999 开发（validate.py 用），3000 用于试点，5000+ 留作正式测试。

单个会话手动跑（调试用）：
```bash
LFF_STYLE=la scripts/start_robot_session.sh l5_slide_to_target 3000 full F1 9800 direct 3 manual rgb 1   # 打印 SANDBOX= 和 LOG=
set -a; source .env; set +a
python scripts/gpt_player.py --session <LOG>/session.json --model gpt-6-astra --effort high --style la
```

### 3.3 Claude Code 子代理当智能体
Claude 没有走 API 驱动，而是让 Claude Code 的子代理在沙箱目录里直接敲 `./robot` / `./shell`：
```bash
python scripts/claude_batch.py start --batch cla_demo --model sonnet --only harness --k 3 --perception rgb --style la \
    --video --port0 11000 --tasks l5_slide_to_target:3000
```
然后在 Claude Code 里为 `runs/claude/<batch>/manifest.json` 中每个 run 派一个子代理（prompt 用该 run 的 `prompt` 字段原文），
结束后 `claude_batch.py audit / stop`、`la_report.py --runs claude`。注意：Claude Code 开着自动记忆时子代理会被注入记忆索引，
正式评测前要在 settings 里关掉 `autoMemoryEnabled` 再新开会话；`audit` 会检查这一点（需 `CLAUDE_TASKS_DIR` 指向该会话的 tasks 目录）。

### 3.4 RPent / GPT-Policy harness（可选）
两个第三方 harness 以"上游固定 commit + 补丁"的形式提供（`integrations/`），装到 `third_party/`：
```bash
bash integrations/rpent/setup.sh        # Python 3.12 venv
bash integrations/gpt-policy/setup.sh   # Python 3.10 venv
python scripts/gpt_batch.py start --batch h_demo --harness rpent --style v2 --perception rgb --k 3 --video --tasks l1_bias_place:3000
```
（两者只支持 v1/v2 接口，不支持 LA。GPT-Policy 的 LICENSE 只允许评审用途，所以本仓库只带补丁、不带它的源码。）

### 3.5 测试
```bash
python -m pytest -q tests/test_agent_payload.py tests/test_agent_protocol.py   # 不跑仿真
python -m pytest -q tests/test_agent_sim.py                                    # 仿真端到端（几分钟）
```

## 4. 目录

| 路径 | 内容 |
|---|---|
| `lffbench/tasks/` | 每个任务一个文件：场景、隐变量采样、成功判定、F2 反馈、脚本参照策略 |
| `lffbench/{envs,objects,bddl,skills,task_base}.py` | LIBERO 环境封装、程序生成物体、BDDL、运动原语、任务基类 |
| `lffbench/agent/robot_server.py` | 机器人服务器：持有仿真和隐变量，给智能体一个沙箱目录（`./robot`、`./shell`、README、`obs/` 图像） |
| `lffbench/agent/` 其余 | 工具接口、提示词、OpenAI / Chat Completions / 脚本后端、指标 |
| `scripts/` | `gpt_batch.py` `gpt_player.py`（GPT）、`claude_batch.py`（Claude 子代理）、`validate.py`、`la_report.py`、`check_setup.py` 等 |
| `task_cards/` | 每个任务的说明（隐变量、为什么第一次会错、反馈示例、验证表） |
| `results/<task>/*.json` | headroom 验证结果（缩略图不进仓库，重跑 `validate.py` 生成） |
| `jail/` | 隔离 shell 的入口脚本和安装脚本 |
| `integrations/` | RPent、GPT-Policy 的补丁和安装脚本 |
| `setup/` | `install.sh`；`requirements.txt`（直接依赖）+ `requirements.lock.txt`（验证过的全部版本，安装时作约束） |

运行产物（`runs/`、`videos/`、`traces/`、`logs/`）和生成文件（`assets_gen/`）不进仓库。

## 5. 环境变量

`run_env.sh` 会先读仓库根目录下的 `env.local.sh`（不进仓库），机器相关的设置写在那里：

| 变量 | 默认 | 作用 |
|---|---|---|
| `LFF_VENV` | `.venv` | 主 venv |
| `LIBERO_CONFIG_PATH` | `.libero` | LIBERO 的 `config.yaml` 所在目录 |
| `LFF_EXTRA_PYTHONPATH` | `third_party/LIBERO` | LIBERO 源码目录（加到 `PYTHONPATH`） |
| `MUJOCO_GL` / `PYOPENGL_PLATFORM` | `osmesa` | 渲染后端 |
| `LFF_ENV_FILE` | `.env` | API 配置文件（`OPENAI_API_KEY` / `OPENAI_BASE_URL` / `SSL_CERT_FILE`） |
| `LFF_SANDBOX_ROOT` | `/tmp/robotlab` | 智能体沙箱目录的父目录 |
| `LFF_RUNS_DIR` | `runs/claude` | `start_robot_session.sh` 的日志目录（batch 脚本会自己设） |
| `CLAUDE_TASKS_DIR` | 空 | Claude Code 会话的 tasks 目录（`claude_batch.py audit` 读子代理 transcript） |
| `RPENT_DIR` / `RPENT_VENV` / `GPTPOLICY_VENV` | `third_party/...` | 第三方 harness 位置 |
| `LFF_AUDIT_PRIVATE` | 空 | RPent / GPT-Policy 审计脚本额外检查的私密字符串（逗号分隔的正则，如你的用户名） |
| `LFF_OPENAI_SDK_DIR` | 空 | 可选：放 `openai` 包的 `pip --target` 目录（venv 里没装 openai 时用） |
| `QWEN_VL_MODEL` / `QWEN_OVERLAY` | HF 上的 Qwen3-VL-8B | 本地 Qwen3-VL 服务（`scripts/qwen_vl_server.*`，原为 PPU 机器写，可选） |
