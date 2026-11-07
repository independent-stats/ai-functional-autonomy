# AI 个体实验记录（P1–P5）

**122 个原始 session · 36 条状态-回应记录 · 7 个底座模型 · 5 个实验阶段 · 配套分析代码**

这套记录对应一个具体问题：

> 如果一个 AI 没有连续上下文、没有持久记忆，每次都是从权重重新开始——
> 那么"同一个个体"这件事，究竟存在于哪里？

六个阶段的实证尝试，本仓库发布其中前五个阶段的原始记录与分析代码。

---

## 六十秒上手

不需要读任何说明，先跑起来：

```bash
pip install -r requirements.txt
python -c "import sys; sys.path.insert(0,'code'); import ailab; s=ailab.load('AIP4-407'); print(len(s.beats), s.entropy[0], s.entropy[-1])"
```

输出：

```
50 1.217329 2.347898
```

拿到这三个数，说明数据在你机器上是好的。然后：

```python
import ailab

# 读一个 session
s = ailab.load("AIP4-407")
s.entropy            # 逐轮熵（B 锚点），None 已剔除
s.self_report        # 0–10 自报值（E 锚点），已转成整数
s.narrative          # 逐轮叙事填空（N 锚点）
s.beats              # 逐轮完整记录（含 ctx_tokens / top1 / dup 标记）
s.meta               # session.json 内容

# 遍历
for s in ailab.iter_sessions(stage="P3"):
    print(s.id, s.model, len(s.entropy_clean()))

# 全部概览，可直接 pd.DataFrame
ailab.catalog()
```

`ailab` 是**唯一需要记的入口**。它已经把各阶段的数据差异全部处理好了（见下节），你直接拿干净数据做统计即可。

## 仓库结构

```
data/sessions/        122 个 session 目录（原始记录）
data/perturbation/     36 条状态-回应记录（另一类数据，不是 session）
code/ailab.py          统一加载器 —— 先读这个
code/analysis/         18 个分析脚本（纯离线，不需要模型）
code/runners/          17 个实验运行器（需要自行下载模型）
docs/DATA_DICTIONARY.md 字段定义与锚点口径
docs/MODELS.md          模型清单与量化设置
docs/REPRODUCE.md       复现指南与预期输出
manifests/sessions.csv  122 行逐目录索引
manifests/perturbation.csv 36 行扰动记录索引
examples/load_one_session.py 命令行示例
```

## 数据规模

| 阶段 | 主题 | session 数 |
|---|---|---|
| P1 | 自报通道与 B–E 耦合 | 21 |
| P2 | 跨架构 / 跨家族复现 | 10 |
| P3 | 因果操控与剂量反应 | 46 |
| P4 | 长期演化（50 轮）与中断恢复 | 15 |
| P5 | 双体对话与社会耦合 | 30 |
| **合计** | | **122** |

另有 `data/perturbation/` 的 36 条状态-回应记录（P3 的 `mid` / `filler` / `neutral` / `high` / `low` 状态扰动），它们是**回答文本记录**，不含逐轮熵，结构与 session 不同。用 `ailab.load_perturbation()` 读。

## 四个锚点

| 锚点 | 通道 | 字段 | 可用性 |
|---|---|---|---|
| **B** | 客观内部读数 | `b_trace.csv` 的 `mean_entropy` | 122/122 |
| **E** | 主观自报 | `n_trace.jsonl` 的 `report.r0` | 122/122 |
| **N** | 叙事文本 | `n_trace.jsonl` 的填空正文 | 122/122 |
| **R** | 输入-输出耦合 | `r_trace.jsonl` | 33/122 |

B 与 E 是**两个不同通道**，不是同一件事的两种测法。

## 加载器替你处理掉的事

各阶段采集时的写法不完全一致，这些差异都已在 `ailab` 内部消化，你不需要知道细节：

- `b_trace.csv` 有 4 种列结构（5 / 6 / 14 / 15 列）→ 一律按**列名**取值，不依赖列序
- `b_trace.csv` 第 4 列是 `gen_tokens`，**不是熵** → 熵只从 `mean_entropy` 取
- `n_trace.jsonl` 的填空 P1/P3 是三道、P4/P5 是一道 → 全部收进 `narrative`
- `report` 的值是字符串、可能带小数 → 统一 `int(float(x))`
- `mean_entropy` 可能是 `nan` → 转成 `None`，不进统计
- `b_trace` 行数可能多于 `n_beats` → `main_beats` / `main_entropy` 只取主段
- **P5 的 beat 是 `段号*100 + 轮次`**（如 `111`）→ `ailab` 自动拆成 `seg` / `turn`，`segments` 按段分组
- 断点续跑造成的**重复行**（5 个目录）→ 已按 `(段号, 轮次)` 去重，保留续跑后的记录
- 缺 `session.json` 的 23 个目录 → 已补录，字段带 `_reconstructed` 标记
- 空 `r_trace.jsonl`（75 个）→ 已清理
- `act_norm_mean` 在 P4/P5 恒为 0 → 这是 hook 未挂载，不是"激活为零"，该字段不要用于分析

## 环境

- Python 3.10+
- `requirements.txt` 已逐条注明用途；**分析脚本全部为纯标准库 + numpy/pandas/scipy，不需要 GPU**
- `code/runners/` 需要自行下载对应底座模型（见 `docs/MODELS.md`），不在仓库内分发权重
- 脚本内所有路径相对于仓库根自动解析，不需要改任何绝对路径

## 授权与引用

- 代码：MIT，见 [`LICENSE`](LICENSE)
- 数据：CC BY 4.0，见 [`LICENSE-DATA`](LICENSE-DATA)

如果你用这批数据做了分析并发现**这里写错了**，欢迎指出——这份档案的价值在于它记录得准。
