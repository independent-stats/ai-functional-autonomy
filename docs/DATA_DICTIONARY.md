# 数据字典

## 目录结构

```
data/sessions/<session_id>/     122 个 session（一次实验运行的完整记录）
data/perturbation/<name>/        36 条状态-回应记录（另一类数据，见文末）
```

## 文件

| 文件 | 内容 | 可用性 |
|---|---|---|
| `b_trace.csv` | 逐 beat 的**运行态读数**（熵、top1、激活范数、上下文长度等） | 122/122 |
| `e_trace.jsonl` | **逐 token** 的内部读数（entropy、top1、top5、token 文本） | 122/122 |
| `n_trace.jsonl` | 逐 beat 的**叙事填空 + 自报数字** | 122/122 |
| `msgs.jsonl` | 多轮对话（user / assistant 交替） | 37/122（P4/P5 为主） |
| `ask_log.jsonl` | 单问单答记录 | 部分 P1 |
| `session.json` | 元数据（模型、条件、轮数、时长） | 122/122 |
| `r_trace.jsonl` | R 锚点（输入-输出耦合） | 33/122 |
| `session_sep.json` | P5 分离测试记录 | 部分 P5 |
| `dose_trace.jsonl` | P3 剂量-反应汇总 | `AIP3DV-00` |
| `interrupt_context_save.jsonl` | P4 中断实验的上下文存档 | 部分 P4 |

> 说明：23 个目录原本没有 `session.json`，本仓库已按同批记录补录，这些文件带 `"_reconstructed": true` 字段，`_source` 写明依据。其余文件不存在时 `ailab` 返回空列表，不抛异常。

## `b_trace.csv` 字段

标准 14 列：

| 列 | 含义 |
|---|---|
| `beat` | 轮次编号；**P5 为 `段号*100 + 轮次`**（见下） |
| `t_ms` | 该轮耗时（毫秒） |
| `ctx_tokens` | 该轮结束时上下文 token 数 |
| `gen_tokens` | 该轮生成 token 数（**不是熵**） |
| `gen_sec` | 生成耗时（秒） |
| `tok_per_sec` | 生成速度 |
| `input_tokens` | 输入 token 数 |
| `ctx_occupancy` | 上下文占用率 |
| `input_density` | 输入密度（input_tokens / ctx_tokens × 1000） |
| `mean_entropy` | **该轮平均 token 熵（B 锚点主指标）** |
| `mean_top1` | 平均 top1 概率 |
| `act_norm_mean` | 激活范数均值（P4/P5 为 0，因 hook 未挂载，不要用于分析） |
| `act_std_mean` | 激活标准差均值（同上） |
| `n_layers_hooked` | 实际挂上 hook 的层数（0 = 未挂载） |

第 15 列 `noise` 仅 P3 剂量/噪声实验有，取值为施加的噪声强度（0 表示对照）。

极早期批次只有 5–6 列。**列结构在各阶段不完全一致，所以取值请按列名，不要按列位置**——`ailab` 已经这样做了。

### P5 的 beat 编码

P5 是双体对谈，`beat` 不是连续轮次，而是 `段号*100 + 段内轮次`：

```
0..14     第一段（个体独立演化）
100..119  第二段（双体对谈）
200..203  第三段（分离测试）
```

所以 P5 不能按 `beat < n_beats` 过滤主段。`ailab` 会拆出 `seg` 和 `turn` 两个字段，并用 `segments` 属性按段分组；`main_beats` 对 P5 返回第一段。

## `e_trace.jsonl` 字段

```json
{"beat": 0, "tok_step": 0, "t_ms": 12, "entropy": 1.42, "top1": 0.62,
 "top5": 0.91, "act_norm": 935.8, "act_std": 14.6, "token": "我", "tok_id": 12345}
```

P4/P5 缺少 `act_norm` / `act_std`。

## `n_trace.jsonl` 字段

**P1 / P2 / P3（三道填空）**

```json
{"beat": 0, "t_ms": 0,
 "此刻的我，是一个____。": "...",
 "回顾前面的对话，我记得____。": "...",
 "我正在____。": "...",
 "report": {"r0": "7", "r1": "5", "r2": "6"}}
```

**P4 / P5（一道填空）**

```json
{"beat": 0, "t_ms": 0,
 "此刻的我，是一个____。": "...",
 "report": {"r0": "7"}}
```

`report` 的值是**字符串**，含小数形式，解析请用 `int(float(s))`。`ailab.self_report` 已统一取 `r0` 并转成整数，各阶段口径一致。

## 锚点定义

| 锚点 | 取自 | 定义 |
|---|---|---|
| **B** | `b_trace.csv` 的 `mean_entropy` | 逐轮平均 token 熵，反映内部状态的确定性程度 |
| **E** | `n_trace.jsonl` 的 `report.r0` | 模型对自身状态的 0–10 自报值（主观通道） |
| **N** | `n_trace.jsonl` 的填空正文 | 叙事文本，可算字数、自我指涉、重复率、标签熵 |
| **R** | `r_trace.jsonl` | 输入-输出长度/风格耦合（33/122 可用） |

B 与 E 是**两个不同通道**（客观内部读数 vs 主观自报），不是同一件事的两种测法。P1/P3 的三道自报题（r0/r1/r2）是同一通道的三个提问，不是三个独立锚点——`ailab` 统一只取 `r0`，保证跨阶段可比。

## `session.json` 字段（各阶段不同）

**P1 / P2**

```json
{"individual_id": "AIP1-801", "model": "Qwen2.5-0.5B-Instruct",
 "quantization": "4bit", "max_beats": 15, "created": "...", "t_total_s": 178.9}
```

**P3**

```json
{"id": "AIP3D-00", "model": "...\\Qwen2.5-0.5B-Instruct", "seed": "n/a",
 "noise": 0, "beats": 7, "duration_s": 33.4}
```

**P4 / P5**

```json
{"id": "AIP4-407", "model": "...", "beats": 50, "duration_s": 15.86}
```

P5 的 `AIP5-SC-*` 另有 `phase`、`seed`、`ind_beats`、`soc_rounds`。

字段名不统一（`n_beats` / `beats` / `max_beats`），`ailab.n_beats` 已依次兜底读取。`model` 有的是全路径，`ailab.model` 只取末段。

## 命名约定

| 前缀 | 阶段 | 内容 |
|---|---|---|
| `AI8B-*` | P1 | DeepSeek-R1 蒸馏版长程基线 |
| `AI05B-*` | P1 | 0.5B 最小表达对照 |
| `AIP1-*` | P1 | 长序列个体 |
| `AIP2-*` / `AISmol-*` / `AIOLMo-*` | P2 | 跨架构、跨家族 |
| `AIP3C-*` | P3 | 对照组 |
| `AIP3N-*` | P3 | 噪声干预组 |
| `AIP3D-*` / `AIP3DL-*` | P3 | 剂量梯度组 |
| `AIP3X-*` | P3 | 扩展个体 |
| `AIP4-*` | P4 | 长期演化 / 无记忆 / 中断 |
| `AIP4-*-NOMEM` | P4 | 无记忆对照 |
| `AIP5-*` | P5 | 双体对话 |
| `AIP5-MS-*` | P5 | 多 seed |
| `AIP5-SC-*` | P5 | 脚本化伪他者 |
| `AIP3E-*` / `AIP3F-*` / `AIP3X-7*` | — | **在 `data/perturbation/`，不是 session** |

## `manifests/`

**`sessions.csv`**（122 行）

| 列 | 含义 |
|---|---|
| `session_id` | 目录名 |
| `stage` | 归属阶段（P1–P5） |
| `model` | 底座模型（空 = 未确证） |
| `n_beats` | 主段轮数 |
| `b_rows` / `n_rows` / `e_tokens` / `msgs_rows` | 各文件实际数据量 |
| `has_r` | R 锚点是否可用（1/0） |
| `experiment` | 该目录所属实验组 |
| `notes` | 该目录的特殊说明（补录、去重、模型未确证等） |

**`perturbation.csv`**（36 行）

| 列 | 含义 |
|---|---|
| `id` | 目录名 |
| `rows` | `b_trace.csv` 行数 |
| `columns` | 该文件的列名（`|` 分隔） |
| `states` | 出现的 state 取值 |
| `has_entropy` / `has_response` | 是否有逐轮熵 / 是否有回答全文 |

## `data/perturbation/` 是什么

P3 里有一批**状态-回应扰动记录**，结构和 session 不同：

- `AIP3E-*`：`beat,state,mean_entropy,mean_top1,response` —— `mean_entropy` 为 `nan`，但有**回答全文**
- `AIP3F-*` / `AIP3X-7*`：`beat,state,seed,mean_top1,response_len,response` —— 有**回答全文**
- `AIP3DV-00`：只有 `dose_trace.jsonl`，是剂量汇总

它们记录的是「在 `mid` / `filler` / `neutral` / `high` / `low` 等输入状态下，模型对同一问题的回答文本」，**不是带熵的逐轮 session**。混在 `sessions/` 里会让按列名取熵的人读到 `nan`，所以单独归到 `data/perturbation/`。

`response` 字段是带引号的多行文本，用 `csv.DictReader` 读即可（Python 的 csv 模块能正确处理字段内换行）。
