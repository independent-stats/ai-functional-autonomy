# 模型清单

本仓库 122 个 session 来自 7 个底座模型。下表按 `manifests/sessions.csv` 的 `model` 列实测统计。

| 模型 | 参数量 | session 数 | 主要出现在 |
|---|---|---|---|
| `Qwen2.5-0.5B-Instruct` | 0.5B | 66 | P1 长序列、P3、P4、P5 |
| `Qwen2.5-3B-Instruct` | 3B | 19 | P3 跨规模、P4、P5 |
| `Qwen2.5-7B-Instruct` | 7B | 13 | P1 主样本、P3 长序列 |
| `DeepSeek-R1-0528-Qwen3-8B` | 8B | 12 | P1 跨家族（推理型） |
| `SmolLM2-1.7B-Instruct` | 1.7B | 5 | P2 跨架构、P5 跨家族 |
| `OLMo-2-0425-1B-Instruct` | 1B | 3 | P2 跨架构 |
| （`model` 写作 `master`） | — | 3 | P2 跨架构（`AIP2-PH1/2/3`） |
| （未确证） | — | 1 | `AIP4-708-50` |

## 需要知道的几条

### 1. `AIP4-407` 的底座是 0.5B

`AIP4-407` 及同批 `AIP4-404`–`408`，`session.json` 明确写的是 `Qwen2.5-0.5B-Instruct`。

P4 的长期演化结论是在 **0.5B** 上得到的，不要当作"7B 系个体"解读。

### 2. `AIP2-PH1/2/3` 的 `model` 字段是 `master`

这三个目录的 `session.json` 里，`model` 值就是字符串 `"master"`——原始运行脚本取模型路径末段（`.../snapshots/master`）造成的截断。

据目录名（`PH`）推断应为 `Phi-3.5-mini-instruct`，但**原始元数据里没有更直接的证据**，所以 `sessions.csv` 保留原值并在 `notes` 列标注。`ailab.model` 返回的也是 `master`。

### 3. 量化与精度不统一

| 标记 | 含义 |
|---|---|
| `"quantization": "4bit"` | 4-bit 量化加载 |
| `"quantization": "fp16"` | fp16 全精度 |

**同一模型的 session 可能混有 4bit 与 fp16**（例如 `AI05B-*` 的 0.5B 最小表达组是 fp16，其余 0.5B 多为 4bit）。

做跨个体比较时，按 `quantization` 分组再用 `mean_entropy`，否则量化差异会混进个体差异里——4bit 会明显改变熵的绝对值（低熵更低、高熵被压），跨精度直接比 `mean_entropy` 不成立。

### 4. 上下文窗口

| 模型 | 上下文长度 |
|---|---|
| Qwen2.5 系列（0.5B / 3B / 7B） | 32768 |
| SmolLM2-1.7B | 8192 |
| Phi-3.5-mini | 4096 |
| OLMo-2-1B | 4096 |
| DeepSeek-R1-0528-Qwen3-8B | 32768 |

`b_trace.csv` 的 `ctx_occupancy` 按各自模型的上下文长度算，**不同模型之间不可比**。

### 5. 模型权重不在本仓库

本仓库只含实验记录与分析脚本，**不含任何模型权重**。

`code/analysis/` 下的脚本纯离线，不需要模型。要跑 `code/runners/` 需要自行下载对应底座，并用环境变量 `AI_LAB_MODELS` 指向本地模型目录：

```powershell
$env:AI_LAB_MODELS = "D:\models"
```

```bash
export AI_LAB_MODELS=/path/to/models
```

脚本内按 `<模型名>/` 组织，例如 `Qwen2.5-7B-Instruct/`、`microsoft--Phi-3.5-mini-instruct/snapshots/master/`。
