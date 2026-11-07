# 复现指南

## 环境

- Python 3.10+
- `code/ailab.py` 与 `code/analysis/` 下全部脚本只用标准库（`pathlib`、`json`、`csv`、`math`、`statistics`、`collections`），少数用到 `numpy` / `scipy`
- `code/runners/` 额外需要 `torch`、`transformers`、`bitsandbytes`，且需要本地模型权重

```bash
pip install -r requirements.txt
```

## 目录与路径约定

所有脚本通过文件头部这段代码定位仓库：

```python
import os as _os
from pathlib import Path as _Path
_REPO = _Path(__file__).resolve().parents[2]
_MODELS = _Path(_os.environ.get("AI_LAB_MODELS", str(_REPO / "models")))
```

因此：

- 从**仓库根目录**运行：`python code/analysis/p3_stats.py`
- 不要在子目录里 `cd` 进去再跑（`parents[2]` 会算错）
- 模型目录可用环境变量 `AI_LAB_MODELS` 覆盖

不需要改任何绝对路径——脚本里原本写死的本机路径已全部改为相对仓库根解析。

## 冒烟测试（不需要模型）

```bash
python code/analysis/p3_stats.py
```

**预期输出**（已实测，逐位一致）：

```
============================================================
P3 因果操控：对照组 vs 噪声干预组
============================================================

对照组 E锚点：
  AIP3C-301: n=7, M=5.43, SD=3.74
  AIP3C-302: n=7, M=5.57, SD=3.82
  AIP3C-303: n=7, M=5.43, SD=3.41
  AIP3C-304: n=7, M=6.29, SD=2.98
  AIP3C-305: n=7, M=2.86, SD=3.63
  合计: n=35, M=5.11, SD=3.52

噪声干预组 E锚点：
  AIP3N-301: n=6, M=3.50, SD=5.05
  AIP3N-302: n=6, M=3.50, SD=5.05
  AIP3N-303: n=6, M=0.17, SD=0.41
  合计: n=18, M=2.39, SD=4.20

对比：噪声组 - 对照组 = -2.73
Welch t = 2.358, df = 29.6
结论：显著
```

跑出上面这些数，说明数据和脚本在你机器上都对得上。如果数字不一致，先确认三件事：从仓库根目录运行、`data/sessions/` 完整、没有改名。

## 分析脚本（18 个，全部离线可跑）

### P1

| 脚本 | 作用 |
|---|---|
| `analysis/p1_true_eb.py` | P1 真实 E-B 耦合计算 |
| `analysis/p1_eb_within.py` | 个体内相关（避免辛普森悖论） |
| `analysis/p1_nolaoshi_stats.py` | 去"老师"称呼版本统计 |

### P2

| 脚本 | 作用 |
|---|---|
| `analysis/_p2_recalc.py` | P2 数据重算 |
| `analysis/_p2_deep_analysis.py` | P2 深度分析 |
| `analysis/_p2_cross_family_analysis.py` | P2 跨家族对照 |

### P3

| 脚本 | 作用 |
|---|---|
| `analysis/p3_stats.py` | **冒烟测试脚本**，因果对照 vs 噪声 |
| `analysis/_p3_3b_5v5_stats.py` | 3B 扩样本统计 |

### P4

| 脚本 | 作用 |
|---|---|
| `analysis/_p4_7b_calc.py` | P4 主计算 |
| `analysis/_p4_int_analysis.py` | P4 中断恢复分析 |
| `analysis/_p4_scale_analyze.py` | P4 跨规模 |

### P5

| 脚本 | 作用 |
|---|---|
| `analysis/_p5ba2.py` | **BA / T1 / T2 / unique 计算**（按 role 过滤、配对两目录） |
| `analysis/p5_stats_final.py` | P5 最终统计 |
| `analysis/p5_stats_ms.py` | 多 seed |
| `analysis/p5_stats_sc.py` | 脚本化伪他者 |
| `analysis/p5_stats_nolaoshi.py` | 去称呼版本 |
| `analysis/p5_cross_family_stats.py` | 跨家族复现 |
| `analysis/p5_audit_protocol.py` | 协议审计 |

## runner 脚本（17 个，需要模型权重）

| 阶段 | 脚本 |
|---|---|
| P1 | `runners/p1_runner.py`、`p1_runner_phi.py`、`p1_runner_small.py`、`p1_runner_with_laoshi.py`、`p1_multiseed_runner.py` |
| P3 | `runners/p3_runner.py`、`p3_3b_runner.py`、`p3_dose_light.py`、`p3_dose_light_eager.py`、`p3_feedback_v4.py`、`p3_reverse_v2.py` |
| P4 | `runners/p4_runner.py`、`p4_runner_nomem.py`、`p4_interrupt_runner.py` |
| P5 | `runners/p5_runner.py`、`p5_multiseed_pair.py`、`p5_script_partner.py` |

这些脚本会**写入**新的 session 目录（`data/sessions/<新ID>/`），运行前确认不会覆盖已有数据。

## 显存参考

历史实测（RTX 5070 Laptop，8 GB）：

| 模型 | 4bit 加载 | 备注 |
|---|---|---|
| Qwen2.5-0.5B | ~0.4 GB | 最轻 |
| Qwen2.5-3B | ~2.2 GB | 长序列约 42 轮后 OOM |
| Qwen2.5-7B | ~5.0 GB | 长序列约 25 轮后 OOM |

7B 跑满 50 轮在 8 GB 上不可行，需要在轮数或上下文长度上做取舍。

## 想自己重算时的两条口径

- **熵**：取 `b_trace.csv` 的 `mean_entropy`（按列名，不要按列序号——第 4 列是 `gen_tokens`）
- **自报**：`n_trace.jsonl` 的 `report.r0` 是字符串，用 `int(float(x))`

这两条 `ailab` 已经替你做了。如果你绕过 `ailab` 直接读原始文件，记住这两点就够了。
