# 使用说明

这一页只写**必须知道的字段口径**。其他可以在代码里处理掉的差异，`ailab` 已经处理完了，不必再看。

## 已经处理好的，你不用管

下面这些各阶段写法不一致的地方，`ailab` 内部已经统一，直接取属性即可：

`b_trace.csv` 有 4 种列结构 · `gen_tokens` 不是熵 · 填空题 P1/P3 三道而 P4/P5 一道 · `report` 值是字符串 · `mean_entropy` 可能是 `nan` · `b_trace` 行数多于 `n_beats` · P5 的 `beat` 是 `段号*100+轮次` · 断点续跑留下的重复行 · 缺 `session.json` 的目录。

## 需要知道的七件事

### 1. `act_norm_mean` 在 P4 / P5 恒为 0

这是 **hook 没有挂上**（`n_layers_hooked = 0`），不是激活值真的为零。

这两条阶段可用的内部读数只有 `mean_entropy`。`act_norm_mean` / `act_std_mean` 不要进任何分析。

### 2. R 锚点只有 33/122

`r_trace.jsonl` 在多数目录里从未被真正采集。B / E / N 三锚点完整，**R 不完整**。

做跨阶段汇总时请把 R 单独处理，不要当成和其他三个等价。

### 3. 零熵不等于塌缩

贪婪解码下 token 熵会自然趋近 0.0X，这是**解码策略的必然结果**。

判断模型是否塌缩，要看**输出文本是否逐字重复**，而不是看熵值低。

### 4. P5 的社会耦合指标必须按 `role` 过滤

`msgs.jsonl` 是严格交替的（偶数行 user、奇数行 assistant）。按行号步长取样会整体落到提问方，指标有值但语义全错。

正确口径：按 `role == "assistant"` 过滤，取 `msgs[30:70]` 区间，去掉空白后算字符 bigram Jaccard。

**且 BA 必须配对两个目录**（个体 A 与个体 B 各自的目录），单目录自比会低 2–3 倍。

### 5. P1 主样本的对话不在 `msgs.jsonl`

`AI8B-*` 等 P1 主样本没有 `msgs.jsonl`，对话内容在 `ask_log.jsonl`，格式不同。

### 6. `AIP1-801 / 802 / 803 / 807` 没有 E 数据

这几个目录的 `n_trace.jsonl` 存在，但 `report` 缺失，E 锚点算不出来。做 P1 个体内相关时会把样本数拉低。

`ailab.self_report` 返回 `None`，需要时自行 `[x for x in ... if x is not None]`。

### 7. `AIP4-407` 的底座是 0.5B

```json
{"id": "AIP4-407", "model": "...\\Qwen2.5-0.5B-Instruct", "beats": 50}
```

不是 7B。同批 `AIP4-404`–`408` 同为 0.5B。

## 关于 `data/perturbation/`

那 36 条是**状态-回应记录**（`mid` / `filler` / `neutral` / `high` / `low` 等输入状态下模型对同一问题的回答全文），不含逐轮熵，结构和 session 不同。用 `ailab.load_perturbation()` 读，不要当 session 处理。

其中 `AIP3DV-00` 只有 `dose_trace.jsonl`，是剂量汇总表。
