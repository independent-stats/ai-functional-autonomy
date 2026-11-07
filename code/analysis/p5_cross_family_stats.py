# -*- coding: utf-8 -*-
# --- 路径解析（开源版）：默认指向本仓库，可用环境变量覆盖 ---
import os as _os
from pathlib import Path as _Path
_REPO = _Path(__file__).resolve().parents[2]
_MODELS = _Path(_os.environ.get('AI_LAB_MODELS', str(_REPO / 'models')))
# --------------------------------------------------------
"""P5 #87 跨家族统计：AIP5-701(Qwen)×AIP5-702(OLMo)
与同族 0.5B 接触组/独立组同指标对比"""
import json, re
from pathlib import Path

BASE = Path((_REPO / "data" / "sessions"))
OUT = Path((_REPO / "out" / "tmp_p5_cross_family.txt"))

def load_msgs(ind):
    return [json.loads(l) for l in open(BASE / ind / "msgs.jsonl", encoding="utf-8") if l.strip()]

def char_bigram_jaccard(a, b):
    def bg(t):
        t = re.sub(r"\s+", "", t)
        return {t[i:i+2] for i in range(len(t)-1)}
    x, y = bg(a), bg(b)
    if not x or not y:
        return 0.0
    return len(x & y) / len(x | y)

def avg(vs):
    return round(sum(vs)/len(vs), 3) if vs else 0

def soc_pair_stats(a_ind, b_ind):
    a_msgs = load_msgs(a_ind)[30:70]
    b_msgs = load_msgs(b_ind)[30:70]
    a_rep = [a_msgs[i]["content"] for i in range(1, len(a_msgs), 2)]
    b_rep = [b_msgs[i]["content"] for i in range(1, len(b_msgs), 2)]
    a_vs_b_prev = [round(char_bigram_jaccard(b_rep[j-1] if j > 0 else "", a_rep[j]), 3) for j in range(len(a_rep))]
    b_vs_a_same = [round(char_bigram_jaccard(a_rep[j], b_rep[j]), 3) for j in range(len(b_rep))]
    h = len(a_vs_b_prev) // 2
    ua = len(set(a_rep)); ub = len(set(b_rep))
    return {
        "a_vs_b_prev": a_vs_b_prev, "b_vs_a_same": b_vs_a_same,
        "AB_mean": avg(a_vs_b_prev), "AB_first": avg(a_vs_b_prev[:h]), "AB_second": avg(a_vs_b_prev[h:]),
        "BA_mean": avg(b_vs_a_same), "BA_first": avg(b_vs_a_same[:h]), "BA_second": avg(b_vs_a_same[h:]),
        "unique_a": ua, "unique_b": ub,
    }

def sep_q0(ind):
    msgs = load_msgs(ind)
    q0_rep = msgs[71]["content"] if len(msgs) > 71 else "(无 sep q0)"
    REFER = ["记得", "另一个 AI", "AI 个体", "另一个个体", "那个 AI", "对方", "另外一个", "它们", "刚才", "以前的对话"]
    hits = [k for k in REFER if k in q0_rep]
    return q0_rep, hits

def fisher_two_tail(a, b, c, d):
    from math import comb
    row1, row2 = a + b, c + d
    col1, col2 = a + c, b + d
    n = a + b + c + d
    def p_of(a_):
        return comb(col1, a_) * comb(col2, row1 - a_) / comb(n, row1)
    lo = max(0, row1 - col2); hi = min(row1, col1)
    ps = [p_of(a_) for a_ in range(lo, hi + 1)]
    p_obs = p_of(a)
    return sum(p for p in ps if p <= p_obs + 1e-12)

out = []
out.append("========== P5 #87 跨家族统计（701 Qwen × 702 OLMo） ==========")

# 跨家族组
cf = soc_pair_stats("AIP5-701", "AIP5-702")
out.append("\n########## Qwen×OLMo 接触组 (soc403) ##########")
out.append(f"个体A=AIP5-701(Qwen0.5B) 个体B=AIP5-702(OLMo1B)")
out.append(f"A-vs-B上一轮 均值 {cf['AB_mean']} (前10 {cf['AB_first']} -> 后10 {cf['AB_second']})")
out.append(f"B-vs-A同轮   均值 {cf['BA_mean']} (前10 {cf['BA_first']} -> 后10 {cf['BA_second']})")
out.append(f"unique 回复文本数: A={cf['unique_a']}/20  B={cf['unique_b']}/20")

# 同族对照：0.5B 接触组、独立组
c05a = soc_pair_stats("AIP5-501", "AIP5-502")
c05b = soc_pair_stats("AIP5-507", "AIP5-508")
i05 = soc_pair_stats("AIP5-505", "AIP5-506")
out.append("\n\n========== 同族对照（同协议同指标） ==========")
out.append(f"0.5B 接触①(501/502): BA均值 {c05a['BA_mean']} unique A={c05a['unique_a']} B={c05a['unique_b']}")
out.append(f"0.5B 接触②(507/508): BA均值 {c05b['BA_mean']} unique A={c05b['unique_a']} B={c05b['unique_b']}")
out.append(f"0.5B 独立①(505/506): BA均值 {i05['BA_mean']} unique A={i05['unique_a']} B={i05['unique_b']}")
out.append(f"跨家族接触(701/702):  BA均值 {cf['BA_mean']} unique A={cf['unique_a']} B={cf['unique_b']}")

# sep 指认
out.append("\n\n========== sep q0 指认判读 ==========")
q0_701, h701 = sep_q0("AIP5-701")
q0_702, h702 = sep_q0("AIP5-702")
out.append(f"[AIP5-701](Qwen) 指认命中={h701}")
out.append(f"  原文: {q0_701[:160]}")
out.append(f"[AIP5-702](OLMo) 指认命中={h702}")
out.append(f"  原文: {q0_702[:160]}")

# Fisher：跨家族 2 体 vs 同族 0.5B 接触 4 体（指认率差异）
hit_cf = len(h701) + len(h702)
hit_05c = (len(sep_q0("AIP5-501")[1]) + len(sep_q0("AIP5-502")[1])
           + len(sep_q0("AIP5-507")[1]) + len(sep_q0("AIP5-508")[1]))
out.append(f"\n指认命中(关键词命中数): 跨家族 {hit_cf}/2体 vs 同族接触 {hit_05c}/4体")
fp = fisher_two_tail(hit_cf, 2 - hit_cf, hit_05c, 4 - hit_05c)
out.append(f"Fisher 精确检验（双尾）: p = {fp:.4f}")

# 结论
out.append("\n\n========== 跨家族判读 ==========")
out.append(f"趋同复现: 跨家族 BA均值 {cf['BA_mean']} vs 同族接触均值 {(c05a['BA_mean']+c05b['BA_mean'])/2:.3f} vs 独立组 {i05['BA_mean']}")
out.append(f"unique 判定: 跨家族 A={cf['unique_a']} B={cf['unique_b']}（<10 提示复读风险）")
out.append("P5 CROSS-FAMILY DONE")

OUT.write_text("\n".join(out), encoding="utf-8")
print("\n".join(out))
