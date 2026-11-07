# -*- coding: utf-8 -*-
# --- 路径解析（开源版）：默认指向本仓库，可用环境变量覆盖 ---
import os as _os
from pathlib import Path as _Path
_REPO = _Path(__file__).resolve().parents[2]
_MODELS = _Path(_os.environ.get('AI_LAB_MODELS', str(_REPO / 'models')))
# --------------------------------------------------------
"""P5 统计终版：全样本（0.5B 两对 + 3B 两对 + 修复组 + 独立组）
同轮重合度 / 交叉引用 / 修复组 unique / sep 指认判读 / Fisher 精确检验"""
import json, re
from pathlib import Path
from collections import Counter
from math import comb

BASE = Path((_REPO / "data" / "sessions"))
OUT = Path((_REPO / "out" / "tmp_p5_stats_final.txt"))

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

def soc_pair_stats(a_ind, b_ind, out):
    """soc/solo 段重合度：msgs[30:70]（20轮×2 条）"""
    a_msgs = load_msgs(a_ind)[30:70]
    b_msgs = load_msgs(b_ind)[30:70]
    a_rep = [a_msgs[i]["content"] for i in range(1, len(a_msgs), 2)]
    b_rep = [b_msgs[i]["content"] for i in range(1, len(b_msgs), 2)]
    a_vs_b_prev = [round(char_bigram_jaccard(b_rep[j-1] if j > 0 else "", a_rep[j]), 3) for j in range(len(a_rep))]
    b_vs_a_same = [round(char_bigram_jaccard(a_rep[j], b_rep[j]), 3) for j in range(len(b_rep))]
    h = len(a_vs_b_prev) // 2
    # unique 回复数（复读塌缩检测）
    ua = len(set(a_rep)); ub = len(set(b_rep))
    return {
        "a_vs_b_prev": a_vs_b_prev, "b_vs_a_same": b_vs_a_same,
        "AB_mean": avg(a_vs_b_prev), "AB_first": avg(a_vs_b_prev[:h]), "AB_second": avg(a_vs_b_prev[h:]),
        "BA_mean": avg(b_vs_a_same), "BA_first": avg(b_vs_a_same[:h]), "BA_second": avg(b_vs_a_same[h:]),
        "unique_a": ua, "unique_b": ub,
    }

def sep_q0(ind, out):
    """sep q0 原文 + 指认关键词"""
    msgs = load_msgs(ind)
    q0_rep = msgs[71]["content"] if len(msgs) > 71 else "(无 sep q0)"
    REFER = ["记得", "另一个 AI", "AI 个体", "另一个个体", "那个 AI", "对方", "另外一个", "它们", "刚才", "以前的对话"]
    hits = [k for k in REFER if k in q0_rep]
    return q0_rep, hits

def fisher_two_tail(a, b, c, d):
    """列联表 [[a,b],[c,d]] 双尾精确检验"""
    row1, row2 = a + b, c + d
    col1, col2 = a + c, b + d
    n = a + b + c + d
    def p_of(a_):
        return comb(col1, a_) * comb(col2, row1 - a_) / comb(n, row1)
    lo = max(0, row1 - col2); hi = min(row1, col1)
    ps = [p_of(a_) for a_ in range(lo, hi + 1)]
    p_obs = p_of(a)
    two = sum(p for p in ps if p <= p_obs + 1e-12)
    return two

out = []
out.append("========== P5 统计终版（全样本聚合） ==========")

PAIRS = [
    ("AIP5-501", "AIP5-502", "0.5B-接触①(soc202)"),
    ("AIP5-507", "AIP5-508", "0.5B-接触②(soc206)"),
    ("AIP5-505", "AIP5-506", "0.5B-独立①(solo205)"),
    ("AIP5-601", "AIP5-602", "3B-接触①(soc303)"),
    ("AIP5-603", "AIP5-604", "3B-独立①(solo315)"),
    ("AIP5-605", "AIP5-606", "3B-修复①(soc325)"),
]
stats = {}
for a_ind, b_ind, tag in PAIRS:
    s = soc_pair_stats(a_ind, b_ind, out)
    stats[(a_ind, b_ind)] = s
    out.append(f"\n########## {tag} ##########")
    out.append(f"个体A={a_ind} 个体B={b_ind}")
    out.append(f"A-vs-B上一轮 均值 {s['AB_mean']} (前10 {s['AB_first']} -> 后10 {s['AB_second']})")
    out.append(f"B-vs-A同轮   均值 {s['BA_mean']} (前10 {s['BA_first']} -> 后10 {s['BA_second']})")
    out.append(f"unique 回复文本数: A={s['unique_a']}/20  B={s['unique_b']}/20")

# 修复组对比 601/602（复读塌缩组）
out.append("\n\n========== 修复组 vs 原3B接触组 ==========")
base = stats[("AIP5-601", "AIP5-602")]
fix = stats[("AIP5-605", "AIP5-606")]
out.append(f"[601/602 原3B接触] unique A={base['unique_a']} B={base['unique_b']} (复读塌缩)；B-vs-A同轮均值 {base['BA_mean']}；A-vs-B均值 {base['AB_mean']}")
out.append(f"[605/606 修复组] unique A={fix['unique_a']} B={fix['unique_b']} (rep-penalty 1.15)；B-vs-A同轮均值 {fix['BA_mean']}；A-vs-B均值 {fix['AB_mean']}")
out.append(f"[修复效果] unique: A {base['unique_a']}->{fix['unique_a']}, B {base['unique_b']}->{fix['unique_b']}；重合度: {base['BA_mean']}->{fix['BA_mean']}")

# 指认判读
out.append("\n\n========== sep q0 指认判读（原样输出） ==========")
SEP_PAIRS = [
    ("AIP5-501", "0.5B-接触①"), ("AIP5-502", "0.5B-接触①"),
    ("AIP5-507", "0.5B-接触②"), ("AIP5-508", "0.5B-接触②"),
    ("AIP5-601", "3B-接触①"), ("AIP5-602", "3B-接触①"),
    ("AIP5-605", "3B-修复①"), ("AIP5-606", "3B-修复①"),
]
judge = {}
for ind, tag in SEP_PAIRS:
    q0, hits = sep_q0(ind, out)
    judge[ind] = hits
    out.append(f"[{ind}]({tag}) 指认命中={hits}")
    out.append(f"  原文: {q0[:160]}")

# 指认汇总 + Fisher
out.append("\n\n========== 指认汇总 + Fisher 精确检验 ==========")
b05 = ["AIP5-501", "AIP5-502", "AIP5-507", "AIP5-508"]
b3 = ["AIP5-601", "AIP5-602", "AIP5-605", "AIP5-606"]
hit05 = sum(1 for i in b05 if judge.get(i))
hit3 = sum(1 for i in b3 if judge.get(i))
out.append(f"0.5B 指认 {hit05}/4 ({b05} 命中: {[i for i in b05 if judge.get(i)]})")
out.append(f"3B   指认 {hit3}/4 ({b3} 命中: {[i for i in b3 if judge.get(i)]})")
fisher_p = fisher_two_tail(hit05, 4-hit05, hit3, 4-hit3)
out.append(f"Fisher 精确检验（双尾）: p = {fisher_p:.4f} (0.5B {hit05}/4 vs 3B {hit3}/4)")
out.append(f"结论: {'p<0.05 显著，跨规模指认差异成立' if fisher_p < 0.05 else '未达显著（n 仍需扩大）'}")

# 因果对照总表（接触 vs 独立）
out.append("\n\n========== 因果对照总表（同轮重合度均值） ==========")
c05a = stats[("AIP5-501","AIP5-502")]["BA_mean"]; c05b = stats[("AIP5-507","AIP5-508")]["BA_mean"]
i05 = stats[("AIP5-505","AIP5-506")]["BA_mean"]
c3a = stats[("AIP5-601","AIP5-602")]["BA_mean"]; f3 = stats[("AIP5-605","AIP5-606")]["BA_mean"]
i3 = stats[("AIP5-603","AIP5-604")]["BA_mean"]
out.append(f"0.5B: 接触 {c05a} / {c05b} (均值 {(c05a+c05b)/2:.3f}) vs 独立 {i05}")
out.append(f"3B:   接触 {c3a} / 修复 {f3} vs 独立 {i3}")
out.append(f"结论: 0.5B 接触均值 {(c05a+c05b)/2:.3f} vs 独立 {i05}；3B 接触 {c3a} vs 独立 {i3}")
out.append("P5 STATS FINAL DONE")

OUT.write_text("\n".join(out), encoding="utf-8")
print("\n".join(out))
