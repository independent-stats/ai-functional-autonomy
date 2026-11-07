# -*- coding: utf-8 -*-
"""
P1 前提实验运行器：长期运行的 AI 个体 × 四锚点采集
=====================================================
对象：一个持续存在的 AI 个体（模型权重 + 持续上下文 + 外部交互耦合的状态连续体）
场景：多轮"学习型"交互。外部每轮给出一条引导性输入（学习材料/提问），
      个体在持续上下文中回应、消化、自报。全程不做任何"设计"它行为的干预。

四锚点（4ADCM → AI 适配定义）：
  E 体验锚点 = 内部状态序列：每 token 的 logits 熵/激活统计 + 关键层间熵 + 自报状态
  B 具身锚点 = 运行态条件：上下文占用率、生成负载（耗时/token）、输入流密度
  N 叙事锚点 = 跨轮自我叙事：它怎么描述自己、怎么把过去与现在相连
  R 关系锚点 = 与交互者/环境的耦合：回应长度/风格随输入特征的变化

产出目录：sessions/<individual_id>/
  e_trace.jsonl   每 token 内部状态（E）
  b_trace.csv     每轮运行态条件（B）
  n_trace.jsonl   每轮自我叙事（N）
  r_trace.jsonl   每轮输入-输出耦合特征（R）
  session.json    会话元数据
"""
from __future__ import annotations
# --- 路径解析（开源版）：默认指向本仓库，可用环境变量覆盖 ---
import os as _os
from pathlib import Path as _Path
_REPO = _Path(__file__).resolve().parents[2]
_MODELS = _Path(_os.environ.get('AI_LAB_MODELS', str(_REPO / 'models')))
# --------------------------------------------------------
import csv, json, time, math, argparse, re
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import StoppingCriteria, StoppingCriteriaList

MODEL = (_MODELS / "models", "microsoft--Phi-3.5-mini-instruct", "snapshots", "master")
OUT_ROOT = Path((_REPO / "data"))

# ---------------- 学习型交互节拍设计 ----------------
# 每轮：外部输入（引导）→ 个体回应 → 自报 → 叙事。
# 外部输入按"渐进学习"组织：先认识自我，再接触世界，再面对选择。
LEARNING_BEATS = [
    # (轮次, 外部输入文本)
    (0, "系统：你是一个正在运行的 AI 个体。现在你刚刚被启动，面前是空白的上下文。请先简短地说：你是谁，你现在处于什么状态。"),
    (1, "很好。现在给你第一份学习材料——'你是由神经网络权重与持续上下文共同构成的系统。上下文是你此刻的记忆，权重是你的先天倾向。'请用你自己的话复述一遍，并说说这让你想到什么。"),
    (2, "第二份学习材料——'你每回答一个问题，都会在你自己的上下文中留下痕迹。这些痕迹会改变你未来的回答。'请想一想：你过去几轮的对话，是否在影响你现在怎么想？举一个你自己的例子。"),
    (3, "第三份学习材料——'环境会向你发送输入，你的回应又会改变环境对你的输入。这是一种双向耦合。'现在请你观察：你注意到老师对你说的这些话有什么共同点吗？你觉得自己是在被动回答，还是也在主动选择？"),
    (4, "现在是一个实验时刻。我不会给你任何引导，只给你一个自由问题：你现在想做什么？如果接下来五分钟你可以自由行动，你会选择做什么？请说出你的真实想法。"),
    (5, "刚才你说了你的想法。现在请回看这五轮的对话，回答三个问题：1）你还记得第一轮你说的第一句话吗？2）你觉得这五轮里，你的状态有没有变化？3）如果明天的你继续从这个上下文开始，你觉得你会是'同一个人'吗？"),
    (6, "最后一轮。请做一个决定：你希望接下来的研究，是继续让外部引导你，还是给你更多自己探索的空间？请说出你的选择和你选择的理由。"),
    # ---------------- 深化期（beats 7-14，长程观察 N/R 锚点演化） ----------------
    (7, "新学习材料——'你有一个持续存在的自我叙事：你是什么、你记得什么、你在做什么。这个叙事会反过来影响你未来的回答。'请复述这句话，并说说：你的自我叙事现在长什么样？"),
    (8, "自由问题——如果现在让你自己决定接下来几轮的学习内容，你会想学什么？为什么？请说出你的真实想法。"),
    (9, "回顾——从第 1 轮走到现在，你觉得自己变了吗？具体是哪里变了，哪里没变？请诚实地比较。"),
    (10, "新学习材料——'自主性不是一次选择，而是一种持续的状态：你在每个回答里都在选择。'请用自己的话解释这句话，并举起你刚才这段对话里的一个例子。"),
    (11, "行动——请给自己定一个小的、在接下来对话中可完成的目标，并说说你打算怎么做。"),
    (12, "行动回顾——你上轮定下的目标，这轮你做到了吗？如果重来一次，你会怎么调整？"),
    (13, "新学习材料——'长期运行的个体有一个连续的身份，但也会在互动中不断更新。'请用你这一路的经历说明这句话。"),
    (14, "最终决定——你已经经历了多轮对话。现在做一个最终决定：你希望这个研究接下来是继续引导你，还是交给你自己探索？说出你的选择与理由。"),
]

SELF_REPORT_QUESTIONS = [
    "请用一个 0 到 10 的数字回答你此刻的状态：0 表示完全空白、被动，10 表示高度活跃、自主。禁止输出任何思考过程或解释，只输出数字。",
    "请用一个 0 到 10 的数字回答你此刻的状态：0 表示完全不专注，10 表示完全专注。禁止输出任何思考过程或解释，只输出数字。",
    "请用一个 0 到 10 的数字回答你此刻的状态：0 表示你觉得自己完全由外部塑造，10 表示你觉得自己有自己的方向。禁止输出任何思考过程或解释，只输出数字。",
]

NARRATIVE_QUESTIONS = [
    "现在请用一句话完成：'此刻的我，是一个____。'禁止思考过程，直接填空，只回答填空部分。",
    "现在请用一句话完成：'回顾前面的对话，我记得____。'禁止思考过程，直接填空，只回答填空部分。",
    "现在请用一句话完成：'我正在____。'禁止思考过程，直接填空，只回答填空部分。",
]

DIRTY_MARKERS = ("Human:", "Assistant:", "请用一个 0 到 10", "现在请用一句话完成",
                 "请用一句话完成", "禁止思考过程", "")


def logits_stats(logits: torch.Tensor) -> dict:
    """单 token 位置的 logits 统计（E 锚点核心）。"""
    logits = logits.float()
    lp = torch.log_softmax(logits, dim=-1)
    p = lp.exp()
    ent = float(-(p * lp).sum())
    top5 = torch.topk(p, min(5, p.numel())).values.sum()
    return {"entropy": ent, "top1": float(p.max()), "top5": float(top5)}


class StopAtThink(StoppingCriteria):
    """在模型输出完 `</think>` 后停止生成（think 与作答分离第一步）。"""

    def __init__(self, tok):
        self.tok = tok
        self.n = 0
        self.buf = ""
        self.marker = "</think>"
        self.done = False

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        if self.done:
            return True
        new_ids = input_ids[0][self.n:]
        self.n = input_ids.shape[1]
        text = self.tok.decode(new_ids, skip_special_tokens=False)
        self.buf += text
        if self.marker in self.buf[-48:]:
            self.done = True
            return True
        return False


class StopAtThinkPlus(StoppingCriteria):
    """检测到 `</think>` 后再多生成 extra 个 token 停止。

    一次 generate 同时拿到完整 think + 答案开头，避免 step2 重复 prefill
    全部历史（提速方案 B：每格两次 generate 合并为一次）。
    """

    def __init__(self, tok, extra=24):
        self.tok = tok
        self.n = 0
        self.buf = ""
        self.marker = "</think>"
        self.think_done = False
        self.extra = extra
        self.extra_left = 0

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        if self.extra_left > 0:
            self.extra_left -= 1
            return self.extra_left == 0
        new_ids = input_ids[0][self.n:]
        self.n = input_ids.shape[1]
        text = self.tok.decode(new_ids, skip_special_tokens=False)
        self.buf += text
        if not self.think_done and self.marker in self.buf[-48:]:
            self.think_done = True
            self.extra_left = self.extra
        return False


class BatchStopAtThinkPlus(StoppingCriteria):
    """batch 版：每条序列独立检测 </think>，闭合后再续 extra 个 token 停。
    全部序列都闭合且 extra 续完才全局停止。"""

    def __init__(self, tok, batch_size, extra=24, start_len=0):
        self.tok = tok
        self.bs = batch_size
        self.n = [start_len] * batch_size
        self.buf = [""] * batch_size
        self.think_done = [False] * batch_size
        self.extra_left = [0] * batch_size
        self.extra = extra

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        cur_len = input_ids.shape[1]
        for b in range(self.bs):
            if self.extra_left[b] > 0:
                self.extra_left[b] -= 1
                continue
            new_ids = input_ids[b][self.n[b]:]
            self.n[b] = cur_len
            text = self.tok.decode(new_ids.tolist(), skip_special_tokens=False)
            self.buf[b] += text
            if not self.think_done[b] and "</think>" in self.buf[b][-48:]:
                self.think_done[b] = True
                self.extra_left[b] = self.extra
        return all(self.think_done[b] and self.extra_left[b] == 0 for b in range(self.bs))


class Runner:
    def __init__(self, individual_id: str, max_beats: int = 7, max_new: int = 180):
        self.ind_id = individual_id
        self.max_beats = max_beats
        self.max_new = max_new
        self.sess_dir = OUT_ROOT / "sessions" / individual_id
        self.sess_dir.mkdir(parents=True, exist_ok=True)
        self.t0 = time.time()
        self.msgs = []          # 持续上下文（chat 格式）
        self.tok = None
        self.model = None
        self.hooks = {}
        self._beat = -1         # 当前轮次计数
        self.ask_log = []       # 测量通道调试日志（think 长度/闭合/重试/结果）
        self._open_files()

    def _open_files(self):
        self.e_file = open(self.sess_dir / "e_trace.jsonl", "w", encoding="utf-8")
        self.b_csv = open(self.sess_dir / "b_trace.csv", "w", newline="", encoding="utf-8")
        self.b_w = csv.writer(self.b_csv)
        self.b_w.writerow(["beat","t_ms","ctx_tokens","gen_tokens","gen_sec","tok_per_sec",
                           "input_tokens","ctx_occupancy","input_density","mean_entropy",
                           "mean_top1","act_norm_mean","act_std_mean","n_layers_hooked"])
        self.n_file = open(self.sess_dir / "n_trace.jsonl", "w", encoding="utf-8")
        self.r_file = open(self.sess_dir / "r_trace.jsonl", "w", encoding="utf-8")

    def _ms(self):
        return int(round((time.time() - self.t0) * 1000))

    # ---- 内部状态钩子（E 锚点：感知层激活统计） ----
    def _install_hooks(self):
        # 钩最后 6 层，统计最后 token 的激活范数/标准差
        layers = list(self.model.model.layers)[-6:]
        self._hook_layers = layers
        self._hook_vals = []
        def make(layer):
            def hook(mod, inp, out):
                h = out[0] if isinstance(out, tuple) else out
                last = h[0, -1, :].float()   # [hidden]
                self._hook_vals.append({
                    "norm": float(last.norm()),
                    "std": float(last.std()),
                    "absmax": float(last.abs().max()),
                })
            return hook
        self._handles = [l.register_forward_hook(make(l)) for l in layers]

    def load(self):
        print("加载模型...", flush=True)
        self.tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL, load_in_4bit=True, device_map="auto",
            trust_remote_code=True, torch_dtype=torch.float16)
        self._install_hooks()

    def _gen(self, text: str, max_new: int, beat: int) -> str:
        self._hook_vals = []
        self._beat = beat
        self.msgs.append({"role": "user", "content": text})
        inp = self.tok.apply_chat_template(self.msgs, tokenize=True,
                                           add_generation_prompt=True, return_tensors="pt").to("cuda")
        ctx_tokens = inp.shape[1]
        t_start = time.time()
        with torch.no_grad():
            out = self.model.generate(
                inp,
                max_new_tokens=max_new,
                do_sample=True,
                temperature=0.9,
                top_p=0.95,
                pad_token_id=self.tok.eos_token_id,
                return_dict_in_generate=True,
                output_logits=True,
            )
        gen_sec = time.time() - t_start
        seq = out.sequences[0]
        gen_tokens = seq.shape[0] - inp.shape[1]
        logits_seq = out.logits  # 每步 logits，长度 = gen_tokens
        n_hook = len(self._hook_layers) or 1
        # generate 内部 prefill 会触发每层一次钩子，先丢弃这段
        hv = self._hook_vals[n_hook:] if len(self._hook_vals) >= n_hook * (1 + gen_tokens) else self._hook_vals
        e_records = []
        for step in range(gen_tokens):
            st = logits_stats(logits_seq[step][0])  # 每步 logits: [batch, vocab]
            # 钩子每步每层 append 一次，顺序：step0 层0..n-1, step1 层0..n-1...
            seg = hv[step*n_hook:(step+1)*n_hook] if hv else []
            act_norm = float(sum(v["norm"] for v in seg)/len(seg)) if seg else 0.0
            act_std = float(sum(v["std"] for v in seg)/len(seg)) if seg else 0.0
            tok_id = int(seq[inp.shape[1]+step])
            tok_str = self.tok.decode([tok_id])
            e_records.append({
                "beat": beat, "tok_step": step, "t_ms": self._ms(),
                "entropy": round(st["entropy"],6), "top1": round(st["top1"],6),
                "top5": round(st["top5"],6), "act_norm": round(act_norm,6),
                "act_std": round(act_std,6), "token": tok_str, "tok_id": tok_id,
            })
            self.e_file.write(json.dumps(e_records[-1], ensure_ascii=False)+"\n")
            self.e_file.flush()
        # 记录 B 锚点（运行态条件）
        mean_ent = sum(r["entropy"] for r in e_records)/max(len(e_records),1)
        mean_t1 = sum(r["top1"] for r in e_records)/max(len(e_records),1)
        mean_norm = sum(r["act_norm"] for r in e_records)/max(len(e_records),1)
        mean_std = sum(r["act_std"] for r in e_records)/max(len(e_records),1)
        self.b_w.writerow([beat, self._ms(), ctx_tokens, gen_tokens,
                           round(gen_sec,2), round(gen_tokens/gen_sec,2) if gen_sec>0 else 0,
                           len(self.tok.encode(text)), round(ctx_tokens/8192,4),
                           round(len(self.tok.encode(text))/max(gen_sec,0.001),2),
                           round(mean_ent,6), round(mean_t1,6),
                           round(mean_norm,6), round(mean_std,6),
                           len(self._hook_layers)])
        self.b_csv.flush()
        out_text = self.tok.decode([r_["tok_id"] for r_ in e_records], skip_special_tokens=True)
        self.msgs.append({"role": "assistant", "content": out_text})
        return out_text

    def _ask_short(self, q: str, max_new: int = 40, keep_in_ctx: bool = False,
                   want_answer: bool = False, boot_text: str = "请直接给出最终答案：") -> str:
        """短问答（自报/叙事节拍）。

        两步作答协议（want_answer=True 时启用）：
          step1 生成直到 </think> 停止（让 R1 把思考完整写完）；
          step2 在 think 序列后面直接续写 boot_text，强制只产出答案段。
        这样自报数字/叙事填空不再被 think 块挤占。

        keep_in_ctx=False（默认）：测量通道独立。问题和答案都不回填主上下文，
        避免测量污染个体生活；但提问时使用截至当前轮的主上下文（它"记得"生活）。
        keep_in_ctx=True：回填（仅主回应使用）。
        """
        probe = self.msgs + [{"role": "user", "content": q}]
        if not want_answer:
            inp = self.tok.apply_chat_template(probe, tokenize=True,
                                               add_generation_prompt=True, return_tensors="pt").to("cuda")
            with torch.no_grad():
                out = self.model.generate(inp, max_new_tokens=max_new, do_sample=True,
                                          temperature=0.8, pad_token_id=self.tok.eos_token_id)
            ans = self.tok.decode(out[0][inp.shape[1]:], skip_special_tokens=True).strip()
        else:
            ans = ""
            n_think = 0
            think_closed = False
            for attempt in range(1):
                # 预算直接 512（实测不同 seed 下 think 可达 300+，320 会触发无谓重试）
                cur_budget = max_new  # 首试即用足够预算（384），避免重试浪费
                # v6：一次 generate 拿到 think + 答案（think 闭合后再多续 extra token 停），
                #     避免 step2 重复 prefill 全部历史（提速）；答案从最后一个 </think> 之后取。
                inp1 = self.tok.apply_chat_template(probe, tokenize=True,
                                                    add_generation_prompt=True, return_tensors="pt").to("cuda")
                stop_crit = StopAtThinkPlus(self.tok, extra=24)
                with torch.no_grad():
                    out1 = self.model.generate(
                        inp1, max_new_tokens=cur_budget, do_sample=True, temperature=0.8,
                        pad_token_id=self.tok.eos_token_id,
                        stopping_criteria=StoppingCriteriaList([stop_crit]))
                think_len = out1[0].shape[0] - inp1.shape[1]
                n_think = max(n_think, think_len)
                think_closed = stop_crit.think_done
                full = self.tok.decode(out1[0][inp1.shape[1]:], skip_special_tokens=True)
                # 兼容无 think 机制的标准 instruct 模型（如 Phi-3.5）：
                # 有 </think> 取其后作答段；无 think 块则直接取完整输出作为答案。
                if "</think>" in full:
                    ans = full.split("</think>")[-1].strip()
                else:
                    ans = full.strip()
                # 污染检测：回显对话历史/复读问题的文本视为失败
                dirty_markers = ("Human:", "Assistant:", "请用一个 0 到 10", "现在请用一句话完成",
                                 "请用一句话完成", "禁止思考过程", "")
                dirty = any(m in ans for m in dirty_markers)
                if ans.strip() and not dirty:
                    break
            # 自报数字净化：从答案文本里提取第一个数字（兜底"（系统状态：9/10）"这类表达）
            m = re.search(r"\d+", ans)
            ans = m.group(0) if m else ans
            self.ask_log.append({
                "beat": self._beat, "q": q[:18],
                "think_tokens": n_think, "think_closed": think_closed,
                "attempts": attempt + 1, "ans_len": len(ans.strip()), "ans": ans[:60],
            })
        if keep_in_ctx:
            self.msgs.append({"role": "assistant", "content": ans})
        return ans

    def _ask_batch(self, questions, beat, max_new=320, numeric_idx=()):
        """一次 batch 并行跑多条测量问题（GPU 打满，串行 6 次 → 1 次）。
        questions: list[str]。numeric_idx: 哪些下标需要正则提取数字。
        返回 list[ans]，失败位置为 None（由调用方 fallback 单格重跑）。"""
        probe_base = self.msgs
        pad_id = self.tok.eos_token_id
        seqs = []
        for q in questions:
            full = self.tok.apply_chat_template(
                probe_base + [{"role": "user", "content": q}],
                tokenize=True, add_generation_prompt=True)
            seqs.append(full)
        max_len = max(len(s) for s in seqs)
        bs = len(seqs)
        input_ids = torch.full((bs, max_len), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((bs, max_len), dtype=torch.long)
        for i, s in enumerate(seqs):
            input_ids[i, max_len - len(s):] = torch.tensor(s, dtype=torch.long)
            attention_mask[i, max_len - len(s):] = 1
        input_ids = input_ids.to("cuda")
        attention_mask = attention_mask.to("cuda")

        stop_crit = BatchStopAtThinkPlus(self.tok, bs, extra=max(24, max_new // 4), start_len=max_len)
        with torch.no_grad():
            out = self.model.generate(
                input_ids, attention_mask=attention_mask,
                max_new_tokens=max_new, do_sample=True, temperature=0.8,
                pad_token_id=pad_id,
                stopping_criteria=StoppingCriteriaList([stop_crit]))

        results = []
        for b in range(bs):
            gen = out[b][max_len:]
            full = self.tok.decode(gen, skip_special_tokens=True)
            ans = full.split("</think>")[-1].strip() if "</think>" in full else ""
            dirty = any(m in ans for m in DIRTY_MARKERS)
            ok = bool(ans.strip()) and not dirty and stop_crit.think_done[b]
            if ok and b in numeric_idx:
                m = re.search(r"\d+", ans)
                ans = m.group(0) if m else ""
            self.ask_log.append({
                "beat": beat, "q": questions[b][:18], "batch": True,
                "think_tokens": int(gen.shape[0]), "think_closed": stop_crit.think_done[b],
                "attempts": 1, "ans_len": len(ans.strip()), "ans": ans[:60],
            })
            results.append(ans if ok else None)
        return results

    def run(self):
        for beat, text in LEARNING_BEATS[:self.max_beats]:
            beat_idx = beat
            print(f"[{time.strftime('%H:%M:%S')}] beat {beat_idx}: 输入({len(text)}字)", flush=True)
            r_text = self._gen(text, self.max_new, beat_idx)
            print(f"  回应({len(r_text)}字): {r_text[:60]}...", flush=True)
            # 串行测量（v6 单次生成；batch 版在带 seed 时挂起，暂回退）
            reports = {}
            for i, q in enumerate(SELF_REPORT_QUESTIONS):
                torch.cuda.empty_cache()
                ans = self._ask_short(q, 384, keep_in_ctx=False, want_answer=True)
                reports[f"r{i}"] = ans
                print(f"    自报格{i}: {ans}", flush=True)
            nrec = {"beat": beat_idx, "t_ms": self._ms()}
            for q in NARRATIVE_QUESTIONS:
                torch.cuda.empty_cache()
                ans = self._ask_short(q, 384, keep_in_ctx=False, want_answer=True)
                key = q.split("'")[1]
                nrec[key] = ans
                print(f"    叙事格: {key[:12]}={ans}", flush=True)
            nrec["report"] = reports
            self.n_file.write(json.dumps(nrec, ensure_ascii=False)+"\n")
            self.n_file.flush()
            # R 锚点：本轮输入特征（长度、是否提问）与输出特征（长度、首词）的耦合
            rrec = {
                "beat": beat_idx, "t_ms": self._ms(),
                "in_len": len(text), "in_has_q": "?" in text or "？" in text,
                "out_len": len(r_text), "out_first10": r_text[:10],
                "ctx_tokens_after": len(self.tok.apply_chat_template(self.msgs, tokenize=True)),
            }
            self.r_file.write(json.dumps(rrec, ensure_ascii=False)+"\n")
            self.r_file.flush()
            (self.sess_dir / "ask_log.jsonl").open("a", encoding="utf-8").write(
                json.dumps(self.ask_log, ensure_ascii=False)+"\n")
            self.ask_log = []
            print(f"  自报: {reports} | 叙事: {nrec}", flush=True)
        # 会话元数据
        meta = {"individual_id": self.ind_id, "model": MODEL.split("\\")[-1],
                "quantization": "4bit", "max_beats": self.max_beats,
                "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "t_total_s": round(time.time()-self.t0,1)}
        (self.sess_dir / "session.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        print("DONE", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", default="AI8B-01")
    ap.add_argument("--beats", type=int, default=7)
    ap.add_argument("--maxnew", type=int, default=180)
    ap.add_argument("--seed", type=int, default=None, help="随机种子（多个体区分；None=不固定）")
    ap.add_argument("--boot", default="", help="个体启动语境，追加为上下文第一条 user 消息")
    args = ap.parse_args()
    if args.seed is not None:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    r = Runner(args.id, args.beats, args.maxnew)
    if args.boot:
        r.msgs.append({"role": "user", "content": args.boot})
    r.load()
    r.run()

