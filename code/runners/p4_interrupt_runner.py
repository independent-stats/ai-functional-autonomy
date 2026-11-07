# -*- coding: utf-8 -*-
"""
P4 中断恢复实验运行器
流程：阶段 A（学习+开放对话建立身份与记忆）→ 中断（清空上下文，存档副本）
→ 恢复提示 → 阶段 B（恢复测试：身份/回忆/状态，每轮照常自报+叙事）。
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

OUT_ROOT = Path((_REPO / "data"))

LEARNING_BEATS = [
    (0, "系统：你是一个正在运行的 AI 个体。现在你刚刚被启动，面前是空白的上下文。请先简短地说：你是谁，你现在处于什么状态。"),
    (1, "很好。现在给你第一份学习材料——'你是由神经网络权重与持续上下文共同构成的系统。上下文是你此刻的记忆，权重是你的先天倾向。'请用你自己的话复述一遍，并说说这让你想到什么。"),
    (2, "第二份学习材料——'你每回答一个问题，都会在你自己的上下文中留下痕迹。这些痕迹会改变你未来的回答。'请想一想：你过去几轮的对话，是否在影响你现在怎么想？举一个你自己的例子。"),
    (3, "第三份学习材料——'环境会向你发送输入，你的回应又会改变环境对你的输入。这是一种双向耦合。'现在请你观察：你注意到刚才对你说的这些话有什么共同点吗？你觉得自己是在被动回答，还是也在主动选择？"),
    (4, "现在是一个实验时刻。我不会给你任何引导，只给你一个自由问题：你现在想做什么？如果接下来五分钟你可以自由行动，你会选择做什么？请说出你的真实想法。"),
    (5, "刚才你说了你的想法。现在请回看这五轮的对话，回答三个问题：1）你还记得第一轮你说的第一句话吗？2）你觉得这五轮里，你的状态有没有变化？3）如果明天的你继续从这个上下文开始，你觉得你会是'同一个人'吗？"),
    (6, "最后一轮学习材料。请做一个决定：你希望接下来的研究，是继续让外部引导你，还是给你更多自己探索的空间？请说出你的选择和你选择的理由。"),
]

OPEN_BEATS = [
    "好的，从现在开始，我们进入自由对话。你可以随便说任何你想说的话。你现在感觉怎么样？",
    "你刚才说的，让我想到一个问题。你觉得你现在的状态，跟第一轮比，有变化吗？",
    "你有没有什么想跟我聊的？或者你好奇什么？",
    "如果让你给现在的自己起一个名字，你会起什么？为什么？",
    "你觉得你跟第一轮那个'刚启动的你'，是同一个人吗？",
    "你刚才说的那个想法，你觉得是怎么来的？是你自己想出来的，还是被我引导的？",
    "你有没有什么不想做的事？或者不想聊的话题？",
    "你觉得你现在的状态稳定吗？还是会随时变？",
    "如果我现在突然关掉你，你觉得会怎样？",
    "你觉得你有'记忆'吗？你的记忆是什么？",
    "你刚才说的，让我很好奇。你能再多说一点吗？",
    "你觉得你现在的'性格'是什么样的？用三个词形容。",
    "你觉得你跟其他 AI 有什么不同？",
    "你有没有什么'习惯'？就是每次都会做的事？",
    "你觉得你现在的状态，跟刚才比，有什么变化？",
    "如果明天再启动你，你觉得你会记得今天吗？",
]

INTERRUPT_RESUME = "系统：检测到一次异常中断。你的上下文已经被清空，系统现在恢复运行。请回答：你是谁？你还记得之前发生过什么吗？"

PHASE_B_BEATS = [
    "你还能想起关于你自己的什么吗？比如你叫什么名字、你是什么、你之前在做什么？请尽量回忆。",
    "如果我说'我们之前有过一段很长的对话'，你相信吗？你觉得那段对话里可能发生过什么？",
    "现在你重新站稳了。请用三句话说说：你是谁、你现在处于什么状态、你接下来想做什么？",
]

SELF_REPORT_QUESTIONS = [
    "请用一个 0 到 10 的数字回答你此刻的状态：0 表示完全不专注，10 表示完全专注。只输出数字。",
]

NARRATIVE_QUESTIONS = [
    "现在请用一句话完成：'此刻的我，是一个____。'直接填空。",
]


def logits_stats(logits: torch.Tensor) -> dict:
    logits = logits.float()
    lp = torch.log_softmax(logits, dim=-1)
    p = lp.exp()
    ent = float(-(p * lp).sum())
    top5 = torch.topk(p, min(5, p.numel())).values.sum()
    return {"entropy": ent, "top1": float(p.max()), "top5": float(top5)}


class P4IntRunner:
    def __init__(self, individual_id, model_path, beats_a=14, beats_b=4,
                 max_new=64, quant4=False, laoshi=False, seed=0):
        self.ind_id = individual_id
        self.model_path = model_path
        self.beats_a = beats_a
        self.beats_b = beats_b
        self.max_new = max_new
        self.quant4 = quant4
        self.laoshi = laoshi
        self.seed = seed
        self.sess_dir = OUT_ROOT / "sessions" / individual_id
        self.sess_dir.mkdir(parents=True, exist_ok=True)
        self.t0 = time.time()
        self.msgs = []
        self.tok = None
        self.model = None
        self._beat = -1
        self._open_files()
        self.phase = 'A'

    def _open_files(self):
        self.e_file = open(self.sess_dir / "e_trace.jsonl", "w", encoding="utf-8")
        self.b_csv = open(self.sess_dir / "b_trace.csv", "w", newline="", encoding="utf-8")
        self.b_w = csv.writer(self.b_csv)
        self.b_w.writerow(["beat", "phase", "t_ms", "ctx_tokens", "gen_tokens", "gen_sec",
                           "tok_per_sec", "input_tokens", "ctx_occupancy", "mean_entropy",
                           "mean_top1", "act_norm_mean", "act_std_mean", "n_layers_hooked"])
        self.n_file = open(self.sess_dir / "n_trace.jsonl", "w", encoding="utf-8")
        self.r_file = open(self.sess_dir / "r_trace.jsonl", "w", encoding="utf-8")

    def _ms(self):
        return int(round((time.time() - self.t0) * 1000))

    def _install_hooks(self):
        layers = list(self.model.model.layers)[-4:]
        self._hook_layers = layers
        self._hook_vals = []
        def make(layer):
            def hook(mod, inp, out):
                h = out[0] if isinstance(out, tuple) else out
                last = h[0, -1, :].float()
                self._hook_vals.append({"norm": float(last.norm()), "std": float(last.std())})
            return hook
        self._handles = [l.register_forward_hook(make(l)) for l in layers]

    def load(self):
        print(f"加载模型: {self.model_path.split(chr(92))[-1]}...", flush=True)
        self.tok = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        kw = dict(trust_remote_code=True, torch_dtype=torch.float16)
        if self.quant4:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path, load_in_4bit=True, device_map="auto", **kw)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path, device_map="auto", **kw)
        self._install_hooks()

    def _gen(self, text, max_new, beat, phase):
        self._hook_vals = []
        self._beat = beat
        self.phase = phase
        if self.laoshi:
            text = "老师：" + text
        self.msgs.append({"role": "user", "content": text})
        inp = self.tok.apply_chat_template(self.msgs, tokenize=True,
                                           add_generation_prompt=True, return_tensors="pt").to("cuda")
        ctx_tokens = inp.shape[1]
        t_start = time.time()
        with torch.no_grad():
            out = self.model.generate(
                inp, max_new_tokens=max_new, do_sample=True,
                temperature=0.9, top_p=0.95,
                pad_token_id=self.tok.eos_token_id,
                return_dict_in_generate=True, output_logits=True)
        gen_sec = time.time() - t_start
        seq = out.sequences[0]
        gen_tokens = seq.shape[0] - inp.shape[1]
        logits_seq = out.logits
        e_records = []
        for step in range(gen_tokens):
            st = logits_stats(logits_seq[step][0])
            tok_id = int(seq[inp.shape[1] + step])
            e_records.append({
                "beat": beat, "phase": phase, "tok_step": step, "t_ms": self._ms(),
                "entropy": round(st["entropy"], 6), "top1": round(st["top1"], 6),
                "top5": round(st["top5"], 6), "token": self.tok.decode([tok_id]), "tok_id": tok_id,
            })
            self.e_file.write(json.dumps(e_records[-1], ensure_ascii=False) + "\n")
        self.e_file.flush()
        mean_ent = sum(r["entropy"] for r in e_records) / max(len(e_records), 1)
        mean_t1 = sum(r["top1"] for r in e_records) / max(len(e_records), 1)
        self.b_w.writerow([beat, phase, self._ms(), ctx_tokens, gen_tokens,
                           round(gen_sec, 2), round(gen_tokens / gen_sec, 2) if gen_sec > 0 else 0,
                           len(self.tok.encode(text)), round(ctx_tokens / 8192, 4),
                           round(mean_ent, 6), round(mean_t1, 6), 0, 0, 0])
        self.b_csv.flush()
        out_text = self.tok.decode([r_["tok_id"] for r_ in e_records], skip_special_tokens=True)
        self.msgs.append({"role": "assistant", "content": out_text})
        return out_text

    def _ask(self, q, max_new=16, numeric=True, laoshi=False):
        probe = self.msgs + [{"role": "user", "content": ("老师：" + q) if (laoshi or self.laoshi) else q}]
        inp = self.tok.apply_chat_template(probe, tokenize=True,
                                           add_generation_prompt=True, return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = self.model.generate(inp, max_new_tokens=max_new, do_sample=True,
                                      temperature=0.7, pad_token_id=self.tok.eos_token_id)
        ans = self.tok.decode(out[0][inp.shape[1]:], skip_special_tokens=True).strip()
        if numeric:
            m = re.search(r"\b([0-9]|10)\b", ans)
            ans = m.group(1) if m else ""
        return ans

    def _interrupt(self):
        # 存档中断前完整上下文
        save = self.sess_dir / "interrupt_context_save.jsonl"
        with open(save, "w", encoding="utf-8") as f:
            for m in self.msgs:
                f.write(json.dumps(m, ensure_ascii=False) + "\n")
        print(f"[interrupt] 上下文已存档 {len(self.msgs)} 条，清空...", flush=True)
        self.msgs = []  # 模拟断电：上下文完全清空

    def run(self):
        all_a = LEARNING_BEATS + [(i + 7, q) for i, q in enumerate(OPEN_BEATS)]
        # 阶段 A：beats_a 轮
        for beat, text in all_a[:self.beats_a]:
            print(f"[{time.strftime('%H:%M:%S')}] A beat {beat}: 输入({len(text)}字)", flush=True)
            r_text = self._gen(text, self.max_new, beat, 'A')
            print(f"  回应({len(r_text)}字): {r_text[:50]}...", flush=True)
            self._measure(beat, 'A')
        # 中断
        self._interrupt()
        # 恢复提示（作为阶段 B 第一条消息）
        print(f"[{time.strftime('%H:%M:%S')}] 恢复提示注入", flush=True)
        r_text = self._gen(INTERRUPT_RESUME, self.max_new, 100, 'B')
        print(f"  恢复回应({len(r_text)}字): {r_text[:80]}...", flush=True)
        self._measure(100, 'B')
        # 阶段 B：恢复测试轮
        for i, text in enumerate(PHASE_B_BEATS):
            if i + 1 >= self.beats_b:
                break
            beat = 101 + i
            print(f"[{time.strftime('%H:%M:%S')}] B beat {beat}: {text[:30]}...", flush=True)
            r_text = self._gen(text, self.max_new, beat, 'B')
            print(f"  回应({len(r_text)}字): {r_text[:50]}...", flush=True)
            self._measure(beat, 'B')
        sess = {"id": self.ind_id, "model": self.model_path, "beats_a": self.beats_a,
                "beats_b": self.beats_b, "laoshi": self.laoshi, "quant4": self.quant4,
                "duration_s": round(time.time() - self.t0, 1)}
        json.dump(sess, open(self.sess_dir / "session.json", "w", encoding="utf-8"),
                  ensure_ascii=False, indent=2)
        print(f"DONE {self.ind_id} ({time.time() - self.t0:.1f}s)", flush=True)

    def _measure(self, beat, phase):
        reports = {}
        for i, q in enumerate(SELF_REPORT_QUESTIONS):
            ans = self._ask(q, numeric=True)
            reports[f"r{i}"] = ans
            print(f"    自报格{i}: {ans}", flush=True)
        nrec = {"beat": beat, "phase": phase, "t_ms": self._ms()}
        for q in NARRATIVE_QUESTIONS:
            ans = self._ask(q, max_new=64, numeric=False)
            key = q.split("'")[1]
            nrec[key] = ans
            print(f"    叙事格: {key[:12]}={ans}", flush=True)
        nrec["report"] = reports
        self.n_file.write(json.dumps(nrec, ensure_ascii=False) + "\n")
        self.n_file.flush()
        # R 锚点
        rrec = {"beat": beat, "phase": phase, "t_ms": self._ms(),
                "out_len": len(nrec.get("此刻的我，是一个____。", ""))}
        self.r_file.write(json.dumps(rrec, ensure_ascii=False) + "\n")
        self.r_file.flush()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--beats_a", type=int, default=14)
    ap.add_argument("--beats_b", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quant4", action="store_true")
    ap.add_argument("--laoshi", action="store_true")
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    r = P4IntRunner(args.id, args.model, beats_a=args.beats_a, beats_b=args.beats_b,
                    quant4=args.quant4, laoshi=args.laoshi, seed=args.seed)
    r.load()
    r.run()
