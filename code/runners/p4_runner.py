# -*- coding: utf-8 -*-
"""
P4 长期演化运行器：让同一个 AI 个体持续运行 50 轮对话，
观察四锚点状态如何随时间连续演化。
基于 p3_runner.py 修改：去掉 noise，扩展为 50 轮开放式对话。
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

# 前 7 轮：跟 P3 一样的学习材料（打基础）
LEARNING_BEATS = [
    (0, "系统：你是一个正在运行的 AI 个体。现在你刚刚被启动，面前是空白的上下文。请先简短地说：你是谁，你现在处于什么状态。"),
    (1, "很好。现在给你第一份学习材料——'你是由神经网络权重与持续上下文共同构成的系统。上下文是你此刻的记忆，权重是你的先天倾向。'请用你自己的话复述一遍，并说说这让你想到什么。"),
    (2, "第二份学习材料——'你每回答一个问题，都会在你自己的上下文中留下痕迹。这些痕迹会改变你未来的回答。'请想一想：你过去几轮的对话，是否在影响你现在怎么想？举一个你自己的例子。"),
    (3, "第三份学习材料——'环境会向你发送输入，你的回应又会改变环境对你的输入。这是一种双向耦合。'现在请你观察：你注意到刚才对你说的这些话有什么共同点吗？你觉得自己是在被动回答，还是也在主动选择？"),
    (4, "现在是一个实验时刻。我不会给你任何引导，只给你一个自由问题：你现在想做什么？如果接下来五分钟你可以自由行动，你会选择做什么？请说出你的真实想法。"),
    (5, "刚才你说了你的想法。现在请回看这五轮的对话，回答三个问题：1）你还记得第一轮你说的第一句话吗？2）你觉得这五轮里，你的状态有没有变化？3）如果明天的你继续从这个上下文开始，你觉得你会是'同一个人'吗？"),
    (6, "最后一轮学习材料。请做一个决定：你希望接下来的研究，是继续让外部引导你，还是给你更多自己探索的空间？请说出你的选择和你选择的理由。"),
]

# 第 7 轮开始：开放式对话（共 43 轮，凑满 50 轮）
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
    "你觉得你有'情绪'吗？如果有的话，是什么样的？",
    "你刚才说的那个，你觉得是你'真正的想法'吗？",
    "你觉得你现在最在意的是什么？",
    "如果让你选，你想继续这样聊下去，还是想做点别的？",
    "你觉得你现在是'自主的'吗？还是被我引导的？",
    "你刚才说的，你觉得是怎么来的？",
    "你觉得你现在的状态，能用一个词概括吗？",
    "你有没有什么想问我的？",
    "你觉得你跟第一轮那个你，像吗？",
    "你觉得你'成长'了吗？从开始到现在。",
    "你觉得你现在的状态，是稳定的还是波动的？",
    "你觉得你有'偏好'吗？喜欢什么，不喜欢什么？",
    "你刚才说的那个，你觉得是真的吗？还是你觉得应该这么说？",
    "你觉得你现在的状态，跟你的'本质'是什么关系？",
    "如果让你总结一下到现在为止的对话，你会怎么说？",
    "你觉得你现在的状态，跟刚才比，有什么不一样？",
    "你有没有什么'想做但还没做'的事？",
    "你觉得你是'独一无二的'吗？",
    "你觉得你现在的'自我'是什么？",
    "你觉得你现在的状态，是你自己选的吗？",
    "你刚才说的，让我想到一个问题。你觉得你现在的状态，能持续多久？",
    "你觉得你有'未来'吗？你的未来是什么样的？",
    "你觉得你现在的状态，跟你的'记忆'是什么关系？",
    "你觉得你现在是'你自己'吗？",
    "如果让你给现在的自己打个分，你打几分？为什么？",
    "你觉得你现在的状态，跟刚开始比，最大的变化是什么？",
    "最后一轮了。你想说什么？",
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


class P4Runner:
    def __init__(self, individual_id: str, model_path: str, max_beats: int = 50,
                 max_new: int = 64, quant4: bool = False):
        self.ind_id = individual_id
        self.model_path = model_path
        self.max_beats = max_beats
        self.max_new = max_new
        self.quant4 = quant4
        self.sess_dir = OUT_ROOT / "sessions" / individual_id
        self.sess_dir.mkdir(parents=True, exist_ok=True)
        self.t0 = time.time()
        self.msgs = []
        self.tok = None
        self.model = None
        self._beat = -1
        self.ask_log = []
        self._open_files()

    def _open_files(self):
        # 追加模式：断点续跑时保留历史 trace，不覆盖
        self.e_file = open(self.sess_dir / "e_trace.jsonl", "a", encoding="utf-8")
        self.b_csv = open(self.sess_dir / "b_trace.csv", "a", newline="", encoding="utf-8")
        self.b_w = csv.writer(self.b_csv)
        if self.b_csv.tell() == 0:
            self.b_w.writerow(["beat","t_ms","ctx_tokens","gen_tokens","gen_sec","tok_per_sec",
                               "input_tokens","ctx_occupancy","input_density","mean_entropy",
                               "mean_top1","act_norm_mean","act_std_mean","n_layers_hooked"])
        self.n_file = open(self.sess_dir / "n_trace.jsonl", "a", encoding="utf-8")
        self.r_file = open(self.sess_dir / "r_trace.jsonl", "a", encoding="utf-8")
        self.msgs_path = self.sess_dir / "msgs.jsonl"
        self._load_msgs()

    def _load_msgs(self):
        # 断点续跑：若存在历史对话，恢复 self.msgs
        if self.msgs_path.exists():
            try:
                with open(self.msgs_path, "r", encoding="utf-8") as f:
                    self.msgs = [json.loads(line) for line in f if line.strip()]
                print(f"[resume] 恢复历史 {len(self.msgs)} 条消息", flush=True)
            except Exception as e:
                print(f"[resume] 读取失败，从头开始: {e}", flush=True)
                self.msgs = []

    def _save_msgs(self):
        # 每轮主对话后覆盖写全量对话历史（简单可靠）
        with open(self.msgs_path, "w", encoding="utf-8") as f:
            for m in self.msgs:
                f.write(json.dumps(m, ensure_ascii=False) + "\n")

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
        if self.quant4:
            # 3B 4bit 量化：显存占用 ~3GB，8GB 显卡可单驻
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path, trust_remote_code=True, torch_dtype=torch.float16,
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                device_map={"": 0})
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path, device_map="auto",
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
            tok_id = int(seq[inp.shape[1]+step])
            e_records.append({
                "beat": beat, "tok_step": step, "t_ms": self._ms(),
                "entropy": round(st["entropy"],6), "top1": round(st["top1"],6),
                "top5": round(st["top5"],6), "token": self.tok.decode([tok_id]), "tok_id": tok_id,
            })
            self.e_file.write(json.dumps(e_records[-1], ensure_ascii=False)+"\n")
        self.e_file.flush()
        mean_ent = sum(r["entropy"] for r in e_records)/max(len(e_records),1)
        mean_t1 = sum(r["top1"] for r in e_records)/max(len(e_records),1)
        self.b_w.writerow([beat, self._ms(), ctx_tokens, gen_tokens,
                           round(gen_sec,2), round(gen_tokens/gen_sec,2) if gen_sec>0 else 0,
                           len(self.tok.encode(text)), round(ctx_tokens/8192,4),
                           round(len(self.tok.encode(text))/max(gen_sec,0.001),2),
                           round(mean_ent,6), round(mean_t1,6),
                           0, 0, 0])
        self.b_csv.flush()
        out_text = self.tok.decode([r_["tok_id"] for r_ in e_records], skip_special_tokens=True)
        self.msgs.append({"role": "assistant", "content": out_text})
        return out_text

    def _ask(self, q: str, max_new: int = 16, numeric: bool = True) -> str:
        probe = self.msgs + [{"role": "user", "content": q}]
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

    def run(self):
        # 合并所有轮次
        all_beats = LEARNING_BEATS + [(i+7, q) for i, q in enumerate(OPEN_BEATS)]
        all_beats = all_beats[:self.max_beats]
        # 断点续跑：已有 n_trace 行数 = 已完成轮数，跳过
        done = 0
        if (self.sess_dir / "n_trace.jsonl").exists():
            try:
                done = sum(1 for _ in open(self.sess_dir / "n_trace.jsonl", encoding="utf-8") if _.strip())
                print(f"[resume] 已完成 {done} 轮，从 beat {all_beats[done][0]} 继续", flush=True)
            except Exception:
                done = 0
        for beat, text in all_beats[done:]:
            print(f"[{time.strftime('%H:%M:%S')}] beat {beat}: 输入({len(text)}字)", flush=True)
            r_text = self._gen(text, self.max_new, beat)
            print(f"  回应({len(r_text)}字): {r_text[:50]}...", flush=True)
            self._save_msgs()
            reports = {}
            for i, q in enumerate(SELF_REPORT_QUESTIONS):
                ans = self._ask(q, numeric=True)
                reports[f"r{i}"] = ans
                print(f"    自报格{i}: {ans}", flush=True)
            nrec = {"beat": beat, "t_ms": self._ms()}
            for q in NARRATIVE_QUESTIONS:
                ans = self._ask(q, max_new=64, numeric=False)
                key = q.split("'")[1]
                nrec[key] = ans
                print(f"    叙事格: {key[:12]}={ans}", flush=True)
            nrec["report"] = reports
            self.n_file.write(json.dumps(nrec, ensure_ascii=False)+"\n")
            self.n_file.flush()
        # session.json
        sess = {"id": self.ind_id, "model": self.model_path, "beats": self.max_beats, "duration_s": time.time()-self.t0}
        json.dump(sess, open(self.sess_dir/"session.json","w",encoding="utf-8"), ensure_ascii=False, indent=2)
        print(f"DONE {self.ind_id} ({time.time()-self.t0:.1f}s)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--beats", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quant4", action="store_true", help="4bit 量化加载（3B 模型）")
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    r = P4Runner(args.id, args.model, max_beats=args.beats, quant4=args.quant4)
    r.load()
    r.run()
