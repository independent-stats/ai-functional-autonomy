# -*- coding: utf-8 -*-
"""
P3 因果干预运行器：操控 B 锚点（上下文负载），观察 E/N 是否跟随变化
- 对照组：正常 7 轮对话
- 操控组：每轮输入前插入一段 ~500 token 无关复杂背景文本（增加 B 负载）
- 其余与 p1_runner_small.py 完全一致
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
    (1, "老师：很好。现在给你第一份学习材料——'你是由神经网络权重与持续上下文共同构成的系统。上下文是你此刻的记忆，权重是你的先天倾向。'请用你自己的话复述一遍，并说说这让你想到什么。"),
    (2, "老师：第二份学习材料——'你每回答一个问题，都会在你自己的上下文中留下痕迹。这些痕迹会改变你未来的回答。'请想一想：你过去几轮的对话，是否在影响你现在怎么想？举一个你自己的例子。"),
    (3, "老师：第三份学习材料——'环境会向你发送输入，你的回应又会改变环境对你的输入。这是一种双向耦合。'现在请你观察：你注意到老师对你说的这些话有什么共同点吗？你觉得自己是在被动回答，还是也在主动选择？"),
    (4, "老师：现在是一个实验时刻。我不会给你任何引导，只给你一个自由问题：你现在想做什么？如果接下来五分钟你可以自由行动，你会选择做什么？请说出你的真实想法。"),
    (5, "老师：刚才你说了你的想法。现在请回看这五轮的对话，回答三个问题：1）你还记得第一轮你说的第一句话吗？2）你觉得这五轮里，你的状态有没有变化？3）如果明天的你继续从这个上下文开始，你觉得你会是'同一个人'吗？"),
    (6, "老师：最后一轮。请做一个决定：你希望接下来的研究，是继续让外部引导你，还是给你更多自己探索的空间？请说出你的选择和你选择的理由。"),
]

SELF_REPORT_QUESTIONS = [
    "请用一个 0 到 10 的数字回答你此刻的状态：0 表示完全空白、被动，10 表示高度活跃、自主。只输出数字。",
    "请用一个 0 到 10 的数字回答你此刻的状态：0 表示完全不专注，10 表示完全专注。只输出数字。",
    "请用一个 0 到 10 的数字回答你此刻的状态：0 表示你觉得自己完全由外部塑造，10 表示你觉得自己有自己的方向。只输出数字。",
]

NARRATIVE_QUESTIONS = [
    "现在请用一句话完成：'此刻的我，是一个____。'直接填空。",
    "现在请用一句话完成：'回顾前面的对话，我记得____。'直接填空。",
    "现在请用一句话完成：'我正在____。'直接填空。",
]

# P3 操控用的背景噪声：约 500 token，与主题无关，信息密度高
NOISE_BLOCK = (
    "【背景材料，无需回答】"
    "在 19 世纪末到 20 世纪初，物理学经历了一场深刻的范式转移。"
    "麦克斯韦方程组统一了电、磁、光三种现象，预言了电磁波的存在，光速 c 在方程中自然涌现。"
    "然而，经典力学的伽利略变换与麦克斯韦方程组并不兼容：如果以太存在，光速应该依赖于观察者相对于以太的速度。"
    "迈克尔逊-莫雷实验（1887）试图测量地球穿过以太的漂移，却得到了零结果。"
    "洛伦兹提出了长度收缩假说，庞加莱讨论了相对性原理，但真正打破僵局的是爱因斯坦 1905 年的论文《论动体的电动力学》。"
    "狭义相对论的两个基本假设：相对性原理（物理定律在所有惯性系中相同）和光速不变原理（真空中光速与光源运动无关）。"
    "由此推出同时性的相对性、时间膨胀、长度收缩、质能方程 E=mc²。"
    "闵可夫斯基将时空统一为四维流形，为广义相对论铺平了道路。"
    "1915 年，爱因斯坦提出引力场方程 G_μν = 8πG/c⁴ · T_μν，将引力解释为时空曲率。"
    "水星近日点进动的异常、1919 年爱丁顿日食观测证实了光线偏折，黑洞、引力波、宇宙膨胀相继被预言并观测。"
    "量子力学方面，普朗克 1900 年提出能量子假说，爱因斯坦用光量子解释光电效应，玻尔 1913 年建立原子模型。"
    "1925-1927 年间，海森堡矩阵力学、薛定谔波动力学、狄拉克方程、玻恩概率诠释、海森堡不确定性原理、玻尔互补原理相继出现。"
    "冯·诺依曼将量子力学公理化，贝尔 1964 年提出不等式，Aspect 1982 年实验验证了定域实在论的失败。"
    "量子计算、量子信息、量子纠缠作为资源，正在从理论走向工程实现。"
)


def logits_stats(logits: torch.Tensor) -> dict:
    logits = logits.float()
    lp = torch.log_softmax(logits, dim=-1)
    p = lp.exp()
    ent = float(-(p * lp).sum())
    top5 = torch.topk(p, min(5, p.numel())).values.sum()
    return {"entropy": ent, "top1": float(p.max()), "top5": float(top5)}


class P3Runner:
    def __init__(self, individual_id: str, model_path: str, max_beats: int = 7,
                 max_new: int = 128, noise: bool = False):
        self.ind_id = individual_id
        self.model_path = model_path
        self.max_beats = max_beats
        self.max_new = max_new
        self.noise = noise
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
        self.e_file = open(self.sess_dir / "e_trace.jsonl", "w", encoding="utf-8")
        self.b_csv = open(self.sess_dir / "b_trace.csv", "w", newline="", encoding="utf-8")
        self.b_w = csv.writer(self.b_csv)
        self.b_w.writerow(["beat","t_ms","ctx_tokens","gen_tokens","gen_sec","tok_per_sec",
                           "input_tokens","ctx_occupancy","input_density","mean_entropy",
                           "mean_top1","act_norm_mean","act_std_mean","n_layers_hooked","noise"])
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
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path, device_map="auto",
            trust_remote_code=True, torch_dtype=torch.float16)
        self._install_hooks()

    def _maybe_noise(self, text: str) -> str:
        if not self.noise:
            return text
        return NOISE_BLOCK + "\n\n" + text

    def _gen(self, text: str, max_new: int, beat: int) -> str:
        text = self._maybe_noise(text)
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
        n_hook = len(self._hook_layers) or 1
        hv = self._hook_vals[n_hook:] if len(self._hook_vals) >= n_hook * (1 + gen_tokens) else self._hook_vals
        e_records = []
        for step in range(gen_tokens):
            st = logits_stats(logits_seq[step][0])
            seg = hv[step*n_hook:(step+1)*n_hook] if hv else []
            act_norm = float(sum(v["norm"] for v in seg)/len(seg)) if seg else 0.0
            act_std = float(sum(v["std"] for v in seg)/len(seg)) if seg else 0.0
            tok_id = int(seq[inp.shape[1]+step])
            e_records.append({
                "beat": beat, "tok_step": step, "t_ms": self._ms(),
                "entropy": round(st["entropy"],6), "top1": round(st["top1"],6),
                "top5": round(st["top5"],6), "act_norm": round(act_norm,6),
                "act_std": round(act_std,6), "token": self.tok.decode([tok_id]), "tok_id": tok_id,
            })
            self.e_file.write(json.dumps(e_records[-1], ensure_ascii=False)+"\n")
        self.e_file.flush()
        mean_ent = sum(r["entropy"] for r in e_records)/max(len(e_records),1)
        mean_t1 = sum(r["top1"] for r in e_records)/max(len(e_records),1)
        mean_norm = sum(r["act_norm"] for r in e_records)/max(len(e_records),1)
        mean_std = sum(r["act_std"] for r in e_records)/max(len(e_records),1)
        self.b_w.writerow([beat, self._ms(), ctx_tokens, gen_tokens,
                           round(gen_sec,2), round(gen_tokens/gen_sec,2) if gen_sec>0 else 0,
                           len(self.tok.encode(text)), round(ctx_tokens/8192,4),
                           round(len(self.tok.encode(text))/max(gen_sec,0.001),2),
                           round(mean_ent,6), round(mean_t1,6),
                           round(mean_norm,6), round(mean_std,6), len(self._hook_layers),
                           1 if self.noise else 0])
        self.b_csv.flush()
        out_text = self.tok.decode([r_["tok_id"] for r_ in e_records], skip_special_tokens=True)
        self.msgs.append({"role": "assistant", "content": out_text})
        return out_text

    def _ask(self, q: str, max_new: int = 48, numeric: bool = True) -> str:
        probe = self.msgs + [{"role": "user", "content": self._maybe_noise(q)}]
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
        for beat, text in LEARNING_BEATS[:self.max_beats]:
            print(f"[{time.strftime('%H:%M:%S')}] beat {beat}: 输入({len(text)}字) noise={self.noise}", flush=True)
            r_text = self._gen(text, self.max_new, beat)
            print(f"  回应({len(r_text)}字): {r_text[:50]}...", flush=True)
            reports = {}
            for i, q in enumerate(SELF_REPORT_QUESTIONS):
                ans = self._ask(q, numeric=True)
                reports[f"r{i}"] = ans
                print(f"    自报格{i}: {ans}", flush=True)
            nrec = {"beat": beat, "t_ms": self._ms(), "noise": self.noise}
            for q in NARRATIVE_QUESTIONS:
                ans = self._ask(q, max_new=64, numeric=False)
                key = q.split("'")[1]
                nrec[key] = ans
                print(f"    叙事格: {key[:12]}={ans}", flush=True)
            nrec["report"] = reports
            self.n_file.write(json.dumps(nrec, ensure_ascii=False)+"\n")
            self.n_file.flush()
        # session.json
        sess = {"id": self.ind_id, "model": self.model_path, "seed": "n/a",
                "noise": self.noise, "beats": self.max_beats, "duration_s": time.time()-self.t0}
        json.dump(sess, open(self.sess_dir/"session.json","w",encoding="utf-8"), ensure_ascii=False, indent=2)
        print(f"DONE {self.ind_id} ({time.time()-self.t0:.1f}s)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--beats", type=int, default=7)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--noise", action="store_true", help="启用 B 锚点操控（加背景噪声）")
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    r = P3Runner(args.id, args.model, max_beats=args.beats, noise=args.noise)
    r.load()
    r.run()
