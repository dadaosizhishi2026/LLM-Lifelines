"""
云韶框架·AI特化·P1验证
相变动力学：Λ_ord ≥ 7/8 → 有序相，< 7/8 → 混沌相
实验：细粒度扫压缩比（95%→25%），找PPL突然跳变的临界点
预测：存在一个阈值，低于它PPL突然崩溃（不是渐变）
"""
import sys, os, json, time
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(OUT_DIR, "p1_scan.json")

MODEL_PATH = os.path.expanduser(
    "~/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B/snapshots/060db6499f32faf8b98477b0a26969ef7d8b9987"
)

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

log("加载 Qwen2.5-0.5B (CPU, eager)...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, trust_remote_code=True,
    dtype=torch.float32, low_cpu_mem_usage=True,
    attn_implementation="eager",
)
model.eval()

# 用两段不同文本（避免单段过拟合）
TEXTS = [
    """The Hamiltonian path problem asks whether a given graph contains a path that visits every vertex exactly once. This is one of the classic NP-complete problems in computer science. For dense random graphs, traditional backtracking search has exponential complexity. However, the spectral lineage model discovers that dense random graphs naturally contain high-dimensional clique structures. By using these cliques as outer embryos to wrap original nodes, the original graph can be compressed into a smaller compressed graph. After solving the Hamiltonian path on the compressed graph and expanding back, the number of virtual edges remains stable at zero to two, with coverage above ninety-nine percent. This discovery means that the Hamiltonian path problem on dense graphs can be solved in polynomial time through dimensional compression. More importantly, any intelligent system based on dense graph structures, including the attention matrix of Transformers and the communication topology of robots, can achieve order-of-magnitude improvement in information transfer efficiency through the same dimensional compression mechanism. The key insight is that dimensional compression preserves the essential topological invariants while dramatically reducing the computational complexity of path finding algorithms on dense random graphs.""",
    """In mixture of experts language models, each token is independently routed to a small subset of specialized expert networks through a learned gating mechanism. The routing decisions create a dynamic bipartite graph between tokens and experts that changes with every forward pass. Analyzing this graph structure reveals critical load balancing issues where certain experts become overloaded while others remain underutilized. The algebraic tension framework provides a principled way to diagnose which routing configurations are structurally stable and which are prone to collapse under distribution shift. When the tension of the routing graph exceeds one eighth, the system enters a chaotic phase where no polynomial routing strategy can maintain balanced expert utilization across diverse input distributions.""",
]

all_ids = [tokenizer(t, return_tensors="pt", truncation=True, max_length=256)["input_ids"] for t in TEXTS]
log(f"{len(TEXTS)} 段文本, 长度: {[ids.shape[1] for ids in all_ids]}")

# ── 工具 ──
def compute_importance(attentions, n):
    imp = np.zeros(n)
    for li in range(len(attentions)):
        attn = attentions[li][0].numpy().mean(axis=0)
        causal = np.tril(attn)
        imp += causal.sum(axis=0) * causal.sum(axis=1)
    return imp

def build_mask(n, keep_idx):
    mask = torch.full((1, 1, n, n), float('-inf'))
    for i in range(n):
        for j in keep_idx:
            if j <= i:
                mask[0, 0, i, j] = 0.0
    return mask

def measure_ppl(model, input_ids, mask=None):
    with torch.no_grad():
        if mask is not None:
            out = model(input_ids, attention_mask=mask)
        else:
            out = model(input_ids)
        logits = out.logits
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    loss = torch.nn.CrossEntropyLoss()(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    return torch.exp(loss).item()

# ── 扫描 ──
# 细粒度：95%, 90%, 85%, 80%, 75%, 70%, 65%, 60%, 55%, 50%, 45%, 40%, 35%, 30%, 25%
RATIOS = [0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.55, 0.50, 0.45, 0.40, 0.35, 0.30, 0.25]

results = {
    "model": "Qwen2.5-0.5B",
    "ratios": RATIOS,
    "texts": []
}

for ti, input_ids in enumerate(all_ids):
    n = input_ids.shape[1]
    log(f"\n═══ 文本{ti} ({n} tokens) ═══")
    
    # 完整PPL
    ppl_full = measure_ppl(model, input_ids)
    log(f"  完整PPL: {ppl_full:.2f}")
    
    # 拿注意力算重要性（只做一次）
    with torch.no_grad():
        out_attn = model(input_ids, output_attentions=True)
        attentions = out_attn.attentions
    importance = compute_importance(attentions, n)
    
    text_results = {
        "n_tokens": n,
        "ppl_full": round(ppl_full, 2),
        "sweep": []
    }
    
    prev_ppl = ppl_full
    prev_ratio = None
    jump_detected = False
    jump_ratio = None
    
    for ratio in RATIOS:
        n_keep = max(int(n * ratio), 4)
        keep_idx = np.sort(np.argsort(-importance)[:n_keep])
        if n - 1 not in keep_idx:
            keep_idx[-1] = n - 1
            keep_idx = np.sort(np.unique(keep_idx))
        
        mask = build_mask(n, keep_idx)
        ppl = measure_ppl(model, input_ids, mask)
        
        ppl_ratio = ppl / ppl_full
        delta = ppl - prev_ppl
        delta_pct = (ppl / prev_ppl - 1) * 100 if prev_ppl > 0 else 0
        
        # 相变检测：PPL突然跳变（相邻两步增幅>50%）
        if not jump_detected and delta_pct > 50 and prev_ratio is not None:
            jump_detected = True
            jump_ratio = ratio
            log(f"  🔴 相变检测！{prev_ratio*100:.0f}%→{ratio*100:.0f}%: PPL {prev_ppl:.1f}→{ppl:.1f} (+{delta_pct:.0f}%)")
        
        sweep_point = {
            "keep_ratio": ratio,
            "n_keep": len(keep_idx),
            "ppl": round(ppl, 2),
            "ppl_ratio": round(ppl_ratio, 4),
            "delta_from_prev": round(delta, 2),
            "delta_pct": round(delta_pct, 1),
        }
        text_results["sweep"].append(sweep_point)
        
        marker = "🔴" if (jump_detected and ratio == jump_ratio) else "  "
        log(f"  {marker} {ratio*100:5.1f}%: PPL={ppl:7.2f} (×{ppl_ratio:.3f}, Δ={delta_pct:+.1f}%)")
        
        prev_ppl = ppl
        prev_ratio = ratio
    
    text_results["jump_detected"] = jump_detected
    text_results["jump_ratio"] = jump_ratio
    results["texts"].append(text_results)

# ── 汇总 ──
log("\n═══ 汇总 ═══")
for ti, tr in enumerate(results["texts"]):
    if tr["jump_detected"]:
        log(f"  文本{ti}: 相变在 {tr['jump_ratio']*100:.0f}% 保留处")
    else:
        log(f"  文本{ti}: 未检测到明显相变（渐变衰减）")

# 理论预测：7/8 = 87.5% 是有序/混沌相变点
# 如果相变在85%-90%之间，与7/8=87.5%吻合
theory_threshold = 7/8  # 0.875
log(f"\n  理论相变阈值: {theory_threshold*100:.1f}% (= 7/8 = η)")

# ── 保存 ──
with open(RESULTS, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
log(f"\n结果保存: {RESULTS}")
log("P1相变阈值扫描完成。")
