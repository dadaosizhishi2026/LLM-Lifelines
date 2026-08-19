"""
云韶框架·AI特化·P0v3修正
逐步代谢：对每层输出的hidden states乘以η^k
这才是真正的R(k) = R₀·(7/8)^k
"""
import sys, os, json, time
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(OUT_DIR, "p0v3_stepwise.json")

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

TEXT = """The Hamiltonian path problem asks whether a given graph contains a path that visits every vertex exactly once. This is one of the classic NP-complete problems in computer science. For dense random graphs, traditional backtracking search has exponential complexity. However, the spectral lineage model discovers that dense random graphs naturally contain high-dimensional clique structures. By using these cliques as outer embryos to wrap original nodes, the original graph can be compressed into a smaller compressed graph. After solving the Hamiltonian path on the compressed graph and expanding back, the number of virtual edges remains stable at zero to two, with coverage above ninety-nine percent. This discovery means that the Hamiltonian path problem on dense graphs can be solved in polynomial time through dimensional compression. More importantly, any intelligent system based on dense graph structures, including the attention matrix of Transformers and the communication topology of robots, can achieve order-of-magnitude improvement in information transfer efficiency through the same dimensional compression mechanism. The key insight is that dimensional compression preserves the essential topological invariants while dramatically reducing the computational complexity of path finding algorithms on dense random graphs."""

input_ids = tokenizer(TEXT, return_tensors="pt")["input_ids"]
n = input_ids.shape[1]
n_layers = len(model.model.layers)
log(f"文本: {n} tokens, 模型: {n_layers}层")

# 完整前向
log("完整前向...")
with torch.no_grad():
    out_full = model(input_ids)
    logits_full = out_full.logits[0]

shift_logits = logits_full[:-1].unsqueeze(0)
shift_labels = input_ids[:, 1:]
loss_full = torch.nn.CrossEntropyLoss()(shift_logits.contiguous().view(-1, shift_logits.size(-1)), shift_labels.contiguous().view(-1))
ppl_full = torch.exp(loss_full).item()
log(f"完整PPL: {ppl_full:.2f}")

ETA = 7/8
ALPHA = 1/8

results = {
    "model": "Qwen2.5-0.5B",
    "n_tokens": n,
    "n_layers": n_layers,
    "eta": ETA,
    "ppl_full": round(ppl_full, 2),
    "experiments": []
}

# 方案：对每层输出的hidden states做衰减
# hook在layer输出上，缩放hidden states
# 方案A：累积衰减 hidden *= η^k（k=层序号）
# 方案B：每层固定衰减 hidden *= η（不累积，每层独立损失1/8）
# 方案C：只衰减后半层
# 方案D：衰减attention输出（不衰减FFN输出）

def run_with_layer_decay(model, input_ids, decay_fn):
    """对每层输出hidden states乘以decay_fn(layer_idx)"""
    hooks = []
    
    def make_hook(layer_idx):
        def hook_fn(module, input, output):
            # output是hidden_states（或tuple）
            if isinstance(output, tuple):
                hidden = output[0]
                decay = decay_fn(layer_idx)
                hidden_decayed = hidden * decay
                return (hidden_decayed,) + output[1:]
            else:
                return output * decay_fn(layer_idx)
        return hook_fn
    
    for li in range(n_layers):
        h = model.model.layers[li].register_forward_hook(make_hook(li))
        hooks.append(h)
    
    with torch.no_grad():
        out = model(input_ids)
        logits = out.logits[0]
    
    for h in hooks:
        h.remove()
    
    return logits

# 方案A：累积衰减 η^k
log("\n═══ 方案A：累积衰减 hidden *= (7/8)^k ═══")
logits_A = run_with_layer_decay(model, input_ids, lambda k: ETA ** k)
loss_A = torch.nn.CrossEntropyLoss()(logits_A[:-1].unsqueeze(0).contiguous().view(-1, logits_A.size(-1)), shift_labels.contiguous().view(-1))
ppl_A = torch.exp(loss_A).item()
kl_A = F.kl_div(F.softmax(logits_A, dim=-1).log(), F.softmax(logits_full, dim=-1), reduction='batchmean').item()
log(f"  PPL: {ppl_full:.2f} → {ppl_A:.2f} (×{ppl_A/ppl_full:.3f})")
log(f"  理论最终保留: (7/8)^23 = {ETA**23:.6f}")

# 方案B：每层固定衰减（不累积）
log("\n═══ 方案B：每层固定衰减 hidden *= 7/8 ═══")
logits_B = run_with_layer_decay(model, input_ids, lambda k: ETA)
loss_B = torch.nn.CrossEntropyLoss()(logits_B[:-1].unsqueeze(0).contiguous().view(-1, logits_B.size(-1)), shift_labels.contiguous().view(-1))
ppl_B = torch.exp(loss_B).item()
kl_B = F.kl_div(F.softmax(logits_B, dim=-1).log(), F.softmax(logits_full, dim=-1), reduction='batchmean').item()
log(f"  PPL: {ppl_full:.2f} → {ppl_B:.2f} (×{ppl_B/ppl_full:.3f})")

# 方案C：只衰减后半层（前12层不动）
log("\n═══ 方案C：后半层累积衰减 ═══")
logits_C = run_with_layer_decay(model, input_ids, lambda k: 1.0 if k < n_layers//2 else ETA ** (k - n_layers//2))
loss_C = torch.nn.CrossEntropyLoss()(logits_C[:-1].unsqueeze(0).contiguous().view(-1, logits_C.size(-1)), shift_labels.contiguous().view(-1))
ppl_C = torch.exp(loss_C).item()
kl_C = F.kl_div(F.softmax(logits_C, dim=-1).log(), F.softmax(logits_full, dim=-1), reduction='batchmean').item()
log(f"  PPL: {ppl_full:.2f} → {ppl_C:.2f} (×{ppl_C/ppl_full:.3f})")

# 方案D：轻微衰减（η=15/16，只损失1/16）作为对照
ETA_MILD = 15/16
log(f"\n═══ 方案D：轻微衰减 hidden *= (15/16)^k ═══")
logits_D = run_with_layer_decay(model, input_ids, lambda k: ETA_MILD ** k)
loss_D = torch.nn.CrossEntropyLoss()(logits_D[:-1].unsqueeze(0).contiguous().view(-1, logits_D.size(-1)), shift_labels.contiguous().view(-1))
ppl_D = torch.exp(loss_D).item()
kl_D = F.kl_div(F.softmax(logits_D, dim=-1).log(), F.softmax(logits_full, dim=-1), reduction='batchmean').item()
log(f"  PPL: {ppl_full:.2f} → {ppl_D:.2f} (×{ppl_D/ppl_full:.3f})")
log(f"  理论最终保留: (15/16)^23 = {ETA_MILD**23:.6f}")

# 方案E：极轻微衰减（η=31/32）
ETA_TINY = 31/32
log(f"\n═══ 方案E：极轻微衰减 hidden *= (31/32)^k ═══")
logits_E = run_with_layer_decay(model, input_ids, lambda k: ETA_TINY ** k)
loss_E = torch.nn.CrossEntropyLoss()(logits_E[:-1].unsqueeze(0).contiguous().view(-1, logits_E.size(-1)), shift_labels.contiguous().view(-1))
ppl_E = torch.exp(loss_E).item()
kl_E = F.kl_div(F.softmax(logits_E, dim=-1).log(), F.softmax(logits_full, dim=-1), reduction='batchmean').item()
log(f"  PPL: {ppl_full:.2f} → {ppl_E:.2f} (×{ppl_E/ppl_full:.3f})")
log(f"  理论最终保留: (31/32)^23 = {ETA_TINY**23:.6f}")

# 汇总
log("\n═══ 汇总 ═══")
log(f"  {'方案':<30} {'PPL':>8} {'倍率':>8} {'最终保留':>10}")
log(f"  {'完整':<30} {ppl_full:>8.2f} {'×1.000':>8} {'1.000':>10}")
log(f"  {'A:(7/8)^k累积':<30} {ppl_A:>8.2f} {'×'+f'{ppl_A/ppl_full:.3f}':>8} {ETA**23:>10.6f}")
log(f"  {'B:每层×7/8':<30} {ppl_B:>8.2f} {'×'+f'{ppl_B/ppl_full:.3f}':>8} {ETA**23:>10.6f}")
log(f"  {'C:后半层(7/8)^k':<30} {ppl_C:>8.2f} {'×'+f'{ppl_C/ppl_full:.3f}':>8} {ETA**11:>10.6f}")
log(f"  {'D:(15/16)^k':<30} {ppl_D:>8.2f} {'×'+f'{ppl_D/ppl_full:.3f}':>8} {ETA_MILD**23:>10.6f}")
log(f"  {'E:(31/32)^k':<30} {ppl_E:>8.2f} {'×'+f'{ppl_E/ppl_full:.3f}':>8} {ETA_TINY**23:>10.6f}")

# 核心验证：PPL增幅是否与α成正比？
# 如果R(k)=(1-α)^k，那么α越大PPL增幅越大
# α=1/8 → 方案A
# α=1/16 → 方案D
# α=1/32 → 方案E
# 如果PPL增幅 ∝ α，则 (ppl_A-1)/(ppl_D-1) ≈ (1/8)/(1/16) = 2
ratio_AD = (ppl_A/ppl_full - 1) / max(ppl_D/ppl_full - 1, 1e-6)
ratio_DE = (ppl_D/ppl_full - 1) / max(ppl_E/ppl_full - 1, 1e-6)
log(f"\n  PPL增幅比 A/D = {ratio_AD:.2f} (理论: α_A/α_D = 2.0)")
log(f"  PPL增幅比 D/E = {ratio_DE:.2f} (理论: α_D/α_E = 2.0)")

results["experiments"] = [
    {"scheme": "A_(7/8)^k", "ppl": round(ppl_A, 2), "ratio": round(ppl_A/ppl_full, 4), "final_retention": round(ETA**23, 6), "alpha": "1/8"},
    {"scheme": "B_per_layer_7/8", "ppl": round(ppl_B, 2), "ratio": round(ppl_B/ppl_full, 4), "final_retention": round(ETA**23, 6), "alpha": "1/8"},
    {"scheme": "C_back_half_(7/8)^k", "ppl": round(ppl_C, 2), "ratio": round(ppl_C/ppl_full, 4), "final_retention": round(ETA**11, 6), "alpha": "1/8"},
    {"scheme": "D_(15/16)^k", "ppl": round(ppl_D, 2), "ratio": round(ppl_D/ppl_full, 4), "final_retention": round(ETA_MILD**23, 6), "alpha": "1/16"},
    {"scheme": "E_(31/32)^k", "ppl": round(ppl_E, 2), "ratio": round(ppl_E/ppl_full, 4), "final_retention": round(ETA_TINY**23, 6), "alpha": "1/32"},
]
results["linearity_test"] = {
    "ratio_AD": round(ratio_AD, 4),
    "ratio_DE": round(ratio_DE, 4),
    "theory_ratio": 2.0,
    "verdict": "linear" if 1.0 < ratio_AD < 4.0 and 1.0 < ratio_DE < 4.0 else "nonlinear"
}

with open(RESULTS, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
log(f"\n结果保存: {RESULTS}")
log("P0v3逐步代谢验证完成。")
