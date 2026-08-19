"""
云韶框架·AI特化·P3
T(G)自适应温度：用每层注意力图的代数张力决定该层正则化强度
- 高T(G)（真实解相，结构清晰）→ 少正则化（T≈1）
- 低T(G)（无解相，随机）→ 多正则化（T>1）
- 对比：均匀温度 vs T(G)自适应温度
这是框架独有的能力：H2O/StreamingLLM没有"诊断"能力
"""
import sys, os, json, time
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(OUT_DIR, "p3_adaptive_temp.json")

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
n_heads = model.config.num_attention_heads
log(f"文本: {n} tokens, {n_layers}层, {n_heads}头")

# ── 1. 完整前向拿注意力 ──
log("完整前向...")
with torch.no_grad():
    out_full = model(input_ids, output_attentions=True)
    logits_full = out_full.logits[0]
    attns_full = out_full.attentions

shift_labels = input_ids[:, 1:]
loss_full = torch.nn.CrossEntropyLoss()(
    logits_full[:-1].unsqueeze(0).contiguous().view(-1, logits_full.size(-1)),
    shift_labels.contiguous().view(-1)
)
ppl_full = torch.exp(loss_full).item()
log(f"完整PPL: {ppl_full:.2f}")

# ── 2. 计算每层T(G) ──
log("\n计算每层T(G)...")
layer_TG = []
layer_phase = []

for li in range(n_layers):
    # 平均所有头的注意力
    attn = attns_full[li][0].numpy().mean(axis=0)  # (n, n)
    causal = np.tril(attn)
    
    # top-k建图
    k = min(64, n // 4)
    topk_idx = np.argpartition(-causal, k, axis=1)[:, :k]
    adj = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in topk_idx[i]:
            adj[i, j] = causal[i, j]
    
    in_deg = adj.sum(axis=0)
    out_deg = adj.sum(axis=1)
    importance = in_deg * out_deg
    mean_imp = importance.mean()
    
    if mean_imp > 1e-10:
        T_G = importance.max() / mean_imp - 1
    else:
        T_G = 0
    
    # 三态
    alpha = 1/8
    if T_G > 1/alpha:
        phase = "real"
    elif T_G > alpha:
        phase = "virtual"
    else:
        phase = "unsolvable"
    
    layer_TG.append(T_G)
    layer_phase.append(phase)
    log(f"  层{li:2d}: T(G)={T_G:8.2f} → {phase}")

# ── 3. 设计自适应温度策略 ──
# 策略：T(G)越高（结构越清晰）→ 越不需要正则化 → T越接近1
# T(G)越低（越随机）→ 越需要正则化 → T越大
# 公式：T_layer = 1 + α × (1 - T_G_normalized)
# 其中T_G_normalized = min(T_G / T_G_max, 1)

TG_max = max(layer_TG)
TG_min = min(layer_TG)

def adaptive_temp(li, alpha_base=1/8):
    """T(G)自适应温度"""
    # 归一化T(G)到[0,1]
    if TG_max > TG_min:
        tg_norm = (layer_TG[li] - TG_min) / (TG_max - TG_min)
    else:
        tg_norm = 0.5
    # T(G)高 → tg_norm高 → 温度低（少正则化）
    # T(G)低 → tg_norm低 → 温度高（多正则化）
    T = 1.0 + alpha_base * (1.0 - tg_norm)
    return T

def uniform_temp(li, alpha_base=1/8):
    """均匀温度（对照）"""
    return 1.0 + alpha_base

def inverse_temp(li, alpha_base=1/8):
    """反向策略（对照）：T(G)高→多正则化（故意做错）"""
    if TG_max > TG_min:
        tg_norm = (layer_TG[li] - TG_min) / (TG_max - TG_min)
    else:
        tg_norm = 0.5
    T = 1.0 + alpha_base * tg_norm  # 反过来了
    return T

def phase_temp(li, alpha_base=1/8):
    """相变策略：只对无解相/虚拟解相层正则化，真实解相层不动"""
    if layer_phase[li] == "real":
        return 1.0  # 不动
    elif layer_phase[li] == "virtual":
        return 1.0 + alpha_base * 0.5  # 轻微
    else:  # unsolvable
        return 1.0 + alpha_base  # 完整正则化

# ── 4. 手动forward with per-layer temperature ──
def manual_forward_perlayer(model, input_ids, temp_fn, n_layers):
    with torch.no_grad():
        hidden = model.model.embed_tokens(input_ids)
        seq_len = input_ids.shape[1]
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        position_embeddings = model.model.rotary_emb(hidden, position_ids)
        
        for li in range(n_layers):
            layer = model.model.layers[li]
            T = temp_fn(li)
            attn_module = layer.self_attn
            original_scaling = attn_module.scaling
            attn_module.scaling = original_scaling / T
            layer_out = layer(hidden, position_embeddings=position_embeddings, attention_mask=None)
            attn_module.scaling = original_scaling
            if isinstance(layer_out, tuple):
                hidden = layer_out[0]
            else:
                hidden = layer_out
        
        hidden = model.model.norm(hidden)
        logits = model.lm_head(hidden)
    return logits[0]

# ── 5. 跑所有策略 ──
strategies = [
    ("均匀_α=1/8", uniform_temp),
    ("T(G)自适应_α=1/8", adaptive_temp),
    ("反向_α=1/8", inverse_temp),
    ("相变策略", phase_temp),
]

results = {
    "model": "Qwen2.5-0.5B",
    "n_tokens": n,
    "n_layers": n_layers,
    "ppl_full": round(ppl_full, 2),
    "layer_TG": [round(t, 2) for t in layer_TG],
    "layer_phase": layer_phase,
    "experiments": []
}

log(f"\n{'═'*60}")
log("策略对比")
log(f"{'═'*60}")

for name, fn in strategies:
    logits_s = manual_forward_perlayer(model, input_ids, fn, n_layers)
    loss_s = torch.nn.CrossEntropyLoss()(
        logits_s[:-1].unsqueeze(0).contiguous().view(-1, logits_s.size(-1)),
        shift_labels.contiguous().view(-1)
    )
    ppl_s = torch.exp(loss_s).item()
    kl_s = F.kl_div(
        F.softmax(logits_s, dim=-1).log(),
        F.softmax(logits_full, dim=-1),
        reduction='batchmean'
    ).item()
    
    # 逐层温度
    temps = [round(fn(li), 4) for li in range(n_layers)]
    
    results["experiments"].append({
        "strategy": name,
        "ppl": round(ppl_s, 2),
        "ppl_ratio": round(ppl_s / ppl_full, 4),
        "ppl_change_pct": round((ppl_s / ppl_full - 1) * 100, 1),
        "kl": round(kl_s, 4),
        "temps": temps,
    })
    
    marker = "🏆" if ppl_s == min(ppl_full, ppl_s) else "  "
    log(f"  {marker} {name:<25}: PPL={ppl_s:.2f} (×{ppl_s/ppl_full:.3f}, {(ppl_s/ppl_full-1)*100:+.1f}%), KL={kl_s:.4f}")

# ── 6. 核心验证 ──
log(f"\n{'═'*60}")
log("核心验证")
log(f"{'═'*60}")

# 找PPL最低的策略
best = min(results["experiments"], key=lambda x: x["ppl"])
log(f"  最优策略: {best['strategy']} (PPL={best['ppl']})")

# 自适应 vs 均匀
adaptive_ppl = next(e for e in results["experiments"] if "自适应" in e["strategy"])["ppl"]
uniform_ppl = next(e for e in results["experiments"] if "均匀" in e["strategy"])["ppl"]
inverse_ppl = next(e for e in results["experiments"] if "反向" in e["strategy"])["ppl"]

log(f"\n  自适应 vs 均匀: {adaptive_ppl:.2f} vs {uniform_ppl:.2f} (差{(adaptive_ppl-uniform_ppl)/uniform_ppl*100:+.1f}%)")
log(f"  自适应 vs 反向: {adaptive_ppl:.2f} vs {inverse_ppl:.2f} (差{(adaptive_ppl-inverse_ppl)/inverse_ppl*100:+.1f}%)")

if adaptive_ppl < uniform_ppl:
    log(f"  ✅ T(G)自适应优于均匀（框架诊断能力有价值）")
else:
    log(f"  ⚠️ T(G)自适应不优于均匀（诊断能力在此场景无额外收益）")

if adaptive_ppl < inverse_ppl:
    log(f"  ✅ 自适应优于反向（T(G)方向正确：高结构→少正则化）")
else:
    log(f"  ❌ 自适应不优于反向（T(G)方向可能错误）")

# ── 7. 保存 ──
with open(RESULTS, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
log(f"\n结果保存: {RESULTS}")
log("P3 T(G)自适应温度验证完成。")
