"""
云韶框架·AI特化·P2
实验1：长序列验证（2K+ tokens）——T(G)三态+相变是否仍成立
实验2：正则化最优点——扫α找PPL最低的"代谢率"
"""
import sys, os, json, time
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(OUT_DIR, "p2_long_and_reg.json")

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

# ══════════════════════════════════════════════════════════
# 实验1：长序列（~2000 tokens）
# ══════════════════════════════════════════════════════════
log("\n" + "═"*60)
log("实验1：长序列验证（~2000 tokens）")
log("═"*60)

# 构造一段长文本（重复+变化，模拟真实长文档）
LONG_TEXT = """The Hamiltonian path problem asks whether a given graph contains a path that visits every vertex exactly once. This is one of the classic NP-complete problems in computer science. For dense random graphs, traditional backtracking search has exponential complexity. However, the spectral lineage model discovers that dense random graphs naturally contain high-dimensional clique structures. By using these cliques as outer embryos to wrap original nodes, the original graph can be compressed into a smaller compressed graph. After solving the Hamiltonian path on the compressed graph and expanding back, the number of virtual edges remains stable at zero to two, with coverage above ninety-nine percent. This discovery means that the Hamiltonian path problem on dense graphs can be solved in polynomial time through dimensional compression. """ * 8 + """In mixture of experts language models, each token is independently routed to a small subset of specialized expert networks through a learned gating mechanism. The routing decisions create a dynamic bipartite graph between tokens and experts that changes with every forward pass. Analyzing this graph structure reveals critical load balancing issues where certain experts become overloaded while others remain underutilized. The algebraic tension framework provides a principled way to diagnose which routing configurations are structurally stable and which are prone to collapse under distribution shift. When the tension of the routing graph exceeds one eighth, the system enters a chaotic phase where no polynomial routing strategy can maintain balanced expert utilization across diverse input distributions. """ * 4

input_ids_long = tokenizer(LONG_TEXT, return_tensors="pt", truncation=True, max_length=2048)["input_ids"]
n_long = input_ids_long.shape[1]
log(f"长文本: {n_long} tokens")

# 完整前向
log("完整前向（长序列）...")
with torch.no_grad():
    out_long = model(input_ids_long, output_attentions=True)
    logits_long = out_long.logits
    attns_long = out_long.attentions

shift_labels_long = input_ids_long[:, 1:]
loss_long = torch.nn.CrossEntropyLoss()(
    logits_long[:, :-1, :].contiguous().view(-1, logits_long.size(-1)),
    shift_labels_long.contiguous().view(-1)
)
ppl_long_full = torch.exp(loss_long).item()
log(f"完整PPL: {ppl_long_full:.2f}")

# T(G)三态分析
n_layers = len(attns_long)
n_heads = attns_long[0].shape[1]
log(f"\nT(G)三态分析 ({n_layers}层 × {n_heads}头 = {n_layers*n_heads}个注意力头)...")

phase_counts = {"real": 0, "virtual": 0, "unsolvable": 0}
phase_per_layer = []

for li in range(n_layers):
    layer_phases = {"real": 0, "virtual": 0, "unsolvable": 0}
    for hi in range(n_heads):
        attn = attns_long[li][0, hi].numpy()  # (n, n)
        # top-k建图（k=64或n//4取小）
        k = min(64, n_long // 4)
        # 对每行取top-k
        topk_idx = np.argpartition(-attn, k, axis=1)[:, :k]
        # 建邻接矩阵
        adj = np.zeros((n_long, n_long), dtype=np.float32)
        for i in range(n_long):
            for j in topk_idx[i]:
                adj[i, j] = attn[i, j]
        
        # 入度×出度
        in_deg = adj.sum(axis=0)
        out_deg = adj.sum(axis=1)
        importance = in_deg * out_deg
        
        # T(G) = max(importance) / mean(importance) - 1
        mean_imp = importance.mean()
        if mean_imp > 1e-10:
            T_G = importance.max() / mean_imp - 1
        else:
            T_G = 0
        
        # 三态判定
        alpha = 1/8
        if T_G > 1/alpha:  # T(G) > 8 → 真实解相
            phase = "real"
        elif T_G > alpha:  # 1/8 < T(G) < 8 → 虚拟解相
            phase = "virtual"
        else:  # T(G) < 1/8 → 无解相
            phase = "unsolvable"
        
        phase_counts[phase] += 1
        layer_phases[phase] += 1
    
    phase_per_layer.append(layer_phases)

total_heads = n_layers * n_heads
log(f"\n  三态分布（{total_heads}个头）:")
log(f"    真实解相: {phase_counts['real']:4d} ({phase_counts['real']/total_heads*100:.1f}%)")
log(f"    虚拟解相: {phase_counts['virtual']:4d} ({phase_counts['virtual']/total_heads*100:.1f}%)")
log(f"    无解相:   {phase_counts['unsolvable']:4d} ({phase_counts['unsolvable']/total_heads*100:.1f}%)")

# 相变扫描（长序列）
log("\n相变扫描（长序列）...")
# 全局重要性
global_imp_long = np.zeros(n_long)
for li in range(n_layers):
    attn = attns_long[li][0].numpy().mean(axis=0)
    causal = np.tril(attn)
    global_imp_long += causal.sum(axis=0) * causal.sum(axis=1)

def build_mask_fast(n, keep_idx):
    """向量化mask构建"""
    mask = torch.full((1, 1, n, n), float('-inf'))
    keep_set = torch.tensor(keep_idx, dtype=torch.long)
    # 对每个位置i，允许attend到keep_idx中≤i的位置
    for i in range(n):
        valid = keep_set[keep_set <= i]
        if len(valid) > 0:
            mask[0, 0, i, valid] = 0.0
    return mask

RATIOS = [0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.50, 0.40, 0.30]
sweep_long = []

for ratio in RATIOS:
    n_keep = max(int(n_long * ratio), 4)
    keep_idx = np.sort(np.argsort(-global_imp_long)[:n_keep])
    if n_long - 1 not in keep_idx:
        keep_idx[-1] = n_long - 1
        keep_idx = np.sort(np.unique(keep_idx))
    
    mask = build_mask_fast(n_long, keep_idx)
    with torch.no_grad():
        out_comp = model(input_ids_long, attention_mask=mask)
        logits_comp = out_comp.logits
    
    loss_comp = torch.nn.CrossEntropyLoss()(
        logits_comp[:, :-1, :].contiguous().view(-1, logits_comp.size(-1)),
        shift_labels_long.contiguous().view(-1)
    )
    ppl_comp = torch.exp(loss_comp).item()
    ppl_ratio = ppl_comp / ppl_long_full
    
    sweep_long.append({
        "keep_ratio": ratio,
        "n_keep": len(keep_idx),
        "ppl": round(ppl_comp, 2),
        "ppl_ratio": round(ppl_ratio, 4),
    })
    log(f"  {ratio*100:5.1f}%: PPL={ppl_comp:8.2f} (×{ppl_ratio:.3f})")

# ══════════════════════════════════════════════════════════
# 实验2：正则化最优点（短序列，快）
# ══════════════════════════════════════════════════════════
log("\n" + "═"*60)
log("实验2：正则化最优点扫描")
log("═"*60)

TEXT_SHORT = """The Hamiltonian path problem asks whether a given graph contains a path that visits every vertex exactly once. This is one of the classic NP-complete problems in computer science. For dense random graphs, traditional backtracking search has exponential complexity. However, the spectral lineage model discovers that dense random graphs naturally contain high-dimensional clique structures. By using these cliques as outer embryos to wrap original nodes, the original graph can be compressed into a smaller compressed graph. After solving the Hamiltonian path on the compressed graph and expanding back, the number of virtual edges remains stable at zero to two, with coverage above ninety-nine percent. This discovery means that the Hamiltonian path problem on dense graphs can be solved in polynomial time through dimensional compression. More importantly, any intelligent system based on dense graph structures, including the attention matrix of Transformers and the communication topology of robots, can achieve order-of-magnitude improvement in information transfer efficiency through the same dimensional compression mechanism. The key insight is that dimensional compression preserves the essential topological invariants while dramatically reducing the computational complexity of path finding algorithms on dense random graphs."""

input_ids_short = tokenizer(TEXT_SHORT, return_tensors="pt")["input_ids"]
n_short = input_ids_short.shape[1]

with torch.no_grad():
    out_short = model(input_ids_short)
    logits_short_full = out_short.logits[0]

shift_labels_short = input_ids_short[:, 1:]
loss_short_full = torch.nn.CrossEntropyLoss()(
    logits_short_full[:-1].unsqueeze(0).contiguous().view(-1, logits_short_full.size(-1)),
    shift_labels_short.contiguous().view(-1)
)
ppl_short_full = torch.exp(loss_short_full).item()
log(f"短文本基线PPL: {ppl_short_full:.2f} ({n_short} tokens)")

def manual_forward_temp(model, input_ids, temp_fn, n_layers):
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

# 扫描α：从1/128到1/4
ALPHAS = [1/128, 1/64, 1/32, 1/16, 1/8, 3/16, 1/4]
reg_results = []

log(f"\n扫描α（每层固定温度T=1/(1-α)，不累积）:")
for alpha in ALPHAS:
    T = 1.0 / (1.0 - alpha)
    logits_t = manual_forward_temp(model, input_ids_short, lambda k, T=T: T, n_layers)
    loss_t = torch.nn.CrossEntropyLoss()(
        logits_t[:-1].unsqueeze(0).contiguous().view(-1, logits_t.size(-1)),
        shift_labels_short.contiguous().view(-1)
    )
    ppl_t = torch.exp(loss_t).item()
    ppl_ratio = ppl_t / ppl_short_full
    
    reg_results.append({
        "alpha": round(alpha, 6),
        "alpha_frac": f"1/{int(1/alpha)}" if alpha == 1/int(1/alpha) else f"{alpha:.4f}",
        "temperature": round(T, 4),
        "ppl": round(ppl_t, 2),
        "ppl_ratio": round(ppl_ratio, 4),
        "ppl_change_pct": round((ppl_ratio - 1) * 100, 1),
    })
    marker = "🟢" if ppl_ratio < 1.0 else "  "
    log(f"  {marker} α={alpha:.5f} (T={T:.4f}): PPL={ppl_t:.2f} (×{ppl_ratio:.3f}, {(ppl_ratio-1)*100:+.1f}%)")

# 找最优α
best = min(reg_results, key=lambda x: x["ppl"])
log(f"\n  🏆 最优α = {best['alpha_frac']} (T={best['temperature']}), PPL={best['ppl']} (×{best['ppl_ratio']:.3f})")
log(f"  理论α = 1/8 = 0.125")
log(f"  最优α vs 理论α: {'吻合' if abs(best['alpha'] - 1/8) < 0.05 else '偏离'}")

# ── 保存 ──
results = {
    "experiment1_long_sequence": {
        "n_tokens": n_long,
        "ppl_full": round(ppl_long_full, 2),
        "phase_distribution": phase_counts,
        "phase_pct": {k: round(v/total_heads*100, 1) for k, v in phase_counts.items()},
        "phase_transition_sweep": sweep_long,
    },
    "experiment2_regularization": {
        "n_tokens": n_short,
        "ppl_full": round(ppl_short_full, 2),
        "sweep": reg_results,
        "best_alpha": best,
    }
}

with open(RESULTS, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
log(f"\n结果保存: {RESULTS}")
log("P2完成。")
