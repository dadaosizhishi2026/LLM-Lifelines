"""
云韶框架·AI特化·P4v2
T(G)引导KV预算分配·长序列验证（1673 tokens）
+ 组合实验：T(G)预算分配 + 温度正则化 联合使用
"""
import sys, os, json, time
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(OUT_DIR, "p4v2_long_combined.json")

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

# 长文本
LONG_TEXT = """The Hamiltonian path problem asks whether a given graph contains a path that visits every vertex exactly once. This is one of the classic NP-complete problems in computer science. For dense random graphs, traditional backtracking search has exponential complexity. However, the spectral lineage model discovers that dense random graphs naturally contain high-dimensional clique structures. By using these cliques as outer embryos to wrap original nodes, the original graph can be compressed into a smaller compressed graph. After solving the Hamiltonian path on the compressed graph and expanding back, the number of virtual edges remains stable at zero to two, with coverage above ninety-nine percent. This discovery means that the Hamiltonian path problem on dense graphs can be solved in polynomial time through dimensional compression. """ * 8 + """In mixture of experts language models, each token is independently routed to a small subset of specialized expert networks through a learned gating mechanism. The routing decisions create a dynamic bipartite graph between tokens and experts that changes with every forward pass. Analyzing this graph structure reveals critical load balancing issues where certain experts become overloaded while others remain underutilized. The algebraic tension framework provides a principled way to diagnose which routing configurations are structurally stable and which are prone to collapse under distribution shift. When the tension of the routing graph exceeds one eighth, the system enters a chaotic phase where no polynomial routing strategy can maintain balanced expert utilization across diverse input distributions. """ * 4

input_ids = tokenizer(LONG_TEXT, return_tensors="pt", truncation=True, max_length=2048)["input_ids"]
n = input_ids.shape[1]
n_layers = len(model.model.layers)
n_heads = model.config.num_attention_heads
log(f"长文本: {n} tokens, {n_layers}层, {n_heads}头")

# ── 1. 完整前向 ──
log("完整前向...")
with torch.no_grad():
    out_full = model(input_ids, output_attentions=True)
    logits_full = out_full.logits
    attns_full = out_full.attentions

shift_labels = input_ids[:, 1:]
loss_full = torch.nn.CrossEntropyLoss()(
    logits_full[:, :-1, :].contiguous().view(-1, logits_full.size(-1)),
    shift_labels.contiguous().view(-1)
)
ppl_full = torch.exp(loss_full).item()
log(f"完整PPL: {ppl_full:.2f}")

# ── 2. 每层T(G) + 重要性 ──
log("\n计算每层T(G)...")
layer_TG = []
layer_importance = []

for li in range(n_layers):
    attn = attns_full[li][0].numpy().mean(axis=0)
    causal = np.tril(attn)
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
    T_G = float(importance.max() / mean_imp - 1) if mean_imp > 1e-10 else 0.0
    layer_TG.append(T_G)
    layer_importance.append(importance)

TG_arr = np.array(layer_TG)
TG_max, TG_min = float(TG_arr.max()), float(TG_arr.min())
log(f"  T(G)范围: [{TG_min:.1f}, {TG_max:.1f}]")

# ── 3. 工具函数 ──
def allocate_budget(ratio_avg, beta, mode="adaptive"):
    budgets = np.zeros(n_layers)
    for li in range(n_layers):
        if mode == "uniform":
            budgets[li] = ratio_avg
        elif mode == "adaptive":
            tg_norm = (layer_TG[li] - TG_min) / (TG_max - TG_min) if TG_max > TG_min else 0.5
            budgets[li] = ratio_avg * (1 + beta * (1 - tg_norm))
    total = budgets.sum()
    target = ratio_avg * n_layers
    budgets = budgets * (target / total)
    return np.clip(budgets, 0.1, 1.0)

def measure_ppl_perlayer(model, input_ids, per_layer_keep, temp_fn=None):
    """逐层mask + 可选温度"""
    with torch.no_grad():
        hidden = model.model.embed_tokens(input_ids)
        seq_len = input_ids.shape[1]
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        position_embeddings = model.model.rotary_emb(hidden, position_ids)
        
        for li in range(n_layers):
            layer = model.model.layers[li]
            keep_idx = per_layer_keep[li]
            
            # 温度
            if temp_fn is not None:
                T = temp_fn(li)
                attn_module = layer.self_attn
                original_scaling = attn_module.scaling
                attn_module.scaling = original_scaling / T
            
            # mask
            mask = torch.full((1, 1, seq_len, seq_len), float('-inf'))
            for i in range(seq_len):
                valid = keep_idx[keep_idx <= i]
                if len(valid) > 0:
                    mask[0, 0, i, valid] = 0.0
            
            layer_out = layer(hidden, position_embeddings=position_embeddings, attention_mask=mask)
            
            if temp_fn is not None:
                attn_module.scaling = original_scaling
            
            if isinstance(layer_out, tuple):
                hidden = layer_out[0]
            else:
                hidden = layer_out
        
        hidden = model.model.norm(hidden)
        logits = model.lm_head(hidden)
    
    loss = torch.nn.CrossEntropyLoss()(
        logits[:, :-1, :].contiguous().view(-1, logits.size(-1)),
        shift_labels.contiguous().view(-1)
    )
    return torch.exp(loss).item()

def build_keep(budgets):
    per_layer_keep = []
    for li in range(n_layers):
        n_keep = max(int(n * budgets[li]), 4)
        imp = layer_importance[li]
        keep = np.sort(np.argsort(-imp)[:n_keep])
        if n - 1 not in keep:
            keep[-1] = n - 1
            keep = np.sort(np.unique(keep))
        per_layer_keep.append(keep)
    return per_layer_keep

# ── 4. 实验A：长序列 T(G)引导 vs 均匀 ──
log(f"\n{'═'*60}")
log("实验A：长序列 T(G)引导 vs 均匀")
log(f"{'═'*60}")

BETA = 0.5
results = {
    "model": "Qwen2.5-0.5B",
    "n_tokens": n,
    "ppl_full": round(ppl_full, 2),
    "layer_TG": [round(t, 2) for t in layer_TG],
    "experiments": []
}

for ratio_avg in [0.90, 0.80, 0.75, 0.70, 0.60, 0.50]:
    for mode_name, mode in [("均匀", "uniform"), ("T(G)引导", "adaptive")]:
        budgets = allocate_budget(ratio_avg, BETA, mode)
        keep = build_keep(budgets)
        ppl = measure_ppl_perlayer(model, input_ids, keep)
        results["experiments"].append({
            "ratio_avg": ratio_avg, "mode": mode_name,
            "ppl": round(ppl, 2), "ppl_ratio": round(ppl/ppl_full, 4),
        })
        log(f"  {ratio_avg*100:.0f}% {mode_name:<8}: PPL={ppl:.2f} (×{ppl/ppl_full:.3f})")

# ── 5. 实验B：组合（T(G)预算 + 温度正则化）──
log(f"\n{'═'*60}")
log("实验B：组合（T(G)预算 + 温度正则化）")
log(f"{'═'*60}")

# 温度策略：T(G)自适应温度（P3验证过的）
def tg_temp(li, alpha_base=1/16):
    tg_norm = (layer_TG[li] - TG_min) / (TG_max - TG_min) if TG_max > TG_min else 0.5
    return 1.0 + alpha_base * (1.0 - tg_norm)

for ratio_avg in [0.80, 0.75, 0.70]:
    # 纯T(G)预算（无温度）
    budgets = allocate_budget(ratio_avg, BETA, "adaptive")
    keep = build_keep(budgets)
    ppl_budget_only = measure_ppl_perlayer(model, input_ids, keep, temp_fn=None)
    
    # T(G)预算 + T(G)温度
    ppl_combined = measure_ppl_perlayer(model, input_ids, keep, temp_fn=tg_temp)
    
    # 均匀预算 + 均匀温度（对照）
    budgets_u = allocate_budget(ratio_avg, BETA, "uniform")
    keep_u = build_keep(budgets_u)
    ppl_uniform_temp = measure_ppl_perlayer(model, input_ids, keep_u, temp_fn=lambda li: 1.0 + 1/16)
    
    results["experiments"].append({
        "ratio_avg": ratio_avg, "mode": "T(G)预算_only",
        "ppl": round(ppl_budget_only, 2), "ppl_ratio": round(ppl_budget_only/ppl_full, 4),
    })
    results["experiments"].append({
        "ratio_avg": ratio_avg, "mode": "T(G)预算+T(G)温度",
        "ppl": round(ppl_combined, 2), "ppl_ratio": round(ppl_combined/ppl_full, 4),
    })
    results["experiments"].append({
        "ratio_avg": ratio_avg, "mode": "均匀预算+均匀温度",
        "ppl": round(ppl_uniform_temp, 2), "ppl_ratio": round(ppl_uniform_temp/ppl_full, 4),
    })
    
    log(f"  {ratio_avg*100:.0f}% T(G)预算only:      PPL={ppl_budget_only:.2f} (×{ppl_budget_only/ppl_full:.3f})")
    log(f"  {ratio_avg*100:.0f}% T(G)预算+T(G)温度:  PPL={ppl_combined:.2f} (×{ppl_combined/ppl_full:.3f})")
    log(f"  {ratio_avg*100:.0f}% 均匀预算+均匀温度:  PPL={ppl_uniform_temp:.2f} (×{ppl_uniform_temp/ppl_full:.3f})")

# ── 6. 汇总 ──
log(f"\n{'═'*60}")
log("汇总")
log(f"{'═'*60}")

# 找每个预算水平下T(G)引导vs均匀的改善
for ratio_avg in [0.90, 0.80, 0.75, 0.70, 0.60, 0.50]:
    u = next((e for e in results["experiments"] if e["ratio_avg"]==ratio_avg and e["mode"]=="均匀"), None)
    a = next((e for e in results["experiments"] if e["ratio_avg"]==ratio_avg and e["mode"]=="T(G)引导"), None)
    if u and a:
        imp = (u["ppl"] - a["ppl"]) / u["ppl"] * 100
        log(f"  {ratio_avg*100:.0f}%: 均匀={u['ppl']}, T(G)={a['ppl']}, 改善={imp:.1f}%")

# ── 7. 保存 ──
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

with open(RESULTS, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2, cls=NpEncoder)
log(f"\n结果保存: {RESULTS}")
log("P4v2完成。")
