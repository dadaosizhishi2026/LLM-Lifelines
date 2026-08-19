"""
云韶框架·AI特化·P4
T(G)引导逐层KV Cache预算分配
- 高T(G)层（真实解相，结构集中）→ 可以压缩更多（重要token明显）
- 低T(G)层（虚拟/无解相，分布均匀）→ 需要保留更多（没有明显重要token）
- 对比：均匀预算 vs T(G)引导预算（总预算相同）
"""
import sys, os, json, time
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(OUT_DIR, "p4_kv_budget.json")

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

# ── 2. 计算每层T(G)和每层重要性 ──
log("\n计算每层T(G) + 逐层token重要性...")
layer_TG = []
layer_importance = []  # 每层的token重要性（入度×出度）

for li in range(n_layers):
    attn = attns_full[li][0].numpy().mean(axis=0)  # (n, n)
    causal = np.tril(attn)
    
    # T(G)
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
    T_G = importance.max() / mean_imp - 1 if mean_imp > 1e-10 else 0
    
    layer_TG.append(float(T_G))
    layer_importance.append(importance)
    
    phase = "real" if T_G > 8 else ("virtual" if T_G > 1/8 else "unsolvable")
    log(f"  层{li:2d}: T(G)={T_G:8.2f} → {phase}")

# ── 3. 预算分配策略 ──
# 总预算：每层平均保留 ratio_avg 的token
# 均匀：每层都保留 ratio_avg
# T(G)引导：高T(G)层保留少（结构集中，少数token就够），低T(G)层保留多
# 公式：budget_li = ratio_avg × (1 + β × (1 - TG_norm_li))
# β控制分配强度，TG_norm归一化到[0,1]

TG_arr = np.array(layer_TG)
TG_max, TG_min = TG_arr.max(), TG_arr.min()

def allocate_budget(ratio_avg, beta, mode="adaptive"):
    """返回每层保留比例"""
    budgets = np.zeros(n_layers)
    for li in range(n_layers):
        if mode == "uniform":
            budgets[li] = ratio_avg
        elif mode == "adaptive":
            # 高T(G) → 少保留（结构集中）
            if TG_max > TG_min:
                tg_norm = (layer_TG[li] - TG_min) / (TG_max - TG_min)
            else:
                tg_norm = 0.5
            budgets[li] = ratio_avg * (1 + beta * (1 - tg_norm))
        elif mode == "inverse":
            # 反向：高T(G) → 多保留（故意做错）
            if TG_max > TG_min:
                tg_norm = (layer_TG[li] - TG_min) / (TG_max - TG_min)
            else:
                tg_norm = 0.5
            budgets[li] = ratio_avg * (1 + beta * tg_norm)
    
    # 归一化：确保总预算 = ratio_avg × n_layers
    total_budget = budgets.sum()
    target_total = ratio_avg * n_layers
    budgets = budgets * (target_total / total_budget)
    # 裁剪到[0.1, 1.0]
    budgets = np.clip(budgets, 0.1, 1.0)
    return budgets

# ── 4. 逐层KV压缩 + PPL测量 ──
def measure_ppl_with_perlayer_mask(model, input_ids, per_layer_keep_idx):
    """对每层用不同的keep_idx做mask，测PPL"""
    # 需要逐层forward，每层用不同的attention mask
    with torch.no_grad():
        hidden = model.model.embed_tokens(input_ids)
        seq_len = input_ids.shape[1]
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        position_embeddings = model.model.rotary_emb(hidden, position_ids)
        
        for li in range(n_layers):
            layer = model.model.layers[li]
            keep_idx = per_layer_keep_idx[li]
            
            # 构建该层的4D mask
            mask = torch.full((1, 1, seq_len, seq_len), float('-inf'))
            for i in range(seq_len):
                valid = keep_idx[keep_idx <= i]
                if len(valid) > 0:
                    mask[0, 0, i, valid] = 0.0
            
            layer_out = layer(hidden, position_embeddings=position_embeddings, attention_mask=mask)
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

# ── 5. 实验 ──
RATIO_AVG = 0.75  # 总预算：平均保留75%
BETA = 0.5  # 分配强度

results = {
    "model": "Qwen2.5-0.5B",
    "n_tokens": n,
    "n_layers": n_layers,
    "ppl_full": round(ppl_full, 2),
    "ratio_avg": RATIO_AVG,
    "beta": BETA,
    "layer_TG": [round(t, 2) for t in layer_TG],
    "experiments": []
}

for mode_name, mode in [("均匀", "uniform"), ("T(G)引导", "adaptive"), ("反向", "inverse")]:
    log(f"\n═══ {mode_name} (总预算={RATIO_AVG*100:.0f}%) ═══")
    
    budgets = allocate_budget(RATIO_AVG, BETA, mode)
    
    # 每层的keep_idx
    per_layer_keep = []
    for li in range(n_layers):
        n_keep = max(int(n * budgets[li]), 4)
        imp = layer_importance[li]
        keep = np.sort(np.argsort(-imp)[:n_keep])
        if n - 1 not in keep:
            keep[-1] = n - 1
            keep = np.sort(np.unique(keep))
        per_layer_keep.append(keep)
    
    # 统计
    actual_ratios = [len(k)/n for k in per_layer_keep]
    log(f"  每层保留: min={min(actual_ratios):.2f}, max={max(actual_ratios):.2f}, "
        f"mean={np.mean(actual_ratios):.2f}")
    
    # 测PPL
    ppl = measure_ppl_with_perlayer_mask(model, input_ids, per_layer_keep)
    ppl_ratio = ppl / ppl_full
    
    results["experiments"].append({
        "mode": mode_name,
        "ppl": round(ppl, 2),
        "ppl_ratio": round(ppl_ratio, 4),
        "ppl_change_pct": round((ppl_ratio - 1) * 100, 1),
        "budget_range": [round(float(min(actual_ratios)), 3), round(float(max(actual_ratios)), 3)],
    })
    log(f"  PPL: {ppl:.2f} (×{ppl_ratio:.3f}, {(ppl_ratio-1)*100:+.1f}%)")

# ── 6. 多预算水平 ──
log(f"\n{'═'*60}")
log("多预算水平对比（T(G)引导 vs 均匀）")
log(f"{'═'*60}")

multi_budget = []
for ratio_avg in [0.90, 0.80, 0.70, 0.60, 0.50]:
    for mode_name, mode in [("均匀", "uniform"), ("T(G)引导", "adaptive")]:
        budgets = allocate_budget(ratio_avg, BETA, mode)
        per_layer_keep = []
        for li in range(n_layers):
            n_keep = max(int(n * budgets[li]), 4)
            imp = layer_importance[li]
            keep = np.sort(np.argsort(-imp)[:n_keep])
            if n - 1 not in keep:
                keep[-1] = n - 1
                keep = np.sort(np.unique(keep))
            per_layer_keep.append(keep)
        
        ppl = measure_ppl_with_perlayer_mask(model, input_ids, per_layer_keep)
        multi_budget.append({
            "ratio_avg": ratio_avg,
            "mode": mode_name,
            "ppl": round(ppl, 2),
            "ppl_ratio": round(ppl / ppl_full, 4),
        })
        log(f"  {ratio_avg*100:.0f}% {mode_name:<8}: PPL={ppl:.2f} (×{ppl/ppl_full:.3f})")

results["multi_budget"] = multi_budget

# ── 7. 核心验证 ──
log(f"\n{'═'*60}")
log("核心验证")
log(f"{'═'*60}")

uniform_75 = next(e for e in results["experiments"] if e["mode"] == "均匀")
adaptive_75 = next(e for e in results["experiments"] if e["mode"] == "T(G)引导")
inverse_75 = next(e for e in results["experiments"] if e["mode"] == "反向")

log(f"  75%预算: 均匀={uniform_75['ppl']}, T(G)引导={adaptive_75['ppl']}, 反向={inverse_75['ppl']}")

if adaptive_75["ppl"] < uniform_75["ppl"]:
    improvement = (uniform_75["ppl"] - adaptive_75["ppl"]) / uniform_75["ppl"] * 100
    log(f"  ✅ T(G)引导优于均匀 ({improvement:.1f}%改善)")
else:
    log(f"  ⚠️ T(G)引导不优于均匀")

if adaptive_75["ppl"] < inverse_75["ppl"]:
    log(f"  ✅ T(G)引导优于反向（方向正确）")
else:
    log(f"  ❌ T(G)引导不优于反向")

# ── 8. 保存 ──
# 修复float32序列化
def convert(obj):
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        c = convert(obj)
        if c is not obj:
            return c
        return super().default(obj)

with open(RESULTS, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2, cls=NpEncoder)
log(f"\n结果保存: {RESULTS}")
log("P4 T(G)引导KV预算分配验证完成。")
