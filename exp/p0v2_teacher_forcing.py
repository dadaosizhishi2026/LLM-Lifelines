"""
云韶框架·AI特化·P0v2
衰减动力学（teacher forcing版）
不给模型自己生成的token，喂真实文本，逐位置测压缩vs完整的KL
k = 距离最后一个保留token的步数
"""
import sys, os, json, time
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(OUT_DIR, "p0v2_teacher_forcing.json")

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
log(f"文本: {n} tokens")

# ── 1. 完整前向（teacher forcing：一次forward拿到所有位置的logit） ──
log("完整前向...")
with torch.no_grad():
    out_full = model(input_ids, output_attentions=True)
    logits_full = out_full.logits[0]  # (n, vocab)
    attentions = out_full.attentions

# 全局重要性
global_imp = np.zeros(n)
for li in range(len(attentions)):
    attn = attentions[li][0].numpy().mean(axis=0)
    causal = np.tril(attn)
    global_imp += causal.sum(axis=0) * causal.sum(axis=1)

# ── 2. 对每个位置，计算"距离最后一个保留token的步数k" ──
# 然后按k分组，看KL是否按(7/8)^k衰减

results = {
    "model": "Qwen2.5-0.5B",
    "n_tokens": n,
    "theory": "KL(k) should increase as retention R(k)=(7/8)^k decreases",
    "experiments": []
}

for keep_ratio in [0.90, 0.75, 0.50]:
    n_keep = max(int(n * keep_ratio), 8)
    keep_idx = np.sort(np.argsort(-global_imp)[:n_keep])
    if n - 1 not in keep_idx:
        keep_idx[-1] = n - 1
        keep_idx = np.sort(np.unique(keep_idx))
    keep_set = set(keep_idx.tolist())
    
    log(f"\n═══ 保留 {keep_ratio*100:.0f}% ({len(keep_idx)}/{n}) ═══")
    
    # 构建4D mask
    mask = torch.full((1, 1, n, n), float('-inf'))
    for i in range(n):
        for j in keep_idx:
            if j <= i:
                mask[0, 0, i, j] = 0.0
    
    # 压缩前向（teacher forcing：同一个input_ids，只是mask不同）
    with torch.no_grad():
        out_comp = model(input_ids, attention_mask=mask)
        logits_comp = out_comp.logits[0]  # (n, vocab)
    
    # 逐位置KL
    kl_per_pos = []
    for pos in range(n):
        p = F.softmax(logits_full[pos], dim=-1)
        q = F.softmax(logits_comp[pos], dim=-1)
        kl = F.kl_div(q.log(), p, reduction='sum').item()
        kl_per_pos.append(kl)
    
    # 计算每个位置的k（距离最后一个保留token的步数）
    k_per_pos = []
    for pos in range(n):
        # 找≤pos的保留token中最大的
        kept_before = [j for j in keep_idx if j <= pos]
        if kept_before:
            k = pos - max(kept_before)
        else:
            k = pos + 1  # 前面没有保留token
        k_per_pos.append(k)
    
    # 按k分组统计KL
    max_k = max(k_per_pos)
    kl_by_k = {}
    for k_val in range(max_k + 1):
        positions = [i for i, kv in enumerate(k_per_pos) if kv == k_val]
        if positions:
            kl_by_k[k_val] = {
                "mean_kl": round(float(np.mean([kl_per_pos[i] for i in positions])), 4),
                "n_positions": len(positions),
            }
    
    # 理论：R(k) = (7/8)^k → KL(k) ≈ -log(R(k)) = k * log(8/7) ≈ 0.1335k
    # 即KL应该随k线性增长，斜率≈0.1335
    # 拟合：KL vs k
    ks = np.array(k_per_pos, dtype=float)
    kls = np.array(kl_per_pos)
    valid = (kls < 20) & np.isfinite(kls)  # 排除极端值
    if valid.sum() > 5:
        slope, intercept = np.polyfit(ks[valid], kls[valid], 1)
    else:
        slope, intercept = 0, 0
    
    theory_slope = np.log(8/7)  # ≈ 0.1335
    
    # 分段统计
    k_bins = [(0, 0), (1, 2), (3, 5), (6, 10), (11, 20), (21, 50)]
    bin_stats = []
    for k_lo, k_hi in k_bins:
        positions = [i for i, kv in enumerate(k_per_pos) if k_lo <= kv <= k_hi]
        if positions:
            mean_kl = float(np.mean([kl_per_pos[i] for i in positions]))
            mean_ret = float(np.mean([np.exp(-kl_per_pos[i]) for i in positions]))
            theory_ret = float((7/8) ** ((k_lo + k_hi) / 2))
            bin_stats.append({
                "k_range": f"{k_lo}-{k_hi}",
                "n_pos": len(positions),
                "mean_kl": round(mean_kl, 4),
                "mean_retention": round(mean_ret, 4),
                "theory_retention": round(theory_ret, 4),
            })
            log(f"  k={k_lo:2d}-{k_hi:2d}: KL={mean_kl:.4f}, 保留={mean_ret:.4f}, "
                f"理论(7/8)^{(k_lo+k_hi)//2}={theory_ret:.4f}, n={len(positions)}")
    
    exp_result = {
        "keep_ratio": keep_ratio,
        "n_keep": len(keep_idx),
        "fitted_slope": round(float(slope), 6),
        "theory_slope": round(float(theory_slope), 6),
        "slope_ratio": round(float(slope / theory_slope), 4),
        "kl_by_k": kl_by_k,
        "bin_stats": bin_stats,
        "overall_mean_kl": round(float(np.mean(kl_per_pos)), 4),
        "overall_mean_retention": round(float(np.mean([np.exp(-k) for k in kl_per_pos])), 4),
    }
    results["experiments"].append(exp_result)
    
    log(f"\n  拟合: KL = {slope:.4f}×k + {intercept:.4f}")
    log(f"  理论斜率: {theory_slope:.4f}, 比值: {slope/theory_slope:.2f}")
    
    ratio = slope / theory_slope
    if 0.3 < ratio < 3.0:
        log(f"  ✅ KL随k增长，量级与(7/8)^k衰减一致（比值{ratio:.2f}）")
    elif ratio > 3.0:
        log(f"  ⚠️ 衰减比理论快（比值{ratio:.2f}），压缩损伤>α预测")
    else:
        log(f"  ⚠️ KL不随k增长（比值{ratio:.2f}），衰减模型不适用")

# ── 3. 保存 ──
with open(RESULTS, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
log(f"\n结果保存: {RESULTS}")
log("P0v2 teacher forcing验证完成。")
