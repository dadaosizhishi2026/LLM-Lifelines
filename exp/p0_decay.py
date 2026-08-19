"""
云韶框架·AI特化·P0验证
衰减动力学：R(k) = R₀·(7/8)^k
实验：KV压缩后逐token生成，测每步信息保留率是否按(7/8)^k衰减
"""
import sys, os, json, time
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(OUT_DIR, "p0_decay_results.json")

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

# 用一段较长的文本做prefill
TEXT = """The Hamiltonian path problem asks whether a given graph contains a path that visits every vertex exactly once. This is one of the classic NP-complete problems in computer science. For dense random graphs, traditional backtracking search has exponential complexity. However, the spectral lineage model discovers that dense random graphs naturally contain high-dimensional clique structures. By using these cliques as outer embryos to wrap original nodes, the original graph can be compressed into a smaller compressed graph. After solving the Hamiltonian path on the compressed graph and expanding back, the number of virtual edges remains stable at zero to two, with coverage above ninety-nine percent. This discovery means that the Hamiltonian path problem on dense graphs can be solved in polynomial time through dimensional compression. More importantly, any intelligent system based on dense graph structures, including the attention matrix of Transformers and the communication topology of robots, can achieve order-of-magnitude improvement in information transfer efficiency through the same dimensional compression mechanism. The key insight is that dimensional compression preserves the essential topological invariants while dramatically reducing the computational complexity of path finding algorithms."""

input_ids = tokenizer(TEXT, return_tensors="pt")["input_ids"]
n = input_ids.shape[1]
log(f"Prefill: {n} tokens")

GEN_STEPS = 30

# ── 1. 完整前向拿注意力+重要性 ──
with torch.no_grad():
    out_full = model(input_ids, output_attentions=True)
    attentions = out_full.attentions
    logits_full = out_full.logits

# 全局重要性
global_imp = np.zeros(n)
for li in range(len(attentions)):
    attn = attentions[li][0].numpy().mean(axis=0)
    causal = np.tril(attn)
    global_imp += causal.sum(axis=0) * causal.sum(axis=1)

# ── 2. 完整生成（greedy） ──
log("完整生成...")
with torch.no_grad():
    gen_full = model.generate(input_ids, max_new_tokens=GEN_STEPS, do_sample=False)
full_tokens = gen_full[0, n:].tolist()
log(f"  生成 {len(full_tokens)} tokens: '{tokenizer.decode(full_tokens)[:80]}...'")

# ── 3. 压缩生成 + 逐token KL测量 ──
def build_4d_mask(n_total, keep_idx):
    """4D causal mask"""
    mask = torch.full((1, 1, n_total, n_total), float('-inf'))
    for i in range(n_total):
        for j in keep_idx:
            if j <= i:
                mask[0, 0, i, j] = 0.0
    return mask

results = {
    "model": "Qwen2.5-0.5B",
    "n_prefill": n,
    "gen_steps": GEN_STEPS,
    "theory_decay": "R(k) = (7/8)^k",
    "theory_log_slope": round(float(np.log(8/7)), 6),  # ≈0.1335
    "experiments": []
}

for keep_ratio in [0.75, 0.50]:
    n_keep = max(int(n * keep_ratio), 8)
    keep_idx = np.sort(np.argsort(-global_imp)[:n_keep])
    if n - 1 not in keep_idx:
        keep_idx[-1] = n - 1
        keep_idx = np.sort(np.unique(keep_idx))
    
    log(f"\n═══ 保留 {keep_ratio*100:.0f}% ({len(keep_idx)}/{n}) ═══")
    
    # 构建mask
    mask = build_4d_mask(n, keep_idx)
    
    # 逐token生成，每步测KL
    current_ids = input_ids.clone()
    current_mask = mask.clone()
    
    kl_per_step = []
    retention_per_step = []  # exp(-KL) = 信息保留率
    top1_agree_per_step = []
    
    for step in range(GEN_STEPS):
        # 完整模型在这一步的logit
        with torch.no_grad():
            out_f = model(current_ids)
            logit_f = out_f.logits[0, -1, :]
        
        # 压缩模型在这一步的logit
        # 扩展mask到新长度
        cur_n = current_ids.shape[1]
        ext_mask = torch.full((1, 1, cur_n, cur_n), float('-inf'))
        # 原始mask部分
        ext_mask[:, :, :n, :n] = mask
        # 新生成的token可以attend到所有保留位置+之前的新token
        for i in range(n, cur_n):
            for j in keep_idx:
                ext_mask[0, 0, i, j] = 0.0
            for j in range(n, i + 1):
                ext_mask[0, 0, i, j] = 0.0
        # 原始位置也可以attend到新token（简化：允许）
        for i in range(n):
            for j in range(n, cur_n):
                ext_mask[0, 0, i, j] = 0.0
        
        with torch.no_grad():
            out_c = model(current_ids, attention_mask=ext_mask)
            logit_c = out_c.logits[0, -1, :]
        
        # KL散度
        p = F.softmax(logit_f, dim=-1)
        q = F.softmax(logit_c, dim=-1)
        kl = F.kl_div(q.log(), p, reduction='sum').item()
        kl_per_step.append(kl)
        
        # 信息保留率 = exp(-KL)（近似）
        retention = np.exp(-kl)
        retention_per_step.append(retention)
        
        # top-1一致性
        agree = 1.0 if logit_f.argmax() == logit_c.argmax() else 0.0
        top1_agree_per_step.append(agree)
        
        # 用完整模型的token继续（保证对比公平）
        next_token = logit_f.argmax().unsqueeze(0).unsqueeze(0)
        current_ids = torch.cat([current_ids, next_token], dim=1)
        
        if step < 5 or step % 5 == 0:
            theory_ret = (7/8) ** (step + 1)
            log(f"  step {step+1:2d}: KL={kl:.4f}, 保留={retention:.4f}, "
                f"理论(7/8)^{step+1}={theory_ret:.4f}, top1={'✓' if agree else '✗'}")
    
    # 拟合：log(retention) vs step 应该是线性的，斜率 = log(7/8) ≈ -0.1335
    steps = np.arange(1, GEN_STEPS + 1)
    log_retention = np.log(np.array(retention_per_step) + 1e-10)
    
    # 线性拟合
    valid = np.isfinite(log_retention)
    if valid.sum() > 2:
        slope, intercept = np.polyfit(steps[valid], log_retention[valid], 1)
    else:
        slope, intercept = 0, 0
    
    theory_slope = np.log(7/8)  # ≈ -0.1335
    
    exp_result = {
        "keep_ratio": keep_ratio,
        "n_keep": len(keep_idx),
        "kl_per_step": [round(k, 4) for k in kl_per_step],
        "retention_per_step": [round(r, 4) for r in retention_per_step],
        "top1_agree_per_step": top1_agree_per_step,
        "fitted_slope": round(float(slope), 6),
        "theory_slope": round(float(theory_slope), 6),
        "slope_ratio": round(float(slope / theory_slope), 4) if theory_slope != 0 else None,
        "mean_retention": round(float(np.mean(retention_per_step)), 4),
        "mean_top1_agree": round(float(np.mean(top1_agree_per_step)), 4),
    }
    results["experiments"].append(exp_result)
    
    log(f"\n  拟合斜率: {slope:.4f} (理论: {theory_slope:.4f}, 比值: {slope/theory_slope:.2f})")
    log(f"  平均保留率: {np.mean(retention_per_step):.4f}")
    log(f"  平均top1一致: {np.mean(top1_agree_per_step):.1%}")
    
    # 判定
    ratio = slope / theory_slope
    if 0.5 < ratio < 2.0:
        log(f"  ✅ 衰减曲线与(7/8)^k量级一致（比值{ratio:.2f}在0.5-2.0内）")
    elif ratio > 2.0:
        log(f"  ⚠️ 衰减比理论快（比值{ratio:.2f}>2），压缩损伤大于α=1/8预测")
    else:
        log(f"  ⚠️ 衰减比理论慢（比值{ratio:.2f}<0.5），可能有其他补偿机制")

# ── 4. 保存 ──
with open(RESULTS, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
log(f"\n结果保存: {RESULTS}")
log("P0衰减动力学验证完成。")
