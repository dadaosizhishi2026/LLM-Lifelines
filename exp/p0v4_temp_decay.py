"""
云韶框架·AI特化·P0v4
逐步代谢（正确版）：注意力温度缩放
每层attention logits除以T_k = (8/7)^k，使注意力分布逐层变平
理论：温度升高 → 分布更均匀 → 信息量减少 → 等效R(k)=(7/8)^k
对比：T=1（完整）vs T_k=(8/7)^k（累积）vs T=8/7（每层固定）
"""
import sys, os, json, time
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(OUT_DIR, "p0v4_temp_decay.json")

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

# 完整前向
log("完整前向...")
with torch.no_grad():
    out_full = model(input_ids, output_attentions=True)
    logits_full = out_full.logits[0]
    attns_full = out_full.attentions  # 每层 (1, heads, n, n)

shift_labels = input_ids[:, 1:]
loss_full = torch.nn.CrossEntropyLoss()(
    logits_full[:-1].unsqueeze(0).contiguous().view(-1, logits_full.size(-1)),
    shift_labels.contiguous().view(-1)
)
ppl_full = torch.exp(loss_full).item()
log(f"完整PPL: {ppl_full:.2f}")

# 计算每层注意力的熵（基线）
layer_entropy_full = []
for li in range(n_layers):
    attn = attns_full[li][0]  # (heads, n, n)
    # 每行熵
    entropy = -(attn * (attn + 1e-10).log()).sum(dim=-1)  # (heads, n)
    layer_entropy_full.append(entropy.mean().item())

ETA = 7/8
ALPHA = 1/8

results = {
    "model": "Qwen2.5-0.5B",
    "n_tokens": n,
    "n_layers": n_layers,
    "eta": ETA,
    "ppl_full": round(ppl_full, 2),
    "layer_entropy_full": [round(e, 4) for e in layer_entropy_full],
    "experiments": []
}

# ── 温度缩放实验 ──
# 通过hook修改attention计算：在softmax之前除以温度T
# Qwen2的self_attn.forward中，attn_weights = matmul(query, key.T) / sqrt(head_dim)
# 我们在attn_weights上再除以T（等效于把sqrt(head_dim)变成sqrt(head_dim)*T）
# 实现：hook在self_attn上，拦截输出中的attn_weights不行（已经softmax了）
# 正确方法：monkey-patch forward，在softmax前插入温度

# 更简单的方法：直接修改model的attention实现
# 用torch的register_forward_pre_hook不行（拿不到中间attn_logits）
# 最可靠：手动逐层forward

def manual_forward_with_temp(model, input_ids, temp_fn):
    """手动逐层forward，在每层attention的softmax前除以温度"""
    with torch.no_grad():
        # Embedding
        hidden = model.model.embed_tokens(input_ids)
        
        seq_len = input_ids.shape[1]
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        
        # 预计算RoPE position_embeddings（Qwen2在model层算好传给每层）
        position_embeddings = model.model.rotary_emb(hidden, position_ids)
        
        # 逐层
        for li in range(n_layers):
            layer = model.model.layers[li]
            T = temp_fn(li)
            
            attn_module = layer.self_attn
            original_scaling = attn_module.scaling
            attn_module.scaling = original_scaling / T  # 除以T = 温度升高
            
            layer_out = layer(
                hidden,
                position_embeddings=position_embeddings,
                attention_mask=None,
            )
            
            attn_module.scaling = original_scaling
            
            if isinstance(layer_out, tuple):
                hidden = layer_out[0]
            else:
                hidden = layer_out
        
        # Final norm + LM head
        hidden = model.model.norm(hidden)
        logits = model.lm_head(hidden)
        
    return logits[0]  # (n, vocab)

# 方案A：累积温度 T_k = (8/7)^k
log("\n═══ 方案A：累积温度 T_k = (8/7)^k ═══")
log(f"  第0层T=1.0, 第23层T=(8/7)^23={((8/7)**23):.3f}")
logits_A = manual_forward_with_temp(model, input_ids, lambda k: (8/7) ** k)
loss_A = torch.nn.CrossEntropyLoss()(
    logits_A[:-1].unsqueeze(0).contiguous().view(-1, logits_A.size(-1)),
    shift_labels.contiguous().view(-1)
)
ppl_A = torch.exp(loss_A).item()
kl_A = F.kl_div(
    F.softmax(logits_A, dim=-1).log(),
    F.softmax(logits_full, dim=-1),
    reduction='batchmean'
).item()
log(f"  PPL: {ppl_full:.2f} → {ppl_A:.2f} (×{ppl_A/ppl_full:.3f}, +{(ppl_A/ppl_full-1)*100:.1f}%)")
log(f"  KL: {kl_A:.4f}")

# 方案B：每层固定温度 T = 8/7（不累积）
log("\n═══ 方案B：每层固定温度 T = 8/7 ═══")
logits_B = manual_forward_with_temp(model, input_ids, lambda k: 8/7)
loss_B = torch.nn.CrossEntropyLoss()(
    logits_B[:-1].unsqueeze(0).contiguous().view(-1, logits_B.size(-1)),
    shift_labels.contiguous().view(-1)
)
ppl_B = torch.exp(loss_B).item()
kl_B = F.kl_div(
    F.softmax(logits_B, dim=-1).log(),
    F.softmax(logits_full, dim=-1),
    reduction='batchmean'
).item()
log(f"  PPL: {ppl_full:.2f} → {ppl_B:.2f} (×{ppl_B/ppl_full:.3f}, +{(ppl_B/ppl_full-1)*100:.1f}%)")
log(f"  KL: {kl_B:.4f}")

# 方案C：轻微累积 T_k = (16/15)^k（α=1/16对照）
log("\n═══ 方案C：轻微累积 T_k = (16/15)^k ═══")
logits_C = manual_forward_with_temp(model, input_ids, lambda k: (16/15) ** k)
loss_C = torch.nn.CrossEntropyLoss()(
    logits_C[:-1].unsqueeze(0).contiguous().view(-1, logits_C.size(-1)),
    shift_labels.contiguous().view(-1)
)
ppl_C = torch.exp(loss_C).item()
kl_C = F.kl_div(
    F.softmax(logits_C, dim=-1).log(),
    F.softmax(logits_full, dim=-1),
    reduction='batchmean'
).item()
log(f"  PPL: {ppl_full:.2f} → {ppl_C:.2f} (×{ppl_C/ppl_full:.3f}, +{(ppl_C/ppl_full-1)*100:.1f}%)")
log(f"  KL: {kl_C:.4f}")

# 方案D：极轻微 T_k = (32/31)^k（α=1/32对照）
log("\n═══ 方案D：极轻微 T_k = (32/31)^k ═══")
logits_D = manual_forward_with_temp(model, input_ids, lambda k: (32/31) ** k)
loss_D = torch.nn.CrossEntropyLoss()(
    logits_D[:-1].unsqueeze(0).contiguous().view(-1, logits_D.size(-1)),
    shift_labels.contiguous().view(-1)
)
ppl_D = torch.exp(loss_D).item()
kl_D = F.kl_div(
    F.softmax(logits_D, dim=-1).log(),
    F.softmax(logits_full, dim=-1),
    reduction='batchmean'
).item()
log(f"  PPL: {ppl_full:.2f} → {ppl_D:.2f} (×{ppl_D/ppl_full:.3f}, +{(ppl_D/ppl_full-1)*100:.1f}%)")
log(f"  KL: {kl_D:.4f}")

# 方案E：只后半层累积 T_k = (8/7)^(k-12) for k>=12
log("\n═══ 方案E：后半层累积 T_k = (8/7)^(k-12) ═══")
logits_E = manual_forward_with_temp(model, input_ids, lambda k: 1.0 if k < n_layers//2 else (8/7) ** (k - n_layers//2))
loss_E = torch.nn.CrossEntropyLoss()(
    logits_E[:-1].unsqueeze(0).contiguous().view(-1, logits_E.size(-1)),
    shift_labels.contiguous().view(-1)
)
ppl_E = torch.exp(loss_E).item()
kl_E = F.kl_div(
    F.softmax(logits_E, dim=-1).log(),
    F.softmax(logits_full, dim=-1),
    reduction='batchmean'
).item()
log(f"  PPL: {ppl_full:.2f} → {ppl_E:.2f} (×{ppl_E/ppl_full:.3f}, +{(ppl_E/ppl_full-1)*100:.1f}%)")
log(f"  KL: {kl_E:.4f}")

# ── 核心验证：KL是否与α线性相关 ──
log("\n═══ 核心验证：KL ∝ α ? ═══")
# α_A = 1/8, α_C = 1/16, α_D = 1/32
# 如果KL ∝ α，则 KL_A/KL_C ≈ 2, KL_C/KL_D ≈ 2
ratio_AC = kl_A / max(kl_C, 1e-8)
ratio_CD = kl_C / max(kl_D, 1e-8)
log(f"  KL_A/KL_C = {ratio_AC:.2f} (理论: α_A/α_C = 2.0)")
log(f"  KL_C/KL_D = {ratio_CD:.2f} (理论: α_C/α_D = 2.0)")

if 1.2 < ratio_AC < 3.5 and 1.2 < ratio_CD < 3.5:
    log(f"  ✅ KL与α近似线性（衰减动力学成立）")
else:
    log(f"  ⚠️ KL与α非线性")

# ── 逐层KL分析（方案A）──
log("\n═══ 方案A逐层KL ═══")
# 重新跑方案A，但逐层记录logit差异
# 简化：用完整模型逐层forward vs 温度模型逐层forward
# 太慢，改用：比较每层attention熵的变化

# 跑方案A拿attention
log("  跑方案A拿attention...")
# 需要修改manual_forward来收集attention
def manual_forward_collect_attn(model, input_ids, temp_fn):
    with torch.no_grad():
        hidden = model.model.embed_tokens(input_ids)
        seq_len = input_ids.shape[1]
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        position_embeddings = model.model.rotary_emb(hidden, position_ids)
        
        layer_attns = []
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
                if len(layer_out) > 1 and layer_out[1] is not None:
                    layer_attns.append(layer_out[1])
            else:
                hidden = layer_out
        
        hidden = model.model.norm(hidden)
        logits = model.lm_head(hidden)
    return logits[0], layer_attns

# Qwen2默认不返回attention，需要output_attentions=True
# 但手动forward时layer不返回attn...用另一种方式
# 直接比较完整模型和温度模型的最终logit逐位置KL
log("  逐位置KL（方案A vs 完整）...")
kl_per_pos_A = []
for pos in range(n):
    p = F.softmax(logits_full[pos], dim=-1)
    q = F.softmax(logits_A[pos], dim=-1)
    kl = F.kl_div(q.log(), p, reduction='sum').item()
    kl_per_pos_A.append(kl)

# 前10个位置和后10个位置的KL对比
log(f"  前10位置平均KL: {np.mean(kl_per_pos_A[:10]):.4f}")
log(f"  后10位置平均KL: {np.mean(kl_per_pos_A[-10:]):.4f}")
log(f"  中间位置平均KL: {np.mean(kl_per_pos_A[10:-10]):.4f}")

# ── 汇总 ──
log("\n═══ 最终汇总 ═══")
log(f"  {'方案':<35} {'PPL':>10} {'倍率':>8} {'KL':>8} {'α':>6}")
log(f"  {'完整':<35} {ppl_full:>10.2f} {'×1.000':>8} {'0':>8} {'0':>6}")
log(f"  {'A:(8/7)^k累积':<35} {ppl_A:>10.2f} {'×'+f'{ppl_A/ppl_full:.3f}':>8} {kl_A:>8.4f} {'1/8':>6}")
log(f"  {'B:每层T=8/7':<35} {ppl_B:>10.2f} {'×'+f'{ppl_B/ppl_full:.3f}':>8} {kl_B:>8.4f} {'1/8':>6}")
log(f"  {'C:(16/15)^k':<35} {ppl_C:>10.2f} {'×'+f'{ppl_C/ppl_full:.3f}':>8} {kl_C:>8.4f} {'1/16':>6}")
log(f"  {'D:(32/31)^k':<35} {ppl_D:>10.2f} {'×'+f'{ppl_D/ppl_full:.3f}':>8} {kl_D:>8.4f} {'1/32':>6}")
log(f"  {'E:后半层(8/7)^k':<35} {ppl_E:>10.2f} {'×'+f'{ppl_E/ppl_full:.3f}':>8} {kl_E:>8.4f} {'1/8':>6}")

results["experiments"] = [
    {"scheme": "A_(8/7)^k", "ppl": round(ppl_A, 2), "ratio": round(ppl_A/ppl_full, 4), "kl": round(kl_A, 4), "alpha": "1/8"},
    {"scheme": "B_per_layer_8/7", "ppl": round(ppl_B, 2), "ratio": round(ppl_B/ppl_full, 4), "kl": round(kl_B, 4), "alpha": "1/8"},
    {"scheme": "C_(16/15)^k", "ppl": round(ppl_C, 2), "ratio": round(ppl_C/ppl_full, 4), "kl": round(kl_C, 4), "alpha": "1/16"},
    {"scheme": "D_(32/31)^k", "ppl": round(ppl_D, 2), "ratio": round(ppl_D/ppl_full, 4), "kl": round(kl_D, 4), "alpha": "1/32"},
    {"scheme": "E_back_half_(8/7)^k", "ppl": round(ppl_E, 2), "ratio": round(ppl_E/ppl_full, 4), "kl": round(kl_E, 4), "alpha": "1/8"},
]
results["linearity_test"] = {
    "kl_ratio_AC": round(ratio_AC, 4),
    "kl_ratio_CD": round(ratio_CD, 4),
    "theory_ratio": 2.0,
    "verdict": "linear" if 1.2 < ratio_AC < 3.5 and 1.2 < ratio_CD < 3.5 else "nonlinear"
}
results["kl_per_pos_A"] = [round(k, 4) for k in kl_per_pos_A]

with open(RESULTS, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
log(f"\n结果保存: {RESULTS}")
log("P0v4温度衰减验证完成。")
