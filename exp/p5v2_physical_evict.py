"""
云韶框架·AI特化·P5v2（修正版）
物理KV删除 + 真实生成
不用mask屏蔽，而是真正从KV Cache中删除不重要的token（缩短序列）
T(G)决定每层删哪些、删多少
"""
import sys, os, json, time
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(OUT_DIR, "p5v2_physical_evict.json")

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

PROMPT = """The Hamiltonian path problem asks whether a given graph contains a path that visits every vertex exactly once. This is one of the classic NP-complete problems in computer science. For dense random graphs, traditional backtracking search has exponential complexity. However, the spectral lineage model discovers that dense random graphs naturally contain high-dimensional clique structures. By using these cliques as outer embryos to wrap original nodes, the original graph can be compressed into a smaller compressed graph. After solving the Hamiltonian path on the compressed graph and expanding back, the number of virtual edges remains stable at zero to two, with coverage above ninety-nine percent. This discovery means that the Hamiltonian path problem on dense graphs can be solved in polynomial time through dimensional compression. Now, the implications for artificial intelligence are profound because the attention mechanism in Transformers can be viewed as a dense graph where each token attends to every previous token."""

input_ids = tokenizer(PROMPT, return_tensors="pt")["input_ids"]
n = input_ids.shape[1]
n_layers = len(model.model.layers)
n_heads = model.config.num_attention_heads
head_dim = model.config.hidden_size // n_heads
log(f"Prompt: {n} tokens, {n_layers}层, {n_heads}头, head_dim={head_dim}")

GEN_TOKENS = 60

# ── 1. 完整前向拿注意力+重要性 ──
log("完整前向...")
with torch.no_grad():
    out_full = model(input_ids, output_attentions=True)
    attns_full = out_full.attentions

per_layer_imp = []
layer_TG = []
for li in range(n_layers):
    attn = attns_full[li][0].numpy().mean(axis=0)
    causal = np.tril(attn)
    imp = causal.sum(axis=0) * causal.sum(axis=1)
    per_layer_imp.append(imp)
    
    k = min(64, n // 4)
    topk_idx = np.argpartition(-causal, k, axis=1)[:, :k]
    adj = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in topk_idx[i]:
            adj[i, j] = causal[i, j]
    in_deg = adj.sum(axis=0)
    out_deg = adj.sum(axis=1)
    imp2 = in_deg * out_deg
    mean_imp = imp2.mean()
    T_G = float(imp2.max() / mean_imp - 1) if mean_imp > 1e-10 else 0.0
    layer_TG.append(T_G)

TG_arr = np.array(layer_TG)
TG_max, TG_min = float(TG_arr.max()), float(TG_arr.min())
log(f"T(G)范围: [{TG_min:.1f}, {TG_max:.1f}]")

# ── 2. 物理KV删除生成 ──
# 方法：prefill后，对每层的KV Cache物理删除不重要的token
# 然后从缩短的KV Cache继续生成
# 关键：所有层用同一个keep_idx（因为生成时token位置必须对齐）
# 但T(G)决定keep_idx的选择策略

def generate_with_eviction(model, tokenizer, input_ids, keep_idx_global, max_new=GEN_TOKENS):
    """
    物理KV删除 + 自回归生成
    keep_idx_global: 全局保留的token索引（所有层共用）
    """
    n_orig = input_ids.shape[1]
    n_keep = len(keep_idx_global)
    
    with torch.no_grad():
        # Step 1: 完整prefill拿KV Cache
        outputs = model(input_ids, use_cache=True)
        past_key_values = outputs.past_key_values
        # past_key_values: tuple of (key, value) per layer
        # key shape: (batch, n_heads, seq_len, head_dim)
        
        # Step 2: 物理删除——只保留keep_idx对应的KV条目
        # 新版transformers: DynamicCache.layers[li].keys / .values
        from transformers.cache_utils import DynamicCache
        evicted_cache = DynamicCache()
        for li in range(n_layers):
            layer_cache = past_key_values.layers[li]
            key = layer_cache.keys      # (1, n_heads, n_orig, head_dim)
            value = layer_cache.values
            key_kept = key[:, :, keep_idx_global, :]
            val_kept = value[:, :, keep_idx_global, :]
            evicted_cache.update(key_kept, val_kept, li)
        evicted_past = evicted_cache
        
        # Step 3: 从最后一个保留token的logit开始生成
        # 需要重新算最后一个位置的logit
        # 用缩短的KV做一步forward
        last_token_id = input_ids[:, keep_idx_global[-1]:keep_idx_global[-1]+1]
        
        generated_tokens = []
        current_past = evicted_past
        current_input = last_token_id
        
        for step in range(max_new):
            out = model(current_input, past_key_values=current_past, use_cache=True)
            logits = out.logits[0, -1, :]
            next_token = logits.argmax()
            generated_tokens.append(next_token.item())
            
            # 更新past
            current_past = out.past_key_values
            current_input = next_token.unsqueeze(0).unsqueeze(0)
            
            # EOS检测
            if next_token.item() == tokenizer.eos_token_id:
                break
    
    return generated_tokens

# ── 3. 三种策略 ──
log("\n生成对比（物理KV删除）...")

# 完整生成（不删除）
log("  完整KV生成...")
gen_full_ids = generate_with_eviction(model, tokenizer, input_ids, np.arange(n))
text_full = tokenizer.decode(gen_full_ids, skip_special_tokens=True)
log(f"  完整({len(gen_full_ids)}t): '{text_full[:120]}...'")

# 均匀压缩（全局top-75%重要性）
log("  均匀压缩(75%)生成...")
global_imp = sum(per_layer_imp)
n_keep_u = int(n * 0.75)
keep_uniform = np.sort(np.argsort(-global_imp)[:n_keep_u])
if n-1 not in keep_uniform:
    keep_uniform[-1] = n-1
    keep_uniform = np.sort(np.unique(keep_uniform))
gen_uniform_ids = generate_with_eviction(model, tokenizer, input_ids, keep_uniform)
text_uniform = tokenizer.decode(gen_uniform_ids, skip_special_tokens=True)
log(f"  均匀({len(gen_uniform_ids)}t): '{text_uniform[:120]}...'")

# T(G)引导：用T(G)加权重要性（高T(G)层的重要性权重更大）
log("  T(G)引导压缩(75%)生成...")
# T(G)加权：高T(G)层的重要性贡献更大（因为结构集中，重要性信号更可靠）
weighted_imp = np.zeros(n)
for li in range(n_layers):
    # 权重：T(G)越高，该层重要性越可靠
    weight = np.log1p(layer_TG[li])  # log(1+T(G))
    weighted_imp += per_layer_imp[li] * weight

n_keep_tg = int(n * 0.75)
keep_tg = np.sort(np.argsort(-weighted_imp)[:n_keep_tg])
if n-1 not in keep_tg:
    keep_tg[-1] = n-1
    keep_tg = np.sort(np.unique(keep_tg))
gen_tg_ids = generate_with_eviction(model, tokenizer, input_ids, keep_tg)
text_tg = tokenizer.decode(gen_tg_ids, skip_special_tokens=True)
log(f"  T(G)({len(gen_tg_ids)}t): '{text_tg[:120]}...'")

# H2O方法（累积注意力最高的token）
log("  H2O(75%)生成...")
h2o_imp = np.zeros(n)
for li in range(n_layers):
    attn = attns_full[li][0].numpy().mean(axis=0)
    causal = np.tril(attn)
    h2o_imp += causal.sum(axis=0)  # 只看入度（被关注次数）
n_keep_h2o = int(n * 0.75)
keep_h2o = np.sort(np.argsort(-h2o_imp)[:n_keep_h2o])
if n-1 not in keep_h2o:
    keep_h2o[-1] = n-1
    keep_h2o = np.sort(np.unique(keep_h2o))
gen_h2o_ids = generate_with_eviction(model, tokenizer, input_ids, keep_h2o)
text_h2o = tokenizer.decode(gen_h2o_ids, skip_special_tokens=True)
log(f"  H2O({len(gen_h2o_ids)}t): '{text_h2o[:120]}...'")

# ── 4. 质量评估 ──
log("\n质量评估...")

def repetition_ratio(text):
    if len(text) < 20:
        return 1.0
    # 检测连续重复
    words = text.split()
    if len(words) < 4:
        return 1.0
    # 检查是否有大量重复的word pattern
    from collections import Counter
    word_counts = Counter(words)
    most_common_count = word_counts.most_common(1)[0][1]
    return most_common_count / len(words)

def coherence_score(text):
    """简单连贯性：平均词长>2且不是纯标点"""
    words = [w for w in text.split() if len(w) > 1]
    if not words:
        return 0
    alpha_ratio = sum(1 for w in words if any(c.isalpha() for c in w)) / len(words)
    return alpha_ratio

results = {
    "model": "Qwen2.5-0.5B",
    "n_prompt": n,
    "gen_tokens": GEN_TOKENS,
    "layer_TG": [round(t, 2) for t in layer_TG],
    "generations": {
        "full": text_full,
        "uniform": text_uniform,
        "tg_weighted": text_tg,
        "h2o": text_h2o,
    },
    "n_generated": {
        "full": len(gen_full_ids),
        "uniform": len(gen_uniform_ids),
        "tg_weighted": len(gen_tg_ids),
        "h2o": len(gen_h2o_ids),
    }
}

eval_results = {}
for name, text in [("full", text_full), ("uniform", text_uniform), ("tg_weighted", text_tg), ("h2o", text_h2o)]:
    rep = repetition_ratio(text)
    coh = coherence_score(text)
    eval_results[name] = {
        "repetition_ratio": round(rep, 4),
        "coherence": round(coh, 4),
        "length_chars": len(text),
        "length_words": len(text.split()),
    }
    log(f"  {name:<12}: 重复率={rep:.3f}, 连贯性={coh:.3f}, 长度={len(text.split())}词")

results["evaluation"] = eval_results

# 判定
tg_rep = eval_results["tg_weighted"]["repetition_ratio"]
uni_rep = eval_results["uniform"]["repetition_ratio"]
h2o_rep = eval_results["h2o"]["repetition_ratio"]
full_rep = eval_results["full"]["repetition_ratio"]

log(f"\n  判定:")
if tg_rep < 0.3 and uni_rep < 0.3:
    log(f"  ✅ 物理删除后生成质量可接受（重复率<30%）")
elif tg_rep < uni_rep:
    log(f"  ⚠️ 生成退化但T(G)优于均匀（{tg_rep:.3f} < {uni_rep:.3f}）")
else:
    log(f"  ❌ 生成退化，T(G)不优于均匀")

if tg_rep < h2o_rep:
    log(f"  ✅ T(G)加权优于H2O（{tg_rep:.3f} < {h2o_rep:.3f}）")
elif abs(tg_rep - h2o_rep) < 0.05:
    log(f"  ≈ T(G)加权与H2O持平")
else:
    log(f"  ⚠️ T(G)加权不如H2O")

# ── 5. 保存 ──
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
log("P5v2物理KV删除生成验证完成。")
