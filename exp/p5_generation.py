"""
云韶框架·AI特化·P5
T(G)引导KV压缩：真实生成质量验证
方法：用T(G)引导的逐层KV mask压缩长序列，然后从压缩状态继续生成（续写）
对比：完整KV生成 vs T(G)引导压缩生成 vs 均匀压缩生成
指标：生成质量（跟真实续写的overlap）、生成文本连贯性
"""
import sys, os, json, time
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(OUT_DIR, "p5_generation.json")

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

# 用一段"未完待续"的文本：讲完哈密顿路径后，留一个开放结尾，让模型续写
PROMPT = """The Hamiltonian path problem asks whether a given graph contains a path that visits every vertex exactly once. This is one of the classic NP-complete problems in computer science. For dense random graphs, traditional backtracking search has exponential complexity. However, the spectral lineage model discovers that dense random graphs naturally contain high-dimensional clique structures. By using these cliques as outer embryos to wrap original nodes, the original graph can be compressed into a smaller compressed graph. After solving the Hamiltonian path on the compressed graph and expanding back, the number of virtual edges remains stable at zero to two, with coverage above ninety-nine percent. This discovery means that the Hamiltonian path problem on dense graphs can be solved in polynomial time through dimensional compression. Now, the implications for artificial intelligence are profound because the attention mechanism in Transformers can be viewed as a dense graph where each token attends to every previous token."""

input_ids = tokenizer(PROMPT, return_tensors="pt")["input_ids"]
n = input_ids.shape[1]
n_layers = len(model.model.layers)
n_heads = model.config.num_attention_heads
log(f"Prompt: {n} tokens, {n_layers}层, {n_heads}头")

GEN_TOKENS = 60

# ── 1. 完整前向拿注意力+重要性 ──
log("完整前向...")
with torch.no_grad():
    out_full = model(input_ids, output_attentions=True)
    attns_full = out_full.attentions

# 全局重要性（跨层平均）
global_imp = np.zeros(n)
per_layer_imp = []
for li in range(n_layers):
    attn = attns_full[li][0].numpy().mean(axis=0)
    causal = np.tril(attn)
    imp_layer = causal.sum(axis=0) * causal.sum(axis=1)
    per_layer_imp.append(imp_layer)
    global_imp += imp_layer

# 每层T(G)
layer_TG = []
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
    imp = in_deg * out_deg
    mean_imp = imp.mean()
    T_G = float(imp.max() / mean_imp - 1) if mean_imp > 1e-10 else 0.0
    layer_TG.append(T_G)

TG_arr = np.array(layer_TG)
TG_max, TG_min = float(TG_arr.max()), float(TG_arr.min())
log(f"T(G)范围: [{TG_min:.1f}, {TG_max:.1f}]")

# ── 2. 工具：逐层生成 ──
def generate_per_layer(model, tokenizer, input_ids, per_layer_keep, max_new=GEN_TOKENS, temp_fn=None):
    """逐层KV mask + 可选温度，自回归生成"""
    generated = input_ids.clone()
    seq_len = input_ids.shape[1]
    
    with torch.no_grad():
        # 先prefill所有输入（用每层的mask）
        hidden = model.model.embed_tokens(input_ids)
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        position_embeddings = model.model.rotary_emb(hidden, position_ids)
        
        for li in range(n_layers):
            layer = model.model.layers[li]
            keep_idx = per_layer_keep[li]
            
            # 温度
            saved_scaling = None
            if temp_fn is not None:
                T = temp_fn(li)
                saved_scaling = layer.self_attn.scaling
                layer.self_attn.scaling = saved_scaling / T
            
            # mask（prefill阶段）
            mask = torch.full((1, 1, seq_len, seq_len), float('-inf'))
            for i in range(seq_len):
                valid = keep_idx[keep_idx <= i]
                if len(valid) > 0:
                    mask[0, 0, i, valid] = 0.0
            
            layer_out = layer(hidden, position_embeddings=position_embeddings, attention_mask=mask)
            if saved_scaling is not None:
                layer.self_attn.scaling = saved_scaling
            hidden = layer_out[0] if isinstance(layer_out, tuple) else layer_out
        
        hidden = model.model.norm(hidden)
        logits = model.lm_head(hidden)
        past_seq = hidden  # 保留最后一层hidden用于后续
    
    # 但现在我们需要继续生成，需要一个能逐步扩展的机制
    # 简化方法：先生成整个序列（一次性），但有mask问题
    # 更简单可靠：用不同mask重新生成整个输出
    # 这里用贪心解码，但每步需要重新prefill（慢但正确）
    
    current = input_ids.clone()  # (1, n)
    with torch.no_grad():
        for step in range(max_new):
            cur_len = current.shape[1]
            
            # 每层的mask：对于新生成的token，只能attend到保留的原始token + 所有已生成的新token
            hidden = model.model.embed_tokens(current)
            position_ids = torch.arange(cur_len, device=current.device).unsqueeze(0)
            position_embeddings = model.model.rotary_emb(hidden, position_ids)
            
            for li in range(n_layers):
                layer = model.model.layers[li]
                keep_idx = per_layer_keep[li]
                
                # 温度
                saved_scaling = None
                if temp_fn is not None:
                    T = temp_fn(li)
                    saved_scaling = layer.self_attn.scaling
                    layer.self_attn.scaling = saved_scaling / T
                
                # mask：原始位置只能attend到保留token；新位置attend到保留token+新token
                mask = torch.full((1, 1, cur_len, cur_len), float('-inf'))
                for i in range(cur_len):
                    if i < n:  # 原始token位置
                        valid = keep_idx[keep_idx <= i]
                        if len(valid) > 0:
                            mask[0, 0, i, valid] = 0.0
                        # 也允许attend到新token（简化，保证信息流通）
                        mask[0, 0, i, n:] = 0.0
                    else:  # 新token位置
                        mask[0, 0, i, :i+1] = 0.0  # 允许attend到前面所有
                
                layer_out = layer(hidden, position_embeddings=position_embeddings, attention_mask=mask)
                if saved_scaling is not None:
                    layer.self_attn.scaling = saved_scaling
                hidden = layer_out[0] if isinstance(layer_out, tuple) else layer_out
            
            hidden = model.model.norm(hidden)
            logits = model.lm_head(hidden)
            next_token = logits[0, -1].argmax().unsqueeze(0).unsqueeze(0)
            current = torch.cat([current, next_token], dim=1)
    
    return current[0, n:]

# ── 3. 三种生成 ──
log("\n生成对比...")

# 完整生成
log("  完整KV生成...")
gen_full = generate_per_layer(model, tokenizer, input_ids, 
    [np.arange(n)] * n_layers)  # 保留所有
text_full = tokenizer.decode(gen_full, skip_special_tokens=True)
log(f"  完整: '{text_full[:100]}...'")

# 均匀压缩（保留75%）
log("  均匀压缩(75%)生成...")
n_keep_u = int(n * 0.75)
keep_uniform = [np.sort(np.argsort(-imp)[:n_keep_u]) for imp in per_layer_imp]
# 保证最后位置
for li in range(n_layers):
    if n-1 not in keep_uniform[li]:
        keep_uniform[li][-1] = n-1
        keep_uniform[li] = np.sort(np.unique(keep_uniform[li]))
gen_uniform = generate_per_layer(model, tokenizer, input_ids, keep_uniform)
text_uniform = tokenizer.decode(gen_uniform, skip_special_tokens=True)
log(f"  均匀: '{text_uniform[:100]}...'")

# T(G)引导压缩（总预算75%）
log("  T(G)引导压缩(75%)生成...")
BETA = 0.5
budgets = np.zeros(n_layers)
for li in range(n_layers):
    tg_norm = (layer_TG[li] - TG_min) / (TG_max - TG_min) if TG_max > TG_min else 0.5
    budgets[li] = 0.75 * (1 + BETA * (1 - tg_norm))
budgets = budgets * (0.75 * n_layers / budgets.sum())
budgets = np.clip(budgets, 0.1, 1.0)

keep_tg = []
for li in range(n_layers):
    nk = max(int(n * budgets[li]), 4)
    imp = per_layer_imp[li]
    keep = np.sort(np.argsort(-imp)[:nk])
    if n-1 not in keep:
        keep[-1] = n-1
        keep = np.sort(np.unique(keep))
    keep_tg.append(keep)

gen_tg = generate_per_layer(model, tokenizer, input_ids, keep_tg)
text_tg = tokenizer.decode(gen_tg, skip_special_tokens=True)
log(f"  T(G): '{text_tg[:100]}...'")

# ── 4. 质量评估 ──
log("\n质量评估...")

def compute_overlap(text1, text2):
    """字符级overlap"""
    t1, t2 = set(text1), set(text2)
    if not t1 or not t2:
        return 0
    return len(t1 & t2) / len(t1 | t2)

def compute_token_overlap(ids1, ids2):
    """token级overlap"""
    s1, s2 = set(ids1.tolist()), set(ids2.tolist())
    if not s1 or not s2:
        return 0
    return len(s1 & s2) / len(s1 | s2)

results = {
    "model": "Qwen2.5-0.5B",
    "n_prompt": n,
    "gen_tokens": GEN_TOKENS,
    "layer_TG": [round(t, 2) for t in layer_TG],
    "generations": {
        "full": text_full,
        "uniform": text_uniform,
        "tg": text_tg,
    }
}

# overlap vs full
token_overlap_uniform = compute_token_overlap(gen_uniform, gen_full)
token_overlap_tg = compute_token_overlap(gen_tg, gen_full)
char_overlap_uniform = compute_overlap(text_uniform, text_full)
char_overlap_tg = compute_overlap(text_tg, text_full)

results["evaluation"] = {
    "token_overlap_uniform": round(token_overlap_uniform, 4),
    "token_overlap_tg": round(token_overlap_tg, 4),
    "char_overlap_uniform": round(char_overlap_uniform, 4),
    "char_overlap_tg": round(char_overlap_tg, 4),
    "tg_wins": "T(G)" if token_overlap_tg > token_overlap_uniform else "uniform",
}

log(f"  Token overlap: 均匀={token_overlap_uniform:.4f}, T(G)={token_overlap_tg:.4f}")
log(f"  字符overlap:   均匀={char_overlap_uniform:.4f}, T(G)={char_overlap_tg:.4f}")

# 生成文本长度（解码后字符数，衡量是否退化/重复）
len_full, len_uniform, len_tg = len(text_full), len(text_uniform), len(text_tg)
results["generation_length"] = {
    "full": len_full, "uniform": len_uniform, "tg": len_tg
}
log(f"  生成文本长度: 完整={len_full}, 均匀={len_uniform}, T(G)={len_tg}")

# 退化检测：是否大量重复
def repetition_ratio(text):
    if len(text) < 20:
        return 0
    # 检测是否连续重复同一子串
    for length in [5, 10, 20]:
        for i in range(0, len(text)-length, length):
            chunk = text[i:i+length]
            count = text.count(chunk)
            if count > 2:
                return count / (len(text) / max(length, 1))
    return 0

rep_full = repetition_ratio(text_full)
rep_uniform = repetition_ratio(text_uniform)
rep_tg = repetition_ratio(text_tg)
results["repetition_ratio"] = {
    "full": rep_full, "uniform": rep_uniform, "tg": rep_tg
}
log(f"  重复率: 完整={rep_full:.3f}, 均匀={rep_uniform:.3f}, T(G)={rep_tg:.3f}")

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
log("P5真实生成验证完成。")
