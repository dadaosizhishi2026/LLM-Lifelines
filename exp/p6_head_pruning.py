"""
云韶框架·AI特化·P6
T(G)引导注意力头剪枝
- 用T(G)相态判定每层每头的结构状态
- 剪掉"无解相"头（T(G)<1/8，注意力接近均匀分布=噪声）
- 对比：T(G)剪枝 vs 随机剪枝 vs 低注意力剪枝 vs 不剪
- 指标：PPL + 生成质量
这是框架诊断能力最直接的工程应用
"""
import sys, os, json, time
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(OUT_DIR, "p6_head_pruning.json")

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
log(f"文本: {n} tokens, {n_layers}层, {n_heads}头, 总{n_layers*n_heads}头")

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

# ── 2. 计算每层每头的T(G) ──
log("\n计算每层每头T(G)...")
head_TG = np.zeros((n_layers, n_heads))
head_phase = [[""] * n_heads for _ in range(n_layers)]
head_entropy = np.zeros((n_layers, n_heads))  # 注意力熵（对照指标）

for li in range(n_layers):
    for hi in range(n_heads):
        attn = attns_full[li][0, hi].numpy()  # (n, n)
        causal = np.tril(attn)
        
        # 熵
        entropy = -(causal * np.log(causal + 1e-10)).sum(axis=1).mean()
        head_entropy[li, hi] = entropy
        
        # T(G)
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
        head_TG[li, hi] = T_G
        
        alpha = 1/8
        if T_G > 1/alpha:
            head_phase[li][hi] = "real"
        elif T_G > alpha:
            head_phase[li][hi] = "virtual"
        else:
            head_phase[li][hi] = "unsolvable"

# 统计
phase_counts = {"real": 0, "virtual": 0, "unsolvable": 0}
for li in range(n_layers):
    for hi in range(n_heads):
        phase_counts[head_phase[li][hi]] += 1

total_heads = n_layers * n_heads
log(f"  三态分布: 真实={phase_counts['real']}({phase_counts['real']/total_heads*100:.1f}%), "
    f"虚拟={phase_counts['virtual']}({phase_counts['virtual']/total_heads*100:.1f}%), "
    f"无解={phase_counts['unsolvable']}({phase_counts['unsolvable']/total_heads*100:.1f}%)")

# ── 3. 剪枝策略 ──
# 策略A：T(G)剪枝——剪掉T(G)最低的N%头（无解相优先）
# 策略B：随机剪枝——随机剪掉N%头
# 策略C：高熵剪枝——剪掉注意力熵最高的N%头（最均匀的=最没信息的）
# 策略D：低注意力剪枝——剪掉平均注意力权重最低的N%头

def get_prune_mask(strategy, prune_ratio):
    """返回 (n_layers, n_heads) 的bool数组，True=保留，False=剪掉"""
    n_prune = int(total_heads * prune_ratio)
    mask = np.ones((n_layers, n_heads), dtype=bool)
    
    if strategy == "tg":
        # 按T(G)排序，剪最低的
        flat_tg = head_TG.flatten()
        prune_idx = np.argsort(flat_tg)[:n_prune]
        for idx in prune_idx:
            li, hi = divmod(idx, n_heads)
            mask[li, hi] = False
    
    elif strategy == "random":
        flat_idx = np.arange(total_heads)
        np.random.seed(42)
        prune_idx = np.random.choice(flat_idx, n_prune, replace=False)
        for idx in prune_idx:
            li, hi = divmod(idx, n_heads)
            mask[li, hi] = False
    
    elif strategy == "entropy":
        # 剪熵最高的（最均匀=最没信息）
        flat_ent = head_entropy.flatten()
        prune_idx = np.argsort(-flat_ent)[:n_prune]
        for idx in prune_idx:
            li, hi = divmod(idx, n_heads)
            mask[li, hi] = False
    
    elif strategy == "low_attn":
        # 剪平均注意力权重最低的
        avg_attn = np.zeros((n_layers, n_heads))
        for li in range(n_layers):
            for hi in range(n_heads):
                avg_attn[li, hi] = attns_full[li][0, hi].numpy().mean()
        flat_attn = avg_attn.flatten()
        prune_idx = np.argsort(flat_attn)[:n_prune]
        for idx in prune_idx:
            li, hi = divmod(idx, n_heads)
            mask[li, hi] = False
    
    return mask

# ── 4. 剪枝后前向（用hook把被剪头的attention置零）──
def forward_with_head_prune(model, input_ids, prune_mask):
    """用hook把被剪头的attention output置零"""
    hooks = []
    
    def make_hook(li, hi):
        def hook_fn(module, input, output):
            # output是attn_output (batch, seq, hidden)
            # 需要把对应头的贡献置零
            # Qwen2的self_attn输出是 (attn_output, attn_weights, past_kv)
            # attn_output已经合并了所有头，无法单独置零某个头
            # 正确方法：在attention weights层面置零
            return output
        return hook_fn
    
    # 更好的方法：直接修改attention weights
    # 用register_forward_hook在self_attn上，修改attn_weights
    def make_attn_hook(li, prune_heads):
        def hook_fn(module, args, kwargs, output):
            # output: (attn_output, attn_weights, past_kv)
            if isinstance(output, tuple) and len(output) >= 2 and output[1] is not None:
                attn_weights = output[1]  # (batch, heads, seq, seq)
                for hi in prune_heads:
                    attn_weights[:, hi, :, :] = 0.0
                # 重新归一化（被剪头的行和为0，需要处理）
                # 实际上置零后softmax已经做过了，直接置零attn_output中对应头的贡献
                # 但attn_output已经合并了...
                # 最简单：把被剪头的attn_weights置零，然后重新计算attn_output
                # 但这需要value vectors...太复杂
                # 简化：直接把被剪头的attn_output贡献置零
                pass
            return output
        return hook_fn
    
    # 最可靠的方法：monkey-patch每层的self_attn.forward
    # 在softmax之后、matmul value之前，把被剪头的weights置零
    # 但Qwen2的forward是一体的...
    
    # 最简方法：用attention_mask的head维度
    # Qwen2支持4D attention_mask (batch, heads, seq, seq)
    # 对被剪头，mask全设为-inf → softmax后全为0 → 该头无贡献
    
    prune_heads_per_layer = [[] for _ in range(n_layers)]
    for li in range(n_layers):
        for hi in range(n_heads):
            if not prune_mask[li, hi]:
                prune_heads_per_layer[li].append(hi)
    
    # 构建4D mask：对被剪头全-inf
    # 但Qwen2的attention_mask是 (batch, 1, seq, seq)，不支持per-head
    # 需要 (batch, heads, seq, seq)
    
    # 用hook方法：在每层self_attn的forward中拦截
    # Qwen2Attention.forward签名：(hidden_states, attention_mask, position_ids, ...)
    # attention_mask如果是4D (batch, heads, seq, seq)，可以直接用
    
    # 构建per-head mask
    head_mask = torch.zeros(1, n_heads, n, n)
    for li in range(n_layers):
        # 每层mask不同，需要逐层处理
        pass
    
    # 正确方法：hook o_proj的输入，把被剪头的向量置零
    # Qwen2Attention: attn_output (batch, heads, seq, head_dim) → transpose → (batch, seq, heads*head_dim) → o_proj
    # 在o_proj之前，把被剪头对应的head_dim切片置零 = 该头无贡献
    head_dim = model.config.hidden_size // n_heads
    hooks = []
    
    for li in range(n_layers):
        pruned_heads = [hi for hi in range(n_heads) if not prune_mask[li, hi]]
        if not pruned_heads:
            continue
        
        def make_hook(ph, hd):
            def hook_fn(module, input):
                x = input[0]  # (batch, seq, n_heads*head_dim)
                for hi in ph:
                    x[:, :, hi*hd:(hi+1)*hd] = 0.0
                return (x,)
            return hook_fn
        
        h = model.model.layers[li].self_attn.o_proj.register_forward_pre_hook(make_hook(pruned_heads, head_dim))
        hooks.append(h)
    
    with torch.no_grad():
        out = model(input_ids)
        logits = out.logits
    
    for h in hooks:
        h.remove()
    
    return logits

# ── 5. 实验 ──
results = {
    "model": "Qwen2.5-0.5B",
    "n_tokens": n,
    "n_layers": n_layers,
    "n_heads": n_heads,
    "total_heads": total_heads,
    "ppl_full": round(ppl_full, 2),
    "phase_distribution": phase_counts,
    "head_TG_stats": {
        "mean": round(float(head_TG.mean()), 2),
        "std": round(float(head_TG.std()), 2),
        "min": round(float(head_TG.min()), 2),
        "max": round(float(head_TG.max()), 2),
    },
    "experiments": []
}

PRUNE_RATIOS = [0.05, 0.10, 0.15, 0.20, 0.30]
STRATEGIES = ["tg", "entropy", "random"]

log(f"\n{'═'*60}")
log("注意力头剪枝实验")
log(f"{'═'*60}")

for prune_ratio in PRUNE_RATIOS:
    n_pruned = int(total_heads * prune_ratio)
    log(f"\n── 剪枝 {prune_ratio*100:.0f}% ({n_pruned}/{total_heads} 头) ──")
    
    for strategy in STRATEGIES:
        mask = get_prune_mask(strategy, prune_ratio)
        logits_p = forward_with_head_prune(model, input_ids, mask)
        
        loss_p = torch.nn.CrossEntropyLoss()(
            logits_p[:, :-1, :].contiguous().view(-1, logits_p.size(-1)),
            shift_labels.contiguous().view(-1)
        )
        ppl_p = torch.exp(loss_p).item()
        ppl_ratio = ppl_p / ppl_full
        
        results["experiments"].append({
            "prune_ratio": prune_ratio,
            "strategy": strategy,
            "n_pruned": n_pruned,
            "ppl": round(ppl_p, 2),
            "ppl_ratio": round(ppl_ratio, 4),
            "ppl_change_pct": round((ppl_ratio - 1) * 100, 1),
        })
        
        marker = "🟢" if ppl_ratio < 1.0 else ("  " if ppl_ratio < 1.1 else "🔴")
        log(f"  {marker} {strategy:<10}: PPL={ppl_p:.2f} (×{ppl_ratio:.3f}, {(ppl_ratio-1)*100:+.1f}%)")

# ── 6. 核心验证 ──
log(f"\n{'═'*60}")
log("核心验证")
log(f"{'═'*60}")

# 在每个剪枝比例下，T(G)是否优于随机和熵？
for prune_ratio in PRUNE_RATIOS:
    tg_ppl = next(e for e in results["experiments"] if e["prune_ratio"]==prune_ratio and e["strategy"]=="tg")["ppl"]
    rand_ppl = next(e for e in results["experiments"] if e["prune_ratio"]==prune_ratio and e["strategy"]=="random")["ppl"]
    ent_ppl = next(e for e in results["experiments"] if e["prune_ratio"]==prune_ratio and e["strategy"]=="entropy")["ppl"]
    
    tg_wins = tg_ppl <= min(rand_ppl, ent_ppl)
    log(f"  {prune_ratio*100:.0f}%: T(G)={tg_ppl:.2f}, 随机={rand_ppl:.2f}, 熵={ent_ppl:.2f} → {'✅ T(G)最优' if tg_wins else '⚠️ T(G)非最优'}")

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
log("P6注意力头剪枝验证完成。")
