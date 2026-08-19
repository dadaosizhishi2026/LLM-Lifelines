"""
云韶框架·AI特化·G^k逐层升维稀疏注意力（G²/G³/G⁴/G⁵）
核心：G^k score = K_norm @ (K_norm^T K_norm)^{k-2} @ probe
  - 永远不materialize N×N邻接矩阵（OOM安全）
  - K^TK 只是 D×D（128×128），矩阵幂trivial
  - key先L2归一化→相关矩阵→幂次有界
数学：
  G² = K·probe（1-hop，p9已验证）
  G³ = K·(K^TK)·probe（2-hop路径计数）
  G⁴ = K·(K^TK)²·probe（3-hop）
  G⁵ = K·(K^TK)³·probe（4-hop）
§4.3警告验证：G³对稀疏/树状key分布有反效果（跨辐射域虚假连接）
  → 如果PPL不单调下降（G³>G²），则警告在注意力场景成立
"""
import sys, os, time, gc, types, json
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import torch
import torch.nn.functional as F
import psutil
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.expanduser(
    "~/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B/snapshots/060db6499f32faf8b98477b0a26969ef7d8b9987"
)

def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)
def mem(): return psutil.Process(os.getpid()).memory_info().rss/1e9

# ══ G^k 稀疏注意力核心 ══
def sparse_prefill_gk(q, k, v, window, n_global, scaling, gk_order=2, q_block=256):
    """
    gk_order: 2=G², 3=G³, 4=G⁴, 5=G⁵
    核心：score = K_norm @ M^{k-2} @ probe，M = K_norm^T @ K_norm (D×D)
    永不materialize N×N。
    """
    H_q, N, D = q.shape
    H_kv = k.shape[0]
    rep = H_q // H_kv
    device = q.device

    # 预计算跨KV头平均 + L2归一化（一次，所有块共享）
    with torch.no_grad():
        k_avg = k.float().mean(dim=0)  # (N, D)
        k_norms = k_avg.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        k_hat = k_avg / k_norms  # (N, D) L2归一化
        # M = K_hat^T @ K_hat: (D, D) — 相关矩阵，特征值有界
        M = k_hat.t() @ k_hat  # (D, D)
        # 归一化M：除以迹/D，使平均特征值≈1，防幂次爆炸
        M = M / (M.trace() / D)
        # 预计算 M^{k-2}（D×D矩阵幂，trivial）
        if gk_order <= 2:
            Mpow = None  # G²不需要M
        else:
            Mpow = torch.linalg.matrix_power(M, gk_order - 2)  # (D, D)

    out = torch.zeros(H_q, N, D, device=device, dtype=q.dtype)

    for s in range(0, N, q_block):
        e = min(s + q_block, N)
        qb = q[:, s:e]
        B = e - s

        # 局部窗口
        local_start = max(0, e - window)
        local_idx = torch.arange(local_start, e, device=device)

        # G^k全局key选择
        if n_global > 0:
            with torch.no_grad():
                probe = qb.float().mean(dim=1).mean(dim=0)  # (D,) 块代表
                probe_hat = probe / probe.norm().clamp(min=1e-8)
                if gk_order <= 2:
                    # G²: score = K_hat @ probe_hat
                    gk_score = k_hat @ probe_hat  # (N,)
                else:
                    # G^k: score = K_hat @ M^{k-2} @ probe_hat
                    transformed = Mpow @ probe_hat  # (D,)
                    gk_score = k_hat @ transformed  # (N,)
                n_eff = min(n_global, N)
                global_idx = torch.topk(gk_score, n_eff).indices.sort().values

            key_idx = torch.cat([local_idx, global_idx]).unique().sort().values
        else:
            key_idx = local_idx

        ks = k[:, key_idx]
        vs = v[:, key_idx]

        # 因果mask
        key_pos = key_idx.unsqueeze(0)
        qpos = torch.arange(s, e, device=device).unsqueeze(1)
        causal = (key_pos <= qpos)
        mask_add = torch.where(causal, torch.zeros((), device=device),
                               torch.full((), float('-inf'), device=device))

        if rep > 1:
            ks = ks.repeat_interleave(rep, dim=0)
            vs = vs.repeat_interleave(rep, dim=0)

        for h in range(H_q):
            scores = (qb[h] @ ks[h].transpose(-1, -2)) * scaling
            scores = scores + mask_add
            weights = F.softmax(scores, dim=-1)
            out[h, s:e] = weights @ vs[h]

    return out


def make_sparse_forward(window, n_global, NUM_HEADS, NUM_KV_HEADS, HEAD_DIM, gk_order):
    def sparse_forward(self, hidden_states, position_embeddings, attention_mask=None,
                       past_key_values=None, **kwargs):
        bsz, q_len, _ = hidden_states.size()
        q = self.q_proj(hidden_states).view(bsz, q_len, NUM_HEADS, HEAD_DIM).transpose(1, 2)
        k = self.k_proj(hidden_states).view(bsz, q_len, NUM_KV_HEADS, HEAD_DIM).transpose(1, 2)
        v = self.v_proj(hidden_states).view(bsz, q_len, NUM_KV_HEADS, HEAD_DIM).transpose(1, 2)
        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        scaling = HEAD_DIM ** -0.5
        out = sparse_prefill_gk(q[0], k[0], v[0], window, n_global, scaling, gk_order=gk_order)
        attn_output = out.transpose(0, 1).contiguous().view(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)
        return attn_output, None
    return sparse_forward


# ══ 主流程 ══
log(f"可用内存: {psutil.virtual_memory().available/1e9:.1f}GB")
log("加载模型 Qwen2.5-0.5B...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, trust_remote_code=True,
    torch_dtype=torch.float32, low_cpu_mem_usage=True, attn_implementation="eager")
model.eval()
n_layers = len(model.model.layers)
n_heads = model.config.num_attention_heads
n_kv = model.config.num_key_value_heads
head_dim = model.config.hidden_size // n_heads
log(f"{n_layers}层 {n_heads}头 {n_kv}KV头 head_dim={head_dim}, 内存{mem():.2f}GB")

BASE_BLOCK = """The Hamiltonian path problem asks whether a given graph contains a path that visits every vertex exactly once. This is one of the classic NP-complete problems in computer science. For dense random graphs, traditional backtracking search has exponential complexity. However, the spectral lineage model discovers that dense random graphs naturally contain high-dimensional clique structures. By using these cliques as outer embryos to wrap original nodes, the original graph can be compressed into a smaller compressed graph. After solving the Hamiltonian path on the compressed graph and expanding back, the number of virtual edges remains stable at zero to two, with coverage above ninety-nine percent. In mixture of experts language models, each token is independently routed to a small subset of specialized expert networks through a learned gating mechanism. The routing decisions create a dynamic bipartite graph between tokens and experts that changes with every forward pass. The algebraic tension framework provides a principled way to diagnose which routing configurations are structurally stable and which are prone to collapse under distribution shift. """

def make_ids(target):
    n_blocks = max(1, int(target / 150) + 2)
    return tokenizer(BASE_BLOCK * n_blocks, return_tensors="pt",
                     truncation=True, max_length=target)["input_ids"]

def ppl_of(logits, input_ids):
    sl = logits[:, :-1, :].contiguous()
    lab = input_ids[:, 1:].contiguous()
    loss = torch.nn.CrossEntropyLoss()(sl.view(-1, sl.size(-1)), lab.view(-1))
    return torch.exp(loss).item()

orig_fw = [model.model.layers[li].self_attn.forward for li in range(n_layers)]

def apply_gk(window, n_global, gk_order):
    fn = make_sparse_forward(window, n_global, n_heads, n_kv, head_dim, gk_order)
    for li in range(n_layers):
        model.model.layers[li].self_attn.forward = types.MethodType(
            fn, model.model.layers[li].self_attn)

def restore():
    for li in range(n_layers):
        model.model.layers[li].self_attn.forward = orig_fw[li]

# ══ 实验：G²/G³/G⁴/G⁵ 逐层PPL对比 ══
CONFIGS = [
    # (N, window, n_global, label)
    (4096, 410, 41, "4K/10%"),
    (8192, 820, 82, "8K/10%"),
]
GK_ORDERS = [2, 3, 4, 5]

results = {"model": "Qwen2.5-0.5B", "task": "Gk_ascend", "runs": []}

for N, window, n_global, label in CONFIGS:
    log(f"\n{'═'*55}\n{label} (window={window}, global={n_global})\n{'═'*55}")
    if mem() > 6.5:
        log(f"  ⛔ 内存近上限({mem():.2f}GB)，停止")
        break

    input_ids = make_ids(N)
    n = input_ids.shape[1]
    log(f"  实际token数: {n}")

    for gk in GK_ORDERS:
        apply_gk(window, n_global, gk)
        gc.collect()
        mb = mem()
        try:
            t0 = time.time()
            with torch.no_grad():
                out = model(input_ids)
                logits = out.logits
            tf = time.time() - t0
            ppl = ppl_of(logits, input_ids)
            mp = mem()
            results["runs"].append({
                "config": label, "N": n, "gk_order": gk,
                "window": window, "n_global": n_global,
                "status": "ok", "ppl": round(ppl, 3),
                "time_s": round(tf, 1), "tokens_per_sec": round(n / tf, 1),
                "mem_peak_gb": round(mp, 2),
            })
            log(f"  [G^{gk}] PPL={ppl:.3f}, {n/tf:.1f}t/s, 内存{mb:.2f}→{mp:.2f}GB")
            del out, logits; gc.collect()
        except Exception as ex:
            log(f"  [G^{gk}] ❌ {type(ex).__name__}: {str(ex)[:120]}")
            results["runs"].append({
                "config": label, "N": n, "gk_order": gk,
                "status": f"error:{type(ex).__name__}", "error": str(ex)[:200]
            })
            gc.collect()

restore()

# ══ 汇总：逐层收敛分析 ══
log(f"\n{'═'*55}\n汇总：G^k逐层PPL（收敛分析）\n{'═'*55}")
log(f"{'Config':<10} {'G²':>8} {'G³':>8} {'G⁴':>8} {'G⁵':>8} {'单调↓?':>8} {'G³反效果?':>10}")
for N, window, n_global, label in CONFIGS:
    ppls = {}
    for gk in GK_ORDERS:
        r = next((x for x in results["runs"]
                  if x["config"] == label and x.get("gk_order") == gk and x.get("status") == "ok"), None)
        ppls[gk] = r["ppl"] if r else None
    vals = [ppls[g] for g in GK_ORDERS if ppls[g] is not None]
    monotone = all(vals[i] >= vals[i+1] for i in range(len(vals)-1)) if len(vals) >= 2 else None
    g3_anti = (ppls[3] is not None and ppls[2] is not None and ppls[3] > ppls[2])
    row = f"{label:<10}"
    for g in GK_ORDERS:
        row += f" {ppls[g]:>8.3f}" if ppls[g] else f" {'ERR':>8}"
    row += f" {'YES' if monotone else 'NO':>8}" if monotone is not None else f" {'N/A':>8}"
    row += f" {'⚠️YES' if g3_anti else 'no':>10}"
    log(row)

# §4.3警告验证结论
log(f"\n{'═'*55}\n§4.3警告验证（G³对稀疏图反效果）\n{'═'*55}")
for N, window, n_global, label in CONFIGS:
    g2r = next((x for x in results["runs"] if x["config"]==label and x.get("gk_order")==2 and x.get("status")=="ok"), None)
    g3r = next((x for x in results["runs"] if x["config"]==label and x.get("gk_order")==3 and x.get("status")=="ok"), None)
    if g2r and g3r:
        delta = (g3r["ppl"] - g2r["ppl"]) / g2r["ppl"] * 100
        if delta > 0:
            log(f"  {label}: G³ PPL比G²高{delta:+.1f}% → ⚠️警告成立（注意力key分布类树/稀疏，G³引入跨域虚假连接）")
        else:
            log(f"  {label}: G³ PPL比G²低{delta:+.1f}% → 警告不成立（注意力key分布足够稠密，G³升维有效）")

class NpEnc(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, (np.floating,)): return float(o)
        if isinstance(o, (np.integer,)): return int(o)
        if isinstance(o, np.ndarray): return o.tolist()
        return super().default(o)

with open(os.path.join(OUT_DIR, "gk_graph_results.json"), "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2, cls=NpEnc)
log(f"\n结果保存: gk_graph_results.json")
log("G^k逐层升维稀疏注意力实验完成。")
