"""
云韶框架·AI特化·P9-真实边稀疏注意力（哈密顿G²升维接入）
核心改动：全局key选择从"key范数top-k"(G1原始边)改为"G²共同邻居"(真实边)
  query q 与 key j 的真实边强度 = |N(q) ∩ N(j)| = 共同关注的token数
  这是哈密顿框架"中层级升维"的注意力实现：G²保持了辐射结构（同辐射域内稠密化）
对照：P8的key范数选择(G1) vs P9的G²共同邻居选择
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

# ══ 稀疏注意力核心（支持两种全局key选择）══
def sparse_prefill(q, k, v, window, n_global, scaling, global_mode="knorm", q_block=256, topk_graph=32):
    """
    global_mode:
      "knorm" = P8方法，key范数top-k（G1原始边）
      "g2"    = P9方法，G²共同邻居（真实边）：每个query块选与之共同邻居最多的key
    """
    H_q, N, D = q.shape
    H_kv = k.shape[0]
    rep = H_q // H_kv
    device = q.device
    
    # G1全局枢纽（knorm模式用）
    if global_mode == "knorm" and n_global > 0:
        with torch.no_grad():
            knorm = k.float().norm(dim=-1).mean(dim=0)
            n_global = min(n_global, N)
            global_idx_fixed = torch.topk(knorm, n_global).indices.sort().values
    else:
        global_idx_fixed = None
    
    out = torch.zeros(H_q, N, D, device=device, dtype=q.dtype)
    
    for s in range(0, N, q_block):
        e = min(s + q_block, N)
        qb = q[:, s:e]
        B = e - s
        
        # 局部窗口
        local_start = max(0, e - window)
        local_idx = torch.arange(local_start, e, device=device)
        
        # 全局枢纽
        if n_global > 0:
            if global_mode == "knorm":
                global_idx = global_idx_fixed
            elif global_mode == "g2":
                # G²真实边：用query块的代表向量找共同邻居最多的key
                # 共同邻居 ≈ 注意力分布的重叠 ≈ query块平均向量与各key的相关性
                # 用q块均值做probe，对k做点积，取top-k = 与该块"共同关注"最多的key
                with torch.no_grad():
                    probe = qb.float().mean(dim=1)  # (H_q, D) 块代表query
                    probe_avg = probe.mean(dim=0)  # (D,) 跨头平均probe
                    k_avg = k.float().mean(dim=0)  # (H_kv, N, D) -> (N, D) 跨KV头
                    # G²分数 = probe·key（共同邻居代理），(N,D)·(D,) -> (N,)
                    g2_score = k_avg @ probe_avg  # (N,)
                    n_global_eff = min(n_global, N)
                    global_idx = torch.topk(g2_score, n_global_eff).indices.sort().values
            else:
                global_idx = torch.empty(0, dtype=torch.long, device=device)
            
            if global_idx is not None and global_idx.numel() > 0:
                key_idx = torch.cat([local_idx, global_idx]).unique().sort().values
            else:
                key_idx = local_idx
        else:
            key_idx = local_idx
        
        M = key_idx.numel()
        ks = k[:, key_idx]
        vs = v[:, key_idx]
        
        # 因果mask
        key_pos = key_idx.unsqueeze(0)
        qpos = torch.arange(s, e, device=device).unsqueeze(1)
        causal = (key_pos <= qpos)
        mask_add = torch.where(causal, torch.zeros((), device=device), torch.full((), float('-inf'), device=device))
        
        if rep > 1:
            ks = ks.repeat_interleave(rep, dim=0)
            vs = vs.repeat_interleave(rep, dim=0)
        
        for h in range(H_q):
            scores = (qb[h] @ ks[h].transpose(-1,-2)) * scaling
            scores = scores + mask_add
            weights = F.softmax(scores, dim=-1)
            out[h, s:e] = weights @ vs[h]
    
    return out

def make_sparse_forward(window, n_global, NUM_HEADS, NUM_KV_HEADS, HEAD_DIM, global_mode):
    def sparse_forward(self, hidden_states, position_embeddings, attention_mask=None,
                       past_key_values=None, **kwargs):
        bsz, q_len, _ = hidden_states.size()
        q = self.q_proj(hidden_states).view(bsz, q_len, NUM_HEADS, HEAD_DIM).transpose(1,2)
        k = self.k_proj(hidden_states).view(bsz, q_len, NUM_KV_HEADS, HEAD_DIM).transpose(1,2)
        v = self.v_proj(hidden_states).view(bsz, q_len, NUM_KV_HEADS, HEAD_DIM).transpose(1,2)
        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        scaling = HEAD_DIM ** -0.5
        out = sparse_prefill(q[0], k[0], v[0], window, n_global, scaling, global_mode=global_mode)
        attn_output = out.transpose(0,1).contiguous().view(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)
        return attn_output, None
    return sparse_forward

# ══ 主流程 ══
log(f"可用内存: {psutil.virtual_memory().available/1e9:.1f}GB")
log("加载模型...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, trust_remote_code=True,
    dtype=torch.float32, low_cpu_mem_usage=True, attn_implementation="eager")
model.eval()
n_layers = len(model.model.layers)
n_heads = model.config.num_attention_heads
n_kv = model.config.num_key_value_heads
head_dim = model.config.hidden_size // n_heads
log(f"{n_layers}层 {n_heads}头 {n_kv}KV头, 内存{mem():.2f}GB")

BASE_BLOCK = """The Hamiltonian path problem asks whether a given graph contains a path that visits every vertex exactly once. This is one of the classic NP-complete problems in computer science. For dense random graphs, traditional backtracking search has exponential complexity. However, the spectral lineage model discovers that dense random graphs naturally contain high-dimensional clique structures. By using these cliques as outer embryos to wrap original nodes, the original graph can be compressed into a smaller compressed graph. After solving the Hamiltonian path on the compressed graph and expanding back, the number of virtual edges remains stable at zero to two, with coverage above ninety-nine percent. In mixture of experts language models, each token is independently routed to a small subset of specialized expert networks through a learned gating mechanism. The routing decisions create a dynamic bipartite graph between tokens and experts that changes with every forward pass. The algebraic tension framework provides a principled way to diagnose which routing configurations are structurally stable and which are prone to collapse under distribution shift. """

def make_ids(target):
    n_blocks = max(1, int(target/150)+2)
    return tokenizer(BASE_BLOCK*n_blocks, return_tensors="pt", truncation=True, max_length=target)["input_ids"]

def ppl_of(logits, input_ids):
    sl = logits[:,:-1,:].contiguous(); lab = input_ids[:,1:].contiguous()
    loss = torch.nn.CrossEntropyLoss()(sl.view(-1,sl.size(-1)), lab.view(-1))
    return torch.exp(loss).item()

orig_fw = [model.model.layers[li].self_attn.forward for li in range(n_layers)]
def apply_sparse(window, n_global, mode):
    fn = make_sparse_forward(window, n_global, n_heads, n_kv, head_dim, mode)
    for li in range(n_layers):
        model.model.layers[li].self_attn.forward = types.MethodType(fn, model.model.layers[li].self_attn)
def restore():
    for li in range(n_layers):
        model.model.layers[li].self_attn.forward = orig_fw[li]

# ══ 对照实验：同配置下 knorm(G1) vs g2(真实边) ══
# 固定window/n_global，只换全局key选择策略
CONFIGS = [
    # (N, window, n_global, 标签)
    (4096,  410, 41,  "4K/10%"),
    (8192,  820, 82,  "8K/10%"),
    (16384, 820, 82,  "16K/5%"),
    (16384, 410, 41,  "16K/2.5%"),
    (32768, 820, 82,  "32K/2.5%"),
]

results = {"model":"Qwen2.5-0.5B", "runs":[]}

for N, window, n_global, label in CONFIGS:
    log(f"\n{'═'*55}\n{label} (window={window}, global={n_global})\n{'═'*55}")
    if mem() > 6.8:
        log(f"  ⛔ 内存近上限({mem():.2f})，停止"); break
    
    input_ids = make_ids(N)
    n = input_ids.shape[1]
    
    for mode in ["knorm", "g2"]:
        apply_sparse(window, n_global, mode)
        gc.collect()
        mb = mem()
        try:
            t0 = time.time()
            with torch.no_grad():
                out = model(input_ids); logits = out.logits
            tf = time.time()-t0
            ppl = ppl_of(logits, input_ids)
            mp = mem()
            results["runs"].append({
                "config": label, "N": n, "mode": mode, "window": window, "n_global": n_global,
                "status":"ok", "ppl": round(ppl,3), "time_s": round(tf,1),
                "tokens_per_sec": round(n/tf,1), "mem_peak_gb": round(mp,2),
                "beyond_32k": n>32768,
            })
            tag = "G²真实边" if mode=="g2" else "G1范数"
            log(f"  [{tag}] PPL={ppl:.3f}, {n/tf:.1f}t/s, 内存{mb:.2f}→{mp:.2f}GB")
            del out, logits; gc.collect()
        except Exception as ex:
            log(f"  [{mode}] ❌ {type(ex).__name__}: {str(ex)[:100]}")
            results["runs"].append({"config":label,"N":n,"mode":mode,"status":f"error:{type(ex).__name__}","error":str(ex)[:150]})
            gc.collect()

restore()

# ══ 汇总：g2 vs knorm PPL对比 ══
log(f"\n{'═'*55}\n汇总：G²真实边 vs G1范数（同配置PPL对比）\n{'═'*55}")
configs_seen = []
for r in results["runs"]:
    if r["config"] not in configs_seen and r.get("status")=="ok":
        configs_seen.append(r["config"])
for cfg in configs_seen:
    kn = next((r for r in results["runs"] if r["config"]==cfg and r["mode"]=="knorm" and r.get("status")=="ok"), None)
    g2 = next((r for r in results["runs"] if r["config"]==cfg and r["mode"]=="g2" and r.get("status")=="ok"), None)
    if kn and g2:
        delta = (g2["ppl"]-kn["ppl"])/kn["ppl"]*100
        winner = "G²胜" if g2["ppl"]<kn["ppl"] else "G1胜"
        log(f"  {cfg}: G1={kn['ppl']}, G²={g2['ppl']} ({delta:+.1f}%) → {winner}")

class NpEnc(json.JSONEncoder):
    def default(self,o):
        if isinstance(o,(np.floating,)): return float(o)
        if isinstance(o,(np.integer,)): return int(o)
        if isinstance(o,np.ndarray): return o.tolist()
        return super().default(o)
with open(os.path.join(OUT_DIR,"p9_co_neighbor_results.json"),"w",encoding="utf-8") as f:
    json.dump(results,f,ensure_ascii=False,indent=2,cls=NpEnc)
log(f"\n结果保存: p9_co_neighbor_results.json")
log("P9真实边稀疏注意力完成。")
