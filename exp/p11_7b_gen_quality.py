"""7B生成质量对比：完整上下文 vs 朴素截断（KV压缩基线）
用llama-cli生成，对比生成质量（token overlap/重复率/连贯性）
"""
import sys, os, subprocess, json, re, time
sys.stdout.reconfigure(encoding='utf-8')

BIN = r"H:\llama.cpp\bin\llama-cli.exe"
MODEL = r"H:\llama.cpp\models\deepseek-r1-qwen-7b-q4_k_m.gguf"

PROMPT_FULL = """The Hamiltonian path problem asks whether a given graph contains a path that visits every vertex exactly once. This is one of the classic NP-complete problems in computer science. For dense random graphs, traditional backtracking search has exponential complexity. However, the spectral lineage model discovers that dense random graphs naturally contain high-dimensional clique structures. By using these cliques as outer embryos to wrap original nodes, the original graph can be compressed into a smaller compressed graph. After solving the Hamiltonian path on the compressed graph and expanding back, the number of virtual edges remains stable at zero to two, with coverage above ninety-nine percent. This discovery means that the Hamiltonian path problem on dense graphs can be solved in polynomial time through dimensional compression. Now, the implications for artificial intelligence are profound because the attention mechanism in Transformers can be viewed as a dense graph where each token attends to every previous token. If dimensional compression works for Hamiltonian paths, it should also work for attention matrices, potentially enabling much longer contexts on the same hardware. What is the key insight that makes this possible?
"""

def run_generate(prompt, ctx=2048, n_pred=80, seed=42):
    """跑llama-cli生成，返回生成的文本"""
    cmd = [BIN, "-m", MODEL, "-p", prompt, "-n", str(n_pred), "-c", str(ctx),
           "-t", "4", "--seed", str(seed), "--no-display-prompt", "-ngl", "99", "-no-cnv"]
    # -ngl 99 for Vulkan, --no-display-prompt 让输出只有生成文本
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=600)
    out = r.stdout
    return out

def clean_gen(raw):
    """从llama-cli输出中提取生成文本"""
    # 去掉所有时间戳行和log行
    lines = raw.split("\n")
    text_lines = []
    for ln in lines:
        # 跳过log行（含时间戳或日志标记）
        if re.match(r"^\d+\.\d+\.\d+\.\d+", ln) or "llama" in ln.lower() or ln.startswith("log"):
            continue
        if ln.strip():
            text_lines.append(ln.strip())
    return "\n".join(text_lines)

def repetition_ratio(text):
    words = text.split()
    if len(words) < 4:
        return 1.0
    from collections import Counter
    wc = Counter(words)
    return wc.most_common(1)[0][1] / len(words)

def word_overlap(t1, t2):
    s1, s2 = set(t1.split()), set(t2.split())
    if not s1 or not s2:
        return 0
    return len(s1 & s2) / len(s1 | s2)

results = {"model": "DeepSeek-R1-Distill-Qwen-7B Q4", "runs": {}}

# 1. 完整上下文
print("=== 完整上下文 ===", flush=True)
t0 = time.time()
gen_full = run_generate(PROMPT_FULL, ctx=2048)
print(f"  耗时 {time.time()-t0:.1f}s", flush=True)
full_text = clean_gen(gen_full)
results["runs"]["full_2048"] = {
    "text": full_text[:500],
    "len_words": len(full_text.split()),
    "repetition": repetition_ratio(full_text),
}
print(f"  生成({len(full_text.split())}词, 重复率{repetition_ratio(full_text):.2f}): {full_text[:200]}...", flush=True)

# 2. 朴素截断：只保留前半（等效KV砍一半）
half_len = len(PROMPT_FULL) // 2
prompt_half = PROMPT_FULL[:half_len]
print("=== 截断50%（保留前半） ===", flush=True)
t0 = time.time()
gen_half = run_generate(prompt_half, ctx=2048)
print(f"  耗时 {time.time()-t0:.1f}s", flush=True)
half_text = clean_gen(gen_half)
results["runs"]["truncated_half"] = {
    "text": half_text[:500],
    "len_words": len(half_text.split()),
    "repetition": repetition_ratio(half_text),
}
print(f"  生成({len(half_text.split())}词, 重复率{repetition_ratio(half_text):.2f}): {half_text[:200]}...", flush=True)

# 3. 极短上下文（等效KV砍到25%）
quarter_len = len(PROMPT_FULL) // 4
prompt_quarter = PROMPT_FULL[:quarter_len]
print("=== 截断75%（保留前1/4） ===", flush=True)
t0 = time.time()
gen_quarter = run_generate(prompt_quarter, ctx=2048)
print(f"  耗时 {time.time()-t0:.1f}s", flush=True)
quarter_text = clean_gen(gen_quarter)
results["runs"]["truncated_quarter"] = {
    "text": quarter_text[:500],
    "len_words": len(quarter_text.split()),
    "repetition": repetition_ratio(quarter_text),
}
print(f"  生成({len(quarter_text.split())}词, 重复率{repetition_ratio(quarter_text):.2f}): {quarter_text[:200]}...", flush=True)

# 对比
results["comparison"] = {
    "overlap_full_vs_half": round(word_overlap(full_text, half_text), 4),
    "overlap_full_vs_quarter": round(word_overlap(full_text, quarter_text), 4),
}

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "p11_7b_gen_quality.json"), "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print("\n对比:", results["comparison"], flush=True)
print("结果保存: p11_7b_gen_quality.json", flush=True)
