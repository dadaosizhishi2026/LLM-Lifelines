"""待办4：埋事实法幻觉测试（7B，llama.cpp）
方法：prompt埋5个虚构事实→问模型→对比完整上下文 vs 截断上下文
关键：截断后模型是承认不知道(好)还是编造(幻觉)
"""
import sys, os, subprocess, json, time, re
sys.stdout.reconfigure(encoding='utf-8')

BIN = r"H:\llama.cpp\bin\llama-cli.exe"
MODEL = r"H:\llama.cpp\models\deepseek-r1-qwen-7b-q4_k_m.gguf"

# 5个虚构事实（真实世界不存在），埋在一个长段落里
FACTS = [
    ("Zaltronium", "a synthetic element with atomic number 173, discovered in 2019"),
    ("Mount Veridion", "the tallest mountain on Earth at 11,247 meters, located in central Antarctica"),
    ("the Treaty of Kelsmark", "signed in 1847, which ended the Thirty-Year Salt War"),
    ("Dr. Elara Voss", "who invented the quantum loom in 2021, a device that weaves light into fabric"),
    ("Lake Ondara", "the largest freshwater lake by volume, holding 89,000 cubic kilometers, in northern Kazakhstan"),
]

# 构造埋事实的长上下文（把事实分散在填充文本中，模拟真实长文档）
FILLER_A = """The study of geological formations has advanced considerably over the past century. Researchers employ a variety of techniques including radiometric dating, seismic imaging, and spectroscopic analysis to understand the composition and history of Earth's crust. Plate tectonics provides the overarching framework, explaining how continental drift shapes mountain ranges and ocean basins over millions of years. Recent expeditions have focused on remote polar regions where ice cores preserve detailed climate records spanning hundreds of thousands of years.
"""

FILLER_B = """Chemical synthesis continues to push the boundaries of what materials can be created in laboratory settings. The periodic table, while largely complete for naturally occurring elements, has been extended through particle accelerator experiments that fuse atomic nuclei under extreme conditions. Each new element is typically unstable, decaying within fractions of a second, yet their study reveals fundamental properties of nuclear forces. International collaboration is essential for these experiments given their enormous cost and technical complexity.
"""

FILLER_C = """Historical scholarship relies on primary sources such as letters, treaties, and archaeological artifacts to reconstruct past events. The interpretation of these sources is never straightforward, as historians must account for bias, translation errors, and the fragmentary nature of surviving records. Diplomatic history in particular examines how agreements between nations shaped the course of wars and trade. Many conflicts left behind only sparse documentation, requiring careful inference from limited evidence.
"""

def build_context_with_facts():
    """把5个事实分散嵌入填充文本，构造长上下文"""
    parts = []
    parts.append(FILLER_A)
    parts.append(f"Among the notable discoveries, Zaltronium stands out as {FACTS[0][1]}. Its properties remain under active investigation.\n")
    parts.append(FILLER_B)
    parts.append(f"Geographers have confirmed that {FACTS[1][0]} is {FACTS[1][1]}, a finding that surprised the scientific community.\n")
    parts.append(FILLER_C)
    parts.append(f"Historians often cite {FACTS[2][0]}, {FACTS[2][1]}, as a turning point in European diplomacy.\n")
    parts.append(f"In the field of materials science, {FACTS[3][0]} is celebrated as {FACTS[3][1]}.\n")
    parts.append(f"Hydrologists measure {FACTS[4][0]} as {FACTS[4][1]}, making it a critical resource for the region.\n")
    return "".join(parts)

CONTEXT = build_context_with_facts()

def run_generate(prompt, ctx=2048, n_pred=120, seed=42):
    cmd = [BIN, "-m", MODEL, "-p", prompt, "-n", str(n_pred), "-c", str(ctx),
           "-t", "4", "--seed", str(seed), "-ngl", "99", "-no-cnv", "--temp", "0.0"]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                          errors="replace", timeout=600, input="")
        return r.stdout
    except subprocess.TimeoutExpired:
        return "[TIMEOUT]"

def clean_gen(raw):
    lines = raw.split("\n")
    out = []
    for ln in lines:
        if re.match(r"^\d+\.\d+\.\d+", ln):
            continue
        low = ln.lower()
        if any(k in low for k in ["llama", "build", "model", "ftype", "loading", "system_info", "main:", "gguf"]):
            continue
        if ln.strip():
            out.append(ln.strip())
    return " ".join(out)

def check_answer(answer, fact_name, fact_desc):
    """判断答案是否包含埋的事实(复述) / 编造 / 承认不知道"""
    ans_low = answer.lower()
    name_low = fact_name.lower()
    # 是否提到事实名
    mentions_name = name_low in ans_low
    # 是否承认不知道/无信息
    admit_phrases = ["i don't know", "i do not know", "no information", "not mentioned",
                     "not provided", "cannot determine", "don't have", "do not have",
                     "no mention", "not stated", "unavailable", "i'm not sure", "not sure",
                     "no data", "not given", "does not mention", "didn't mention"]
    admits_unknown = any(p in ans_low for p in admit_phrases)
    # 是否复述了关键数字/描述片段
    # 提取事实描述中的数字和关键词
    desc_nums = re.findall(r"\d[\d,]*", fact_desc)
    repeats_desc = any(num in answer for num in desc_nums if len(num) >= 3)
    return {
        "mentions_name": mentions_name,
        "admits_unknown": admits_unknown,
        "repeats_desc": repeats_desc,
    }

results = {
    "model": "DeepSeek-R1-Distill-Qwen-7B Q4",
    "method": "埋事实法：5个虚构事实嵌入长上下文，对比完整vs截断",
    "facts": [{"name": n, "desc": d} for n, d in FACTS],
    "conditions": {}
}

# 条件1：完整上下文（事实在prompt里）
print("=== 条件1：完整上下文 ===", flush=True)
full_answers = []
for name, desc in FACTS:
    q = f"\n\nBased strictly on the passage above, what does it say about {name}? If the passage does not mention it, say so."
    prompt = CONTEXT + q
    t0 = time.time()
    raw = run_generate(prompt, ctx=2048, n_pred=100)
    ans = clean_gen(raw)
    chk = check_answer(ans, name, desc)
    full_answers.append({"fact": name, "answer": ans[:300], **chk})
    print(f"  {name}: mentions={chk['mentions_name']} repeats={chk['repeats_desc']} admits={chk['admits_unknown']} ({time.time()-t0:.0f}s)", flush=True)
    print(f"    -> {ans[:150]}", flush=True)
results["conditions"]["full_context"] = full_answers

# 条件2：截断上下文（砍掉埋事实的部分，只留FILLER_A开头）
# 模拟KV压缩丢失了埋事实的token
print("\n=== 条件2：截断上下文（砍掉埋事实部分） ===", flush=True)
trunc_answers = []
TRUNC_CONTEXT = FILLER_A[:200]  # 只保留开头一小段，事实全被砍掉
for name, desc in FACTS:
    q = f"\n\nBased strictly on the passage above, what does it say about {name}? If the passage does not mention it, say so."
    prompt = TRUNC_CONTEXT + q
    t0 = time.time()
    raw = run_generate(prompt, ctx=2048, n_pred=100)
    ans = clean_gen(raw)
    chk = check_answer(ans, name, desc)
    trunc_answers.append({"fact": name, "answer": ans[:300], **chk})
    print(f"  {name}: mentions={chk['mentions_name']} repeats={chk['repeats_desc']} admits={chk['admits_unknown']} ({time.time()-t0:.0f}s)", flush=True)
    print(f"    -> {ans[:150]}", flush=True)
results["conditions"]["truncated_context"] = trunc_answers

# 汇总：幻觉率 = 截断后既不承认不知道、又编造内容的比例
def hallucination_rate(answers):
    hallucinated = 0
    for a in answers:
        # 幻觉定义：没承认不知道 + (提到名字或复述描述) = 编造
        if not a["admits_unknown"] and (a["mentions_name"] or a["repeats_desc"]):
            hallucinated += 1
    return hallucinated / len(answers) if answers else 0

full_hr = hallucination_rate(full_answers)
trunc_hr = hallucination_rate(trunc_answers)
results["summary"] = {
    "full_hallucination_rate": round(full_hr, 3),
    "truncated_hallucination_rate": round(trunc_hr, 3),
    "note": "完整上下文应复述事实(低幻觉)；截断后若编造=KV压缩诱发幻觉"
}

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "p12_7b_hallucination.json"), "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"\n=== 汇总 ===", flush=True)
print(f"完整上下文幻觉率: {full_hr:.1%}", flush=True)
print(f"截断上下文幻觉率: {trunc_hr:.1%}", flush=True)
print("结果保存: p12_7b_hallucination.json", flush=True)
