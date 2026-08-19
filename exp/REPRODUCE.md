# 实验复现说明 · 大模型的命根子（LLM-Lifelines）

本目录包含论文《大模型的"命根子"：拆掉10%，它崩溃136倍》全部实验的**完整代码、语料与原始结果数据**。

## 环境要求

- Python 3.10+
- `pip install transformers torch accelerate`
- 模型：Qwen2.5-0.5B / Qwen2.5-7B（开源免费）
  - 国内：ModelScope 搜索 `Qwen2.5-0.5B-Instruct` / `Qwen2.5-7B-Instruct` 下载
  - 海外：HuggingFace 同名模型

## 快速开始

1. 下载模型到本地（如 `~/models/Qwen2.5-0.5B-Instruct`）
2. 打开任意脚本，把 `MODEL_PATH` 改成你的模型路径
3. 运行脚本，结果输出为同名 `*_results.json`

## 脚本与论文内容对应

| 脚本 | 论文对应内容 |
|------|-------------|
| p1_scan.py | 注意力头扫描：三态分布（骨架/冗余/均质） |
| p6_head_pruning.py | 剪枝实验：剪骨架头崩溃136–5000倍，剪冗余头几乎无影响 |
| p8_sparse_attn.py | 稀疏注意力：结构判据 vs 范数判据（困惑度最低降38.3%） |
| p9_co_neighbor_attn.py | 共同邻居法（结构判据）稀疏注意力 |
| p4_kv_budget.py / p4v2_long_combined.py | KV缓存按结构集中度分配（75%保留比最优，降7.9%） |
| p5_generation.py / p5v2_physical_evict.py | 生成长度与物理驱逐实验 |
| p7_baseline_ladder.py / p7c_compressed_ladder.py | 基线阶梯与压缩阶梯对比 |
| p10_7b_kv_curve.json / p11_7b_gen_quality.py / p12_7b_hallucination.py | 7B模型：KV曲线 / 生成质量 / 幻觉诊断 |
| base_topk.py / base_kv_compress.py / base_connect.py | 基线方法（范数top-k / KV压缩 / 连通性） |
| exp1_* / exp2_* | 早期探索实验（PPL验证 / 合并 / 长文本） |

## 数据说明

所有 `*_results.json` 即论文表格中的原始数据，可逐项核对。语料文件（`corpus_*.txt`）为论文评测所用测试集。

---

*仅供学术交流。全部阈值为可调参数，实验结果可完整复现。*
