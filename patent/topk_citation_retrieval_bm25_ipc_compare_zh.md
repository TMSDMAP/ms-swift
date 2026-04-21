# topk_citation_retrieval_bm25_ipc_compare.py 中文说明

## 1. 脚本定位

该脚本用于做专利检索 Top-K 评测，对比两个向量模型在同一批查询与候选集上的排序效果：

- finetuned 模型：`--finetuned-model-path`
- base 模型：`--base-model-path`

核心目标是衡量在固定候选集合下，正样本被排到前列的能力，输出 MRR、MAP、NDCG、Hit@K、Recall@K 等指标，并给出逐查询对比结果。

脚本文件：

- `topk_citation_retrieval_bm25_ipc_compare.py`

## 2. 数据与评测对象

### 2.1 查询集来源

支持两种查询池构建方式：

1. 数据库扫描模式（默认）
- 按 `--query-year` 选查询年份记录
- 使用自引/他引或 XY 字段构建正样本候选

2. CSV 引证模式（常用）
- 通过 `--query-source-citation-csv` 指定 CSV
- 读取列：`anchor_pubno,self_pubno,other_pubno`
- 由 `anchor` 作为 query，`self/other` 作为正样本来源

说明：`--xy-citation-only` 与 `--query-source-citation-csv` 互斥。

### 2.2 候选池来源

候选专利来自查询年前窗口：

- 年份窗口：`[query_year-window_years, query_year-1]`
- 按 IPC 建立 BM25 池（同 IPC）
- 可选全局 BM25 池（跨 IPC）

### 2.3 正负样本组成

每个 query 的候选列表由以下部分拼接后去重：

- 正样本：`positive_pubnos`
- 同 IPC BM25 hard negatives：`bm25_hard_neg_pubnos`
- 同 IPC BM25 semantic negatives：`ipc_semantic_neg_pubnos`
- 同 IPC random negatives：`ipc_random_neg_pubnos`
- 跨 IPC BM25 hard negatives：`cross_ipc_hard_neg_pubnos`

## 3. 主要流程

1. 参数解析与合法性检查
2. 生成运行目录 `output_dir/run_name`
3. 构建或加载数据集产物（queries/candidates/dataset_meta）
4. 分别评测 finetuned 与 base
5. 输出汇总 JSON、逐查询 CSV、指标表 CSV

其中第 3 步和第 4 步都支持 resume。

## 4. Resume 与断点续跑机制

### 4.1 数据集层 resume

开启 `--resume` 且提供 `--run-name` 时：

- 直接加载已有 `queries.jsonl` 与 `candidates_global.jsonl`
- 读取 `dataset_meta.json` 校验关键参数一致性
- 参数不一致会报错，避免混跑

### 4.2 向量编码层 resume

编码函数 `encode_with_resume_cache` 支持分批断点续算：

- 缓存目录：`eval_finetuned_embed_cache` 或 `eval_base_embed_cache`
- 缓存文件：
  - `candidate.f32memmap` + `candidate.meta.json`
  - `query.f32memmap` + `query.meta.json`
- 若中断，可从 `completed_batches` 继续

### 4.3 评测结果层缓存

- finetuned 缓存：`eval_finetuned_cache.json`
- base 缓存：`eval_base_cache.json`

若 `--resume` 且缓存存在，直接读取，不重复评测。

## 5. 输出文件说明

运行目录下主要产物：

- `queries.jsonl`：最终查询样本与正负样本列表
- `candidates_global.jsonl`：评测涉及到的候选文本全集
- `dataset_meta.json`：数据构建元信息
- `eval_finetuned_cache.json`：finetuned 评测缓存
- `eval_base_cache.json`：base 评测缓存
- `per_query_comparison.csv`：逐 query 对比表
- `summary_metrics_table.csv`：关键指标表
- `summary_topk_retrieval.json`：完整汇总（推荐主读）

## 6. 关键参数速查

### 6.1 基础参数

- `--db-path`：SQLite 路径
- `--table`：表名，默认 `patents`
- `--finetuned-model-path`：微调模型路径
- `--base-model-path`：基座模型路径
- `--output-dir`：输出根目录
- `--run-name`：运行名
- `--resume`：启用断点续跑

### 6.2 查询与窗口

- `--query-year`
- `--window-years`
- `--query-size`
- `--query-pool-multiplier`
- `--max-query-scan`

### 6.3 数据来源模式

- `--query-source-citation-csv`
- `--xy-citation-only`
- `--xy-categories`
- `--citation-pubno-field`
- `--citation-category-field`

### 6.4 负样本构成

- `--hard-neg-per-query`
- `--ipc-semantic-per-query`
- `--ipc-random-per-query`
- `--cross-ipc-hard-per-query`
- `--bm25-top-k`
- `--cross-ipc-bm25-top-k`
- `--bm25-pool-per-ipc`
- `--global-bm25-pool-size`

### 6.5 编码与推理

- `--device`
- `--batch-size`
- `--max-length`
- `--rank-chunk-size`

## 7. 典型用法

### 7.1 新任务

```bash
python topk_citation_retrieval_bm25_ipc_compare.py \
  --db-path /home/ljh/data1/patent/patent.sqlite \
  --table patents \
  --query-year 2018 \
  --window-years 3 \
  --query-size 1500 \
  --query-source-citation-csv /home/ljh/data1/patent/eval_outputs/xy_q5000_2018_xy/xy_query_source_2018_xy_q5000.csv \
  --hard-neg-per-query 50 \
  --ipc-semantic-per-query 300 \
  --ipc-random-per-query 300 \
  --cross-ipc-hard-per-query 50 \
  --bm25-top-k 200 \
  --cross-ipc-bm25-top-k 600 \
  --bm25-pool-per-ipc 5000 \
  --global-bm25-pool-size 600000 \
  --exclude-pubno-file /home/ljh/data1/patent/eval_outputs/exclude_pubnos_trainval_full_20260405.txt \
  --finetuned-model-path /data4/ljh/output/Qwen3_topk_alpha1/v0-20260419-215609/checkpoint-1250 \
  --base-model-path /home/ljh/data1/Qwen3-embedding-4B \
  --output-dir /data4/ljh/patent/eval_outputs \
  --run-name topk_citation_bm25_ipc_q1500_n1000_y2018_xy_seed45_alpha1_ckpt1250_q5000pool \
  --seed 45 \
  --device cuda:0 \
  --batch-size 96 \
  --max-length 768 \
  --rank-chunk-size 64
```

### 7.2 中断后续跑

在同样参数基础上加：

```bash
--resume
```

并保证：

- `--run-name` 指向同一运行目录
- 关键参数与已有 `dataset_meta.json` 一致

## 8. 指标解读建议

- `mean_rank` 越低越好
- `MRR / MAP / NDCG / hit@k / recall@k` 越高越好
- `paired_comparison` 中：
  - `wins_ft_better` > `losses_ft_worse` 说明 finetuned 更优
  - `sign_test_pvalue_approx` 越小，差异越显著

建议优先结合以下三组看结论：

- `mrr_delta_ft_minus_base`
- `map_delta_ft_minus_base`
- `ndcg_delta_ft_minus_base`

再配合 `wins/losses/ties` 判断稳定性。

## 9. 常见注意事项

1. 若 `--cross-ipc-hard-per-query > 0`，必须设置 `--global-bm25-pool-size > 0`
2. `--xy-citation-only` 与 `--query-source-citation-csv` 不能同时开启
3. resume 模式下参数不一致会被拒绝，这是设计行为
4. 大规模编码时间长，建议后台运行并持续记录日志

## 10. 版本与维护建议

- 每次评测建议固定 `run-name` 命名规则，便于 resume 与追溯
- 统一保留 `summary_topk_retrieval.json` 作为主结果入口
- 合并多 seed 时，优先做 pooled 的 wins/losses 与平均 delta 分析
