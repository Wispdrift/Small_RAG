# 项目结构规划与阶段实施步骤

## 1. 目标

本项目目标是搭建一个可自动化运行的长 PDF RAG 系统框架。系统默认复用当前已有 MinerU 解析缓存，也保留后续接入本地 MinerU CPU/GPU 解析的扩展入口。第一版框架应做到：

- 自动读取 MinerU 解析产物。
- 自动清洗和切分 chunk。
- 自动生成带上下文的索引文本。
- 自动构建 BM25 和轻量向量索引。
- 自动执行混合检索。
- 支持命令行提问。
- 预留 LLM API、embedding API、reranker API 接口。

第一版不依赖人工手动建立图谱、手动标注 chunk 或手工整理索引。轻量实体索引、摘要索引等增强模块应由脚本自动生成，作为后续阶段能力。

## 2. 推荐目录结构

```text
.
├── README.md
├── TASK_SOLUTION.md
├── PROJECT_STRUCTURE_AND_STEPS.md
├── qa_pairs.jsonl
├── 产品手册.pdf
├── 杂志.pdf
├── 产品手册.pdf-.../
├── 杂志.pdf-.../
├── pyproject.toml
├── .env.example
├── scripts/
│   ├── build_chunks.py
│   ├── build_index.py
│   └── ask.py
├── src/
│   └── rag_project/
│       ├── __init__.py
│       ├── config.py
│       ├── models.py
│       ├── mineru_loader.py
│       ├── text_utils.py
│       ├── chunker.py
│       ├── bm25.py
│       ├── vector_index.py
│       ├── hybrid_retriever.py
│       ├── generator.py
│       └── pipeline.py
└── data/
    ├── processed/
    │   └── chunks.jsonl
    └── index/
        ├── bm25.json
        ├── vectors.json
        └── manifest.json
```

## 3. 阶段实施路线

### 阶段 1：项目骨架

产物：

- `pyproject.toml`
- `.env.example`
- `src/rag_project/`
- `scripts/`
- `data/processed/`
- `data/index/`

目标：

- 建立可维护的 Python 包结构。
- 所有脚本从项目根目录运行。
- 所有输出集中写入 `data/`，不污染 MinerU 原始解析目录。

### 阶段 2：MinerU 解析产物读取

产物：

- `mineru_loader.py`
- `text_utils.py`

目标：

- 自动发现包含 `content_list.json`、`block_list.json`、`full.md` 的 MinerU 目录。
- 优先读取 `*_content_list.json`，因为其中包含 `page_idx`、`text_level`、`type` 等结构化信息。
- 如果 JSON 不可用，则回退到 `full.md`。
- 跳过页眉、页脚、页码、装饰性块。
- 保留正文、标题、表格、图片说明。

### 阶段 3：结构化 chunk 构建

产物：

- `chunker.py`
- `scripts/build_chunks.py`
- `data/processed/chunks.jsonl`

目标：

- 按文档、页码、标题层级生成基础 chunk。
- 对长段落进行长度控制。
- 对每个 chunk 保存元数据：文件名、页码、章节、块类型。
- 为每个 chunk 生成 contextual prefix。
- 输出 `raw_text`、`index_text`、`display_text` 三种文本。

### 阶段 4：索引构建

产物：

- `bm25.py`
- `vector_index.py`
- `scripts/build_index.py`
- `data/index/bm25.json`
- `data/index/vectors.json`
- `data/index/manifest.json`

目标：

- 构建 BM25 关键词索引。
- 构建轻量向量索引。
- 第一版使用哈希向量作为无依赖 fallback，保证没有 embedding API 时也能运行。
- 后续接入 embedding API 后，只需替换 `vector_index.py` 中的 embedder。

### 阶段 5：混合检索与命令行问答

产物：

- `hybrid_retriever.py`
- `pipeline.py`
- `generator.py`
- `scripts/ask.py`

目标：

- 对问题同时执行 BM25 和向量检索。
- 使用 RRF 融合两路结果。
- 返回 top-k 证据。
- 如果没有配置 LLM API，则输出“基于证据的抽取式回答草稿”。
- 如果配置 LLM API，则调用 LLM 生成最终答案。

### 阶段 6：API 接入

产物：

- `.env`
- `generator.py` 中的 API client

目标：

- 支持 OpenAI-compatible Chat Completions API。
- 支持配置 `LLM_BASE_URL`、`LLM_API_KEY`、`LLM_MODEL`。
- 支持后续接入 embedding API 和 reranker API。

第一版不会强制依赖 API。没有 API 时，系统仍可构建索引和返回证据。

### 阶段 7：增强模块

建议在主链路跑通后再做。

可选模块：

- Query Expansion：自动扩展产品推荐类问题中的关键词。
- 轻量摘要索引：按页、文章、产品生成摘要 chunk。
- 轻量实体索引：自动抽取实体到 chunk 的映射。
- Reranker：接入本地或 API reranker。
- 评估脚本：基于 `qa_pairs.jsonl` 计算 Recall@k。

## 4. 自动化运行方式

构建 chunk：

```bash
python scripts/build_chunks.py
```

构建索引：

```bash
python scripts/build_index.py
```

提问：

```bash
python scripts/ask.py "山东烟台开展黑广播治理时联合了哪些部门？"
```

如果配置了 LLM API：

```bash
copy .env.example .env
# 填写 LLM_API_KEY、LLM_BASE_URL、LLM_MODEL
python scripts/ask.py "对比两个干扰检测相关产品的功能差异"
```

## 5. 当前执行顺序

本轮先执行到阶段 5：

1. 新建项目结构文档。
2. 创建 Python 包结构。
3. 实现 MinerU 结构化读取。
4. 实现 chunk 构建。
5. 实现索引构建。
6. 实现混合检索。
7. 实现 CLI 问答。
8. 用现有 PDF 解析结果跑一遍构建和检索验证。

阶段 6 的 API 接入会预留代码接口，但只有在提供 API key 后才真正调用模型。

## 6. 当前已完成的检索增强

当前实现已经不只是基础 BM25 检索，而是完成了以下通用增强：

- 基于 MinerU `content_list.json` 读取正文、标题、表格、图注和页码。
- 为每个 chunk 生成 `contextual_prefix`，索引文本包含文档名、页码、章节路径和块类型。
- 修正部分 MinerU 标题层级误继承问题，将产品、设备、装置、终端、系统等标题提升为章节根节点。
- 中文检索 token 使用连续中文片段内的 unigram、bigram、trigram，避免跨标点生成无意义 n-gram。
- 自动生成多粒度知识块：原文块、标题块、页面聚合块、章节聚合块和文档大纲块。
- 检索阶段使用 BM25 + 哈希向量 fallback + RRF 融合。
- 对复杂问题进行通用子问题拆分检索，再进行融合，不依赖领域词表。
- 对 top-k 进行页级多样性重排，避免同一页或同一文章重复占满候选证据。
- 对导航类命中进行候选池内的原文证据回查，优先展示 text/table/page_summary 等可支撑答案的块。

已验证结果：

```text
Recall@5  = 3/4
Recall@10 = 4/4
```

当前仍未默认接入：

- LLM 生成。
- LLM verifier / 自动拒答。
- 真实 embedding API。
- reranker。
- LLM 生成的摘要索引。
- 轻量实体索引。

这些模块应在检索层稳定后继续接入。尤其是不可回答问题，当前只能返回候选证据；最终拒答应由检索分数、reranker 分数和 LLM verifier 共同判断。

## 7. 本轮推进记录

本轮继续沿“先知识库构建与检索层，最后再接 LLM API”的路线推进，没有把答案写死到规则里，主要增强点如下：

- 在 `BM25Index.search` 和 `VectorIndex.search` 中增加 `allowed_doc_ids` 参数，使检索层支持来源受限召回。
- 在 `HybridRetriever` 中加入显式来源提示处理：当问题出现“产品手册”“杂志”“中国无线电”等来源线索时，会额外在对应 PDF 的 chunk 集合内召回，再参与融合。
- 将来源提示从单纯软加权升级为“来源内候选优先排序”，避免用户明确要求“根据产品手册”时被杂志中的相似场景压过。
- 将 RRF 融合改为 `RRF + normalized raw score`，在保留多路召回稳定性的同时，让 BM25/向量检索中分数明显更高的强相关证据得到体现。
- 加入证据质量权重：短标题块和大纲块降权，正文、表格、页面聚合块更容易进入候选证据。
- 加入导航型问题识别：当问题是“有哪些/列举/记录了哪些/案例/文章/主题”等结构化浏览意图时，自动扩大候选池，并临时提升 outline、page_summary、section_summary 权重。
- 扩展 `qa_pairs.jsonl`，新增产品手册强来源约束题和杂志无人机监测通用要求题。
- 增强 `scripts/evaluate_retrieval.py`，除 Recall@k 外输出首个命中 rank、MRR@k 和 SourceAccuracy@1。

当前验证结果：

```text
Recall@5          = 5/6 = 0.833
MRR@5             = 0.667
SourceAccuracy@1  = 6/6 = 1.000

Recall@10         = 6/6 = 1.000
MRR@10            = 0.694
SourceAccuracy@1  = 6/6 = 1.000
```

当前仍需要继续推进的点：

- `q_02` 这类“列举典型案例”的宽泛问题经过导航型提权后已从 Top-10 第 9 位推进到第 6 位，但 Top-5 仍不稳定。下一步更适合做案例级聚合索引或 reranker，而不是继续堆关键词规则。
- 默认向量仍是 hashing fallback，已经预留 OpenAI-compatible embedding API。提供 `EMBEDDING_API_KEY`、`EMBEDDING_BASE_URL`、`EMBEDDING_MODEL` 后需要重建索引。
- reranker 还未接入。下一阶段建议加入 cross-encoder 或 API reranker，对 Top-30 候选做二阶段精排。
- 不可回答问题目前仍返回候选证据，最终拒答应放到 reranker 分数阈值和 LLM verifier 接入后处理。

## 8. 局部标题索引与导航检索修正

继续推进后，知识库构建层新增了 `local_heading_index` 块：

- 按页码窗口而不是标题数量窗口生成局部标题索引，当前使用 8 页窗口、4 页步进。
- 用于支撑“有哪些案例、有哪些文章、涉及哪些主题”等导航型问题。
- 普通事实问答和产品选型问题默认仍优先返回正文、表格、页面聚合等证据块。
- 导航型问题会扩大候选池，但 `local_heading_index` 只有覆盖查询核心词项时才提权，覆盖不足时降权，避免泛相关标题索引排到前面。

当前构建结果：

```text
Loaded blocks = 2048
Built chunks  = 1806
Indexed chunks = 1806
local_heading_index = 28
```

当前验证结果：

```text
Recall@5          = 5/6 = 0.833
MRR@5             = 0.667
SourceAccuracy@1  = 6/6 = 1.000

Recall@10         = 6/6 = 1.000
MRR@10            = 0.694
SourceAccuracy@1  = 6/6 = 1.000
```

关于 `q_02` 的说明：

- 当前 Top-5 返回的是福建厦门干扰排查培训、5G 基站干扰排查、无人机反制设备案、无线电信号屏蔽器案、多普勒气象雷达干扰排查等证据，语义上都符合“典型无线电干扰排查案例”。
- 旧 gold 只标注了 39/40/110/111/112 页，因此自动指标仍显示 `q_02` Top-5 miss、Top-10 hit。
- 不建议为了单一指标继续堆人工规则；下一步应接入真实 embedding 或 reranker，让二阶段语义精排解决宽泛列举题的排序问题。

## 9. 真实 Embedding 与 Reranker 接入

本轮已经把真实 embedding 和 reranker 接入为可选运行能力：

- `embeddings.py` 支持 OpenAI-compatible `/embeddings` 接口。
- `vector_index.py` 会校验“当前 embedding 模型”和“索引构建时 embedding 模型”是否一致；如果 `.env` 切换到真实 embedding 但没有重建索引，会直接报错，避免用真实查询向量去检索 hashing fallback 索引。
- 新增 `rerankers.py`，支持常见 `/rerank` API：请求体包含 `model`、`query`、`documents`、`top_n`，响应支持 `results` 或 `data`，每项支持 `index/document_index/id` 和 `relevance_score/score/rerank_score`。
- `HybridRetriever` 会先召回较大的候选池，再把候选交给 reranker 精排到最终 top-k。
- 未配置 `RERANKER_API_KEY` 和 `RERANKER_MODEL` 时，自动使用 no-op reranker，不影响离线运行。
- 新增 `scripts/check_runtime.py`，用于检查当前 active embedder、index embedder、active reranker 和索引状态。

配置示例：

```bash
EMBEDDING_API_KEY=your_key
EMBEDDING_BASE_URL=https://api.openai.com/v1
EMBEDDING_MODEL=text-embedding-3-small

RERANKER_API_KEY=your_key
RERANKER_BASE_URL=https://your-reranker-provider/v1
RERANKER_MODEL=your-reranker-model
RERANKER_PATH=/rerank
```

启用真实 embedding 后必须重建索引：

```bash
python scripts/check_runtime.py
python scripts/build_index.py
python scripts/evaluate_retrieval.py --top-k 10
python scripts/ask.py "《中国无线电》杂志记录了哪些典型的无线电干扰排查案例？"
```

当前离线验证结果仍基于 hashing fallback 和 no-op reranker：

```text
Recall@10         = 6/6 = 1.000
MRR@10            = 0.694
SourceAccuracy@1  = 6/6 = 1.000
```

已验证保护机制：

```text
Vector index was built with 'hashing-fallback', but current embedder is 'openai-compatible:fake-embedding-model'.
Run scripts/build_index.py after changing EMBEDDING_MODEL or EMBEDDING_API_KEY.
```

## 10. LLM API 实测记录

已使用环境变量方式测试 OpenAI-compatible Chat Completions API。注意：`aihubmix.com` 需要使用 `/v1` 路径。

运行时配置：

```bash
LLM_BASE_URL=https://aihubmix.com/v1
LLM_MODEL=deepseek-v4-pro
```

不要把 API key 写入文档或提交到代码仓库；建议仅通过当前 shell 环境变量或本地 `.env` 使用。

运行时检查结果：

```text
active_embedder: hashing-fallback
index_embedder: hashing-fallback
active_reranker: none
llm_enabled: True
llm_base_url: https://aihubmix.com/v1
llm_model: deepseek-v4-pro
chunk_count: 1806
status: ready
```

已测试通过：

- 产品手册问题可以完成“检索证据 -> LLM 基于证据生成答案”。
- 不可回答问题可以基于证据拒答，例如“2022 年全国无线电管理机构购买了多少台量子通信设备？”返回“文档中没有提供相关信息”。

已发现并确认：

- `LLM_BASE_URL=https://aihubmix.com` 会返回 `HTTP Error 401: Unauthorized`。
- `LLM_BASE_URL=https://aihubmix.com/v1` 可正常调用。

## 11. 当前完成度与增强结论

本轮已按“先完成 LLM 接入前可推进部分，再接真实模型评测”的顺序继续补强。当前系统不再只是基础 BM25 + 向量混合检索，而是具备如下链路：

- PDF 解析层：继续复用已有 MinerU 解析结果；本地侧保留 CPU/GPU 运行 MinerU 的能力即可，不需要为了形式再用 CPU 重跑一遍同一批 PDF。
- 知识库构建层：保留页级、章节级、局部标题索引，并新增 `graph_index` 结构索引块，用于组织文档、章节、页码、主题之间的轻量关系。
- 检索层：BM25 + 本地向量检索 + 查询变体 + 来源提示 + 证据质量权重 + 导航型问题扩展 + 结构索引召回。
- Embedding 层：支持 OpenAI-compatible embedding API，也支持本地 `sentence-transformers` 模型。当前已用 CPU 离线缓存模型 `sentence-transformers/all-MiniLM-L6-v2` 重建索引。
- Reranker 层：支持 API reranker 和本地 `transformers` CrossEncoder reranker；已验证 `BAAI/bge-reranker-base` 可在 CPU 加载运行，但当前小评测集上会降低 MRR，因此默认不启用。
- LLM 增强层：已实现查询改写、HyDE、LLM 生成摘要索引脚本、严格证据 verifier；默认关闭，需要环境变量显式启用。
- 生成层：LLM 只基于检索证据回答，开启 verifier 后会对答案进行二次证据支持判断，不支持则拒答。

当前默认运行状态：

```text
active_embedder: sentence-transformers:sentence-transformers/all-MiniLM-L6-v2
index_embedder: sentence-transformers:sentence-transformers/all-MiniLM-L6-v2
active_reranker: none
local_embedding_model: sentence-transformers/all-MiniLM-L6-v2
local_reranker_model: unset
llm_enabled: False
query_enhancement_enabled: False
llm_verifier_enabled: False
chunk_count: 1820
status: ready
```

当前默认离线评测结果：

```text
Recall@5          = 6/6 = 1.000
MRR@5             = 0.608
SourceAccuracy@1  = 6/6 = 1.000

Recall@10         = 6/6 = 1.000
MRR@10            = 0.608
SourceAccuracy@1  = 6/6 = 1.000
```

真实模型测试结论：

- 本地 embedding：`sentence-transformers/all-MiniLM-L6-v2` 已在 CPU 离线模式下可用，并完成索引重建。`BAAI/bge-m3` 更适合中文和多语义长文档，但本机下载曾超时，因此保留为推荐强模型配置。
- 本地 reranker：`BAAI/bge-reranker-base` 已下载并可加载；在当前 6 条样例评测上，纯 rerank 和低权重融合均未超过默认检索，因此保留可用能力但默认关闭。
- GraphRAG：本项目 PDF 数量少、关系类型主要是文档结构和主题关系，完整实体图谱式 GraphRAG 的收益预计有限。当前采用轻量 `graph_index` 更匹配任务规模，可增强“有哪些文章/案例/主题”类导航问题。
- 查询改写与 HyDE：已接入真实 LLM API 测试。为避免增强文本稀释原始问题，pipeline 已改为“原查询检索 + 增强查询检索 + 双路融合排序”。
- verifier：真实 API 测试中，文档外问题能够返回“文档中没有提供相关信息”。
- LLM 摘要索引：新增 `scripts/build_llm_summary_index.py`，支持 `--limit`、`--max-candidates`、`--dry-run`、`--retries`、`--workers`、`--force-lock`。已用真实 API 完成全量 416 个摘要块生成；脚本带单实例锁和按 `chunk_id` 合并写入，避免并发 checkpoint 覆盖已完成结果。

推荐默认策略：

- 本地无 GPU 时，不强制重跑 MinerU；保留 CPU/GPU 解析入口和复现实验说明即可。
- 默认使用“本地 embedding + 混合检索 + 结构索引 + 无 reranker”，这是当前样例集最稳的检索配置。
- 答辩展示时可开启 `ENABLE_QUERY_ENHANCEMENT=1`、`QUERY_ENHANCEMENT_MODE=hyde` 或 `rewrite`、`ENABLE_LLM_VERIFIER=1`，体现前沿 RAG 增强能力。
- `LLM 摘要索引`、`bge-m3`、`reranker` 作为可插拔增强项保留，是否全量启用应以更大验证集上的 MRR、拒答准确率和延迟成本共同决定。

## 12. 工程化交付状态

本轮完成了全量摘要索引、一键脚本、自包含路径检查和提交文档。

全量摘要索引结果：

```text
llm_summary = 416
total_chunks = 2236
indexed_chunks = 2236
```

当前默认 embedding 已切换为更适合中文检索的 `BAAI/bge-base-zh-v1.5`：

```text
active_embedder: sentence-transformers:BAAI/bge-base-zh-v1.5
index_embedder: sentence-transformers:BAAI/bge-base-zh-v1.5
chunk_count: 2236
status: ready
```

加入全量 LLM 摘要块后的当前评测：

```text
Recall@5          = 6/6 = 1.000
MRR@5             = 0.611
SourceAccuracy@1  = 6/6 = 1.000

Recall@10         = 6/6 = 1.000
MRR@10            = 0.611
SourceAccuracy@1  = 6/6 = 1.000
```

阶段判断：

- `BAAI/bge-base-zh-v1.5` 相比 MiniLM 的纯检索版本有小幅排序提升，曾达到 `MRR@5 = 0.653`。
- 加入全量 LLM 摘要块后，召回保持满分，但少数问题中聚合摘要会使精确正文页后移，当前 `MRR@5 = 0.611`。
- 因此全量摘要块应作为“语义覆盖/导航问题增强”来解释，不应简单宣称对所有精确定位问题都有提升。
- 工程上已保留摘要块、reranker、HyDE、query rewrite、verifier 的开关式能力，便于按验证集调参启用。

新增工程脚本：

```text
run.ps1
scripts/build_all.ps1
scripts/check_submission.ps1
requirements.txt
SUBMISSION_GUIDE.md
```

GitHub 提交检查：

- 源码、脚本和文档使用项目根目录推导路径，不依赖本机绝对路径。
- `data/index/`、`data/models/`、`data/pycache/`、`.env` 已加入 `.gitignore`。
- `data/index/manifest.json` 已改为相对路径。
- 文档中仅保留 `your_key` 占位示例，没有真实 API key。
