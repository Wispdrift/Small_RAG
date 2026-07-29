# 数据检索与 RAG 项目完整方案

## 1. 任务理解

本项目要求基于给定的两个长篇 PDF 构建一个面向中文长文档的 RAG 问答系统。任务评价重点不只是系统能否回答少量样例问题，而是系统在真实长文档问答场景下是否具备合理、可复现、可解释的检索与生成设计。

输入材料包括：

- `产品手册.pdf`：49 页，内容以公司介绍、产品体系、功能说明、技术参数、应用场景为主。该文档结构相对清晰，问题通常会涉及产品名称、功能对比、参数约束、应用推荐等。
- `杂志.pdf`：132 页，内容以杂志栏目、行业资讯、案例报道、专项活动为主。该文档版式更复杂，存在跨页文章、多栏目排版、图片说明、案例集合等情况，检索难度更高。
- `qa_pairs.jsonl`：样例问答数据，用于理解问题类型、答案风格、证据粒度和拒答要求。该文件不是最终测试集，不应针对样例问题做硬编码规则。

系统需要完成以下目标：

1. 对长 PDF 进行解析、清洗、切分、索引和检索。
2. 根据用户问题从 PDF 中定位相关证据。
3. 基于证据生成准确、可追溯的中文答案。
4. 对文档中没有答案的问题明确拒答，避免幻觉。
5. 在答案中尽量给出文件名、页码、章节或证据片段。

因此，本项目的核心难点在于：

- PDF 解析结果是否保留了足够准确的文本、页码和版面结构。
- 切分策略是否适合长文档、跨页文章、表格参数和图片说明。
- 检索策略是否能兼顾语义问题和精确关键词问题。
- 生成阶段是否严格依据证据，且能够输出可追溯引用。
- 系统是否具备可复现的数据处理流程，而不是只依赖一次性的手工结果。

## 2. 总体技术路线

推荐采用“结构化 PDF 解析 + 混合检索 + 重排序 + 受约束生成”的 RAG 架构。

整体流程如下：

```text
原始 PDF
  -> MinerU 结构化解析
  -> Markdown / JSON / 图片 / 版面块
  -> 文本清洗与元数据补全
  -> 按页码、标题、段落、表格切分 chunk
  -> 建立向量索引 + 关键词索引
  -> 用户问题改写或扩展
  -> Hybrid Retrieval 初召回
  -> Reranker 精排
  -> 证据充分性判断
  -> LLM 生成答案或拒答
  -> 输出答案 + 引用依据
```

该方案的关键思想是：PDF 解析阶段尽量保留文档结构，检索阶段同时使用语义和关键词信号，生成阶段严格受证据约束。

## 3. PDF 解析方案

### 3.1 解析工具选择

PDF 解析建议采用 MinerU。原因如下：

- 支持复杂 PDF 的版面分析，不只是简单抽取文本。
- 能输出 Markdown、JSON、图片和版面模型结果。
- 能较好保留阅读顺序、标题层级、表格、图片说明等信息。
- 支持 CPU 和 GPU 两种运行方式，便于兼顾可复现性和效率。

当前项目目录中已经存在 MinerU 解析产物，例如：

```text
产品手册.pdf-.../
  full.md
  layout.json
  block_list.json
  *_content_list.json
  *_content_list_v2.json
  *_model.json
  images/

杂志.pdf-.../
  full.md
  layout.json
  block_list.json
  *_content_list.json
  *_content_list_v2.json
  *_model.json
  images/
```

这些文件可以作为默认的解析缓存直接用于后续索引构建。

### 3.2 网页高性能 MinerU 解析结果是否可用

可以使用，但需要在项目文档中说明其定位。

推荐表述为：

> PDF 解析阶段采用 MinerU。考虑到本地环境无 GPU，实验默认使用高性能 MinerU 服务生成的结构化解析缓存；同时系统保留本地 MinerU 解析入口，支持 CPU pipeline 与 GPU 加速两种模式，保证从原始 PDF 到 Markdown/JSON 结构化文档的流程可复现。

这样做的优势是：

- 网页端高性能 MinerU 解析结果质量较高，可以减少本地耗时。
- 项目仍然保留本地可复现能力，不依赖网页服务才能运行。
- 本地无 GPU 时可以使用 CPU 模式复现，只是速度较慢。
- 有 GPU 的评测环境可以直接切换到 GPU 加速解析。

不建议把网页解析结果描述成不可复现的手工中间结果。更合理的定位是：它是由 MinerU 生成的结构化缓存产物，项目默认复用该缓存；必要时可以从原始 PDF 重新生成。

### 3.3 是否必须本地 CPU 重跑

不必须。

如果当前目标是完成 RAG 系统设计、实现和答辩，本地 CPU 重跑一版收益不高。CPU 解析可能耗时较长，而且不同环境、不同 MinerU 版本还可能产生轻微差异。

更重要的是项目应保留以下能力：

- `--device cpu`：无 GPU 环境下可运行。
- `--device cuda`：有 NVIDIA GPU 时加速运行。
- `--device auto`：自动检测 GPU，不可用时回退到 CPU。
- 已解析结果存在时默认跳过解析，直接使用缓存。
- 提供 `--force` 参数用于强制重新解析。

### 3.4 推荐解析脚本行为

后续实现时可提供如下命令：

```bash
python scripts/parse_with_mineru.py --input data/raw --output data/parsed --device auto
python scripts/parse_with_mineru.py --input data/raw --output data/parsed --device cpu
python scripts/parse_with_mineru.py --input data/raw --output data/parsed --device cuda
python scripts/parse_with_mineru.py --input data/raw --output data/parsed --device auto --force
```

脚本逻辑建议：

1. 扫描原始 PDF。
2. 检查目标解析目录是否已存在 `full.md`、`layout.json`、`content_list.json`。
3. 若缓存完整且未指定 `--force`，跳过解析。
4. 若需要解析，根据 `device` 参数选择 CPU 或 GPU。
5. 解析完成后记录解析配置、时间、MinerU 版本和输出文件路径。

## 4. 数据清洗与规范化

MinerU 输出不能直接原样全部入库，需要做轻量清洗和规范化。

### 4.1 清洗目标

清洗不是为了改写原文，而是为了让检索单元更稳定。

需要处理的问题包括：

- 页眉、页脚、重复目录、装饰性文本。
- Markdown 图片路径噪声。
- 空行过多、断行异常、乱码符号。
- 表格内容被拆散。
- 图片说明与上下文分离。
- 标题层级丢失或不稳定。

### 4.2 保留信息

以下信息不应轻易删除：

- 产品名称、型号、参数、单位。
- 页码、标题、章节、栏目名。
- 图片标题、表格标题。
- 案例中的地点、部门、时间、事件。
- 原文中的关键数值，如 `15s`、`1Hz`、`-25℃~+60℃` 等。

长文档问答中，很多问题依赖精确参数和命名实体。如果过度清洗，反而会降低召回。

### 4.3 元数据设计

每个 chunk 建议保存如下元数据：

```json
{
  "chunk_id": "product_manual_p020_003",
  "source_file": "产品手册.pdf",
  "doc_type": "product_manual",
  "page_start": 20,
  "page_end": 20,
  "section": "欺骗式/压制式干扰检测设备",
  "block_type": "text",
  "text": "支持 GNSS 欺骗式、压制式干扰检测...",
  "image_refs": [],
  "token_count": 423
}
```

对于跨页内容：

```json
{
  "chunk_id": "magazine_p111_p112_001",
  "source_file": "杂志.pdf",
  "doc_type": "magazine",
  "page_start": 111,
  "page_end": 112,
  "section": "干扰排查",
  "block_type": "article",
  "text": "...",
  "token_count": 687
}
```

页码和文件名必须进入元数据，不能只存在于文本中。这样最终答案才能稳定输出引用。

## 5. Chunk 切分策略

### 5.1 为什么不能只按固定长度切分

固定长度切分简单，但对本任务不够理想：

- 产品手册中一个产品的功能、参数、应用场景可能被切断。
- 杂志文章可能跨页，固定切分容易丢失上下文。
- 表格参数被拆开后，问题问到参数时很难召回完整证据。
- 图片标题、栏目标题与正文分离后，证据引用质量下降。

因此推荐使用结构优先的切分策略。

### 5.2 推荐切分规则

综合使用以下规则：

1. 优先按文档、页码、标题层级组织。
2. 标题下的短段落可以合并。
3. 长段落按 300-800 中文字切分。
4. 相邻 chunk 保留 50-150 字 overlap。
5. 表格作为独立 chunk，必要时增加表格标题和页码上下文。
6. 图片说明可合并到相邻正文，也可作为 caption chunk。
7. 跨页文章允许 `page_start` 和 `page_end` 不同。

推荐 chunk 大小：

- 初始召回 chunk：300-800 中文字。
- 上下文扩展 chunk：召回命中后可拼接同页前后 chunk。
- 生成上下文：控制在 LLM 上下文窗口内，优先放重排分最高的证据。

### 5.3 文档类型差异化策略

对 `产品手册.pdf`：

- 按产品名称、章节标题、参数表切分。
- 参数表和功能说明尽量不拆散。
- 保留产品所属类别和应用场景。
- 对型号、指标、时间、频率、温度范围等建立关键词索引。

对 `杂志.pdf`：

- 按栏目、文章标题、页码切分。
- 跨页文章需要合并上下文。
- 案例类内容保留地点、机构、事件、处理结果。
- 对专栏名称、地名、部门、行业术语建立关键词索引。

## 6. 检索方案

### 6.1 采用混合检索

本任务不建议只使用向量检索。原因是 PDF 中有大量专有名词、产品名称、数字参数和地名机构名，纯语义检索容易遗漏精确匹配。

推荐采用 Hybrid Retrieval：

```text
用户问题
  -> 向量检索 top_k_semantic
  -> BM25 / 关键词检索 top_k_keyword
  -> 合并去重
  -> Reranker 重排
  -> 选取 top_n 作为证据
```

向量检索负责解决语义表达差异，例如“适合户外长期部署的设备”与原文“室外工作温度范围”之间的语义关联。

关键词检索负责解决精确实体匹配，例如“山东烟台”“黑广播”“GSM-R”“GPS-L1”“BDS-B1”“15s”等。

### 6.2 向量模型选择

中文 RAG 推荐使用中文或多语言 embedding 模型，例如：

- `bge-large-zh-v1.5`
- `bge-m3`
- `m3e-base`
- 其他支持中文语义检索的 embedding API

如果本地资源有限，可以优先选轻量模型；如果追求效果，可以选更强的多语言或中文向量模型。

### 6.3 关键词检索

关键词检索可以使用：

- BM25
- Elasticsearch / OpenSearch
- SQLite FTS
- Whoosh
- `rank-bm25`

中文场景建议接入分词，或至少做字符级/词级混合处理。对产品型号、英文缩写、数字参数要避免被错误切碎。

### 6.4 重排序

初召回后建议使用 reranker 精排。原因是初召回通常会包含相似但不真正支持答案的片段，尤其是产品手册中多个产品描述相似，杂志中多个案例主题相近。

可选 reranker：

- `bge-reranker-base`
- `bge-reranker-large`
- 支持中文的 cross-encoder reranker
- LLM 小规模重排

推荐流程：

1. 向量检索取 top 20。
2. BM25 检索取 top 20。
3. 合并去重后得到 20-40 个候选。
4. Reranker 排序。
5. 取 top 5-8 个进入生成。

## 7. 问题类型处理

样例问题体现出至少三类题型。

### 7.1 单点事实题

示例：山东烟台开展打击治理“黑广播”专项行动时，联合了哪些部门？

处理方式：

- 优先通过关键词召回“山东烟台”“黑广播”“联合”等片段。
- 需要精确返回部门名称。
- 答案应简洁，并附引用页码。

### 7.2 综合归纳题

示例：《中国无线电》杂志记录了哪些典型的无线电干扰排查案例？

处理方式：

- 需要召回多个页面、多个案例。
- 不能只返回 top1 片段。
- 可按案例列表生成答案。
- 每个案例最好附对应页码或栏目。

### 7.3 对比分析题

示例：对比两个产品的功能、告警能力、应用场景。

处理方式：

- 分别召回两个产品的证据。
- 生成阶段按维度对比。
- 如果某一维度证据不足，应明确说明未找到对应信息，而不是补全。

### 7.4 推荐决策题

示例：根据需求推荐最适合产品，并结合功能和参数说明理由。

处理方式：

- 将需求拆成多个约束条件。
- 分别检索“发现异常”“定位信号来源”“短时间反馈”“室外部署”等证据。
- 最终推荐必须能逐条对应证据。

### 7.5 不可回答题

示例：2022 年全国无线电管理机构购买了多少台量子通信设备？

处理方式：

- 检索结果如果只包含相似背景但没有明确答案，应拒答。
- 不能因为出现“2022”“无线电管理机构”“量子通信”等词就编造数量。
- 推荐回答：“文档中没有提供相关信息。”

## 8. 拒答与幻觉控制

拒答机制是本任务的重要评分点。

建议采用两层判断：

### 8.1 检索分数判断

如果 top-k 检索结果分数整体过低，或 reranker 分数低于阈值，则判定证据不足。

示例规则：

```text
if top1_rerank_score < threshold:
    return "文档中没有提供相关信息。"
```

阈值需要通过样例问题和人工问题调试，不宜固定照搬。

### 8.2 LLM 证据充分性判断

生成前让 LLM 先判断证据是否足够：

```text
请判断给定证据是否足以回答用户问题。
如果证据没有直接给出答案，输出 insufficient。
如果足够，输出 sufficient，并列出支持答案的证据编号。
```

只有 `sufficient` 时才进入正式回答。

### 8.3 生成提示词约束

生成 prompt 应明确要求：

- 只能依据给定证据回答。
- 不允许使用常识补充文档外信息。
- 如果证据不足，必须回答“文档中没有提供相关信息”。
- 答案中包含引用来源。

示例：

```text
你是一个基于 PDF 文档的问答助手。
请只依据下方证据回答问题。
如果证据中没有明确答案，请回答：文档中没有提供相关信息。
不要编造文档外信息。
回答后给出引用，引用格式为：[文件名, 页码, 章节]。
```

## 9. 生成答案格式

推荐输出格式：

```text
答案：
...

依据：
1. [产品手册.pdf, 第 20 页, 欺骗式/压制式干扰检测设备] ...
2. [产品手册.pdf, 第 22 页, 北斗授时安全隔离防护装置] ...
```

对于简单事实题，可以更简洁：

```text
烟台市工信局联合公安、综合执法、广电等部门开展了该专项行动。

引用：[杂志.pdf, 第 111 页, 山东烟台：重拳治理私设“黑广播”违法犯罪活动]
```

对于不可回答题：

```text
文档中没有提供相关信息。

说明：检索到的片段未包含“2022 年全国无线电管理机构购买量子通信设备数量”的明确信息。
```

## 10. 系统模块设计

推荐项目实现时拆分为以下模块：

```text
rag_project/
  data/
    raw/
    parsed/
    processed/
    index/
  scripts/
    parse_with_mineru.py
    build_chunks.py
    build_index.py
    evaluate.py
  src/
    config.py
    pdf_parser.py
    cleaner.py
    chunker.py
    retriever.py
    reranker.py
    generator.py
    pipeline.py
  app.py
  README.md
```

### 10.1 `pdf_parser.py`

职责：

- 调用 MinerU 或读取已有 MinerU 缓存。
- 统一不同 PDF 的解析产物路径。
- 输出标准化文档对象。

### 10.2 `cleaner.py`

职责：

- 清理重复页眉页脚。
- 规范 Markdown 文本。
- 保留表格、图片说明、标题。

### 10.3 `chunker.py`

职责：

- 基于页码、标题、段落和表格切分。
- 生成 chunk id。
- 写入元数据。

### 10.4 `retriever.py`

职责：

- 建立向量索引。
- 建立 BM25 或关键词索引。
- 执行混合检索。
- 合并去重候选片段。

### 10.5 `reranker.py`

职责：

- 对候选片段进行精排。
- 输出证据分数和排序。

### 10.6 `generator.py`

职责：

- 构造证据上下文。
- 执行证据充分性判断。
- 调用 LLM 生成答案。
- 格式化引用。

### 10.7 `pipeline.py`

职责：

- 串联检索、重排、判断、生成。
- 提供统一 `answer(question)` 接口。

## 11. 数据库与索引选择

### 11.1 轻量实现

如果项目以课程实践为主，可采用：

- FAISS / Chroma：向量索引。
- `rank-bm25`：关键词检索。
- JSONL / SQLite：保存 chunk 和元数据。

优点：

- 本地运行简单。
- 部署成本低。
- 便于调试和答辩展示。

### 11.2 工程化实现

如果希望更接近生产系统，可采用：

- Milvus / Qdrant：向量数据库。
- Elasticsearch / OpenSearch：关键词检索。
- SQLite / PostgreSQL：元数据管理。

但对于当前两个 PDF 的规模，轻量实现已经足够。工程复杂度应服务于任务目标，不必为了堆技术而引入过重组件。

## 12. 实验与评估设计

项目报告中建议包含检索和问答两类评估。

### 12.1 检索评估

基于 `qa_pairs.jsonl` 中的 `gold_chunks`，可以评估：

- Recall@k：正确证据是否出现在 top-k。
- MRR：正确证据排名是否靠前。
- Hit Rate：是否命中正确页码或正确章节。

示例：

```text
Recall@5 = 命中 gold page 或 gold chunk 的问题数 / 可回答问题总数
MRR = 平均第一个正确证据排名倒数
```

由于样例数量很少，指标只能作为调试参考，不能代表最终泛化效果。

### 12.2 问答评估

可以从以下维度人工评估：

- 答案正确性：是否与文档一致。
- 证据支持性：答案中的关键结论是否能被引用片段支持。
- 完整性：综合题是否覆盖多个关键点。
- 拒答准确性：不可回答问题是否拒答。
- 引用准确性：页码、文件名、章节是否正确。

### 12.3 消融实验

为了体现设计动机，可以做简单消融：

1. 仅向量检索 vs 混合检索。
2. 无 reranker vs 有 reranker。
3. 固定长度切分 vs 结构化切分。
4. 不启用拒答阈值 vs 启用拒答阈值。

报告中重点说明这些设计如何提升检索效果和问答质量。

## 13. 推荐实现优先级

如果时间有限，建议按以下优先级推进：

1. 使用现有 MinerU 解析结果构建 chunk。
2. 实现 chunk 元数据和引用输出。
3. 实现向量检索。
4. 增加 BM25 关键词检索。
5. 增加 reranker。
6. 增加拒答机制。
7. 增加本地 MinerU 解析脚本。
8. 编写评估脚本和实验结果。
9. 做简单 Web UI 或命令行问答界面。

其中第 1-6 项直接决定系统效果，第 7 项决定可复现性，第 8-9 项决定展示和答辩完整度。

## 14. 答辩时的关键说法

可以围绕以下几点展开：

1. 任务不是简单 PDF QA，而是面向长文档的可追溯 RAG。
2. PDF 解析采用 MinerU，保留 Markdown、JSON、版面块和图片信息。
3. 网页高性能 MinerU 结果作为缓存使用，同时保留本地 CPU/GPU 解析入口。
4. 切分不是固定长度暴力切，而是结合页码、标题、段落、表格和文档类型。
5. 检索采用向量检索与关键词检索结合，兼顾语义匹配和精确实体匹配。
6. Reranker 用于提升证据排序质量。
7. 生成阶段只依据证据回答，并通过阈值和证据充分性判断控制幻觉。
8. 答案输出包含文件名、页码和章节，保证可追溯。
9. 对不可回答问题明确拒答。
10. 使用样例数据进行流程调试和检索评估，但不针对样例硬编码。

## 15. 风险与应对

### 15.1 PDF 解析错误

风险：

- OCR 错字。
- 阅读顺序错乱。
- 表格结构丢失。
- 页码对应错误。

应对：

- 优先使用 MinerU 的结构化 JSON，而不是只用 Markdown。
- 对关键页进行人工抽样检查。
- chunk 中保留页码和原始片段，便于追溯。

### 15.2 检索召回不足

风险：

- 向量检索漏掉数字、型号、机构名。
- 关键词检索漏掉语义改写问题。

应对：

- 使用混合检索。
- 对问题进行关键词抽取。
- 对 top-k 结果做上下文扩展。

### 15.3 生成幻觉

风险：

- LLM 根据常识补充文档外信息。
- 检索证据相似但不支持答案。

应对：

- 证据充分性判断。
- 严格 prompt 约束。
- 低分拒答。
- 输出引用，便于检查。

### 15.4 可复现性不足

风险：

- 只使用网页解析结果，别人无法复现。

应对：

- 保留原始 PDF。
- 保留 MinerU 缓存。
- 提供本地 CPU/GPU 解析脚本。
- 记录解析版本和参数。

## 16. 最终推荐方案摘要

本项目建议采用 MinerU 进行 PDF 结构化解析，默认复用当前高性能 MinerU 网页端生成的解析缓存，同时保留本地 CPU/GPU 可运行的解析入口。解析结果经清洗后，按文档、页码、标题、段落、表格和图片说明构建带元数据的 chunk。检索阶段采用向量检索与 BM25 关键词检索融合，并通过 reranker 对候选证据精排。生成阶段使用受约束 prompt 和证据充分性判断，确保答案只来自文档，并在证据不足时明确拒答。最终答案输出文件名、页码、章节和证据片段，实现可追溯、可解释、可复现的长文档 RAG 问答系统。

## 17. 可借鉴的 RAG 前沿研究与落地设计

结合近期 RAG 研究和工程实践，本项目可以在基础 RAG 之上吸收一些更先进但仍可落地的设计。需要注意的是，本项目只有两个 PDF，数据规模不大，因此不应盲目堆复杂架构。更合理的策略是：把前沿方法转化为轻量、可解释、可评估的模块。

### 17.1 Contextual Retrieval：给 chunk 补上下文

Anthropic 在 Contextual Retrieval 中指出，传统 RAG 把文档切成孤立 chunk 后，会丢失文档级上下文，导致检索时找不到本来相关的片段。它提出在每个 chunk 前添加一段由文档上下文生成的说明，再分别用于 embedding 和 BM25 索引。该方法还强调 embeddings + BM25 + reranking 的组合效果更好。

对本项目非常适合借鉴。

原因是：

- `产品手册.pdf` 中大量 chunk 单独看可能只出现“该设备”“本产品”“告警时间”，如果没有产品名和章节背景，检索容易失败。
- `杂志.pdf` 中案例报道可能跨页，单个 chunk 可能只有处理过程，没有栏目、地点、事件主题。
- 用户问题经常问“某产品”“某案例”“某类应用场景”，上下文说明能显著提高召回。

建议在 chunk 入库前生成一个 `contextual_prefix`：

```text
该片段来自《产品手册.pdf》第 20 页，章节为“欺骗式/压制式干扰检测设备”，主要描述 GNSS 欺骗式和压制式干扰检测设备的功能、告警时间、测向精度和应用场景。
```

然后构造两份文本：

```json
{
  "raw_text": "支持 GNSS 欺骗式、压制式干扰检测...",
  "index_text": "该片段来自《产品手册.pdf》第 20 页，章节为...。支持 GNSS 欺骗式、压制式干扰检测...",
  "display_text": "支持 GNSS 欺骗式、压制式干扰检测..."
}
```

其中：

- `index_text` 用于 embedding 和 BM25。
- `display_text` 用于最终证据展示。
- `raw_text` 用于保留原文。

这样既提升检索，又不会污染最终引用内容。

参考：

- Anthropic, Introducing Contextual Retrieval: https://www.anthropic.com/engineering/contextual-retrieval
- Anthropic Cookbook, Enhancing RAG with Contextual Retrieval: https://platform.claude.com/cookbook/capabilities-contextual-embeddings-guide

### 17.2 Late Chunking：先看全文上下文，再形成 chunk 表征

Late Chunking 的核心思想是：不要先把长文档切成小块再分别 embedding，而是先用长上下文 embedding 模型编码较长文本，再在 token 表征层面切分并池化成 chunk embedding。这样 chunk embedding 能保留周围上下文信息。

对本项目的启发是：即使不完整实现 Late Chunking，也应避免让 chunk 完全孤立。

可落地方案分两档：

1. 轻量方案：使用 Contextual Retrieval，为每个 chunk 补充文档名、页码、章节、产品名、栏目名。
2. 进阶方案：如果 embedding 模型支持 8k 或更长上下文，可以按“章节/文章”为单位编码，再产出带上下文的 chunk embedding。

本项目优先采用轻量方案即可。完整 Late Chunking 对模型接口和实现要求更高，不是必要项。

参考：

- Late Chunking 论文页：https://huggingface.co/papers/2409.04701
- Jina AI 技术说明：https://jina.ai/news/late-chunking-in-long-context-embedding-models/

### 17.3 RAPTOR：构建“原文 chunk + 层级摘要”的树形索引

RAPTOR 提出递归地对 chunk 进行 embedding、聚类和摘要，形成树形检索结构。传统 RAG 往往只检索短片段，难以回答需要整体理解或多步整合的问题；RAPTOR 通过不同抽象层级的摘要改善长文档问答。

对本项目的价值主要体现在综合题：

- “杂志记录了哪些典型干扰排查案例？”
- “产品体系中不同类别有什么区别？”
- “某类设备适用于哪些场景？”

这些问题不一定能靠单个 chunk 回答，需要跨页、跨章节整合。

建议实现一个轻量版 RAPTOR：

```text
Level 0：原始 chunk
Level 1：每页摘要 / 每篇文章摘要 / 每个产品摘要
Level 2：每个栏目摘要 / 每个产品类别摘要
```

检索时：

1. 先判断问题类型。
2. 如果是单点事实题，优先检索 Level 0。
3. 如果是综合归纳题，先检索 Level 1/2 摘要，再回到 Level 0 找支撑证据。
4. 最终答案仍引用原始页码和原文 chunk，摘要只用于导航，不作为唯一证据。

参考：

- RAPTOR, ICLR 2024: https://proceedings.iclr.cc/paper_files/paper/2024/hash/8a2acd174940dbca361a6398a4f9df91-Abstract-Conference.html

### 17.4 GraphRAG：适合“全局主题”和“实体关系”问题

Microsoft GraphRAG 针对传统 RAG 不擅长回答全局性问题的问题，提出从文本中抽取实体、关系和 claims，构建知识图谱，再进行社区发现和社区摘要。其查询方式包括 Local Search、Global Search、DRIFT Search 和 Basic Search。

对本项目而言，完整 GraphRAG 成本偏高，但思想可以借鉴：

- 对 `杂志.pdf` 抽取地点、机构、部门、案例、技术术语之间的关系。
- 对 `产品手册.pdf` 抽取产品、功能、参数、应用场景之间的关系。
- 对综合问题使用“实体导航 + 原文证据检索”。

推荐实现轻量实体索引，而不是完整知识图谱系统：

```json
{
  "entity": "山东烟台",
  "type": "location",
  "related_entities": ["黑广播", "公安", "综合执法", "广电"],
  "source_chunks": ["magazine_p111_002"]
}
```

可以用于：

- 查询扩展：用户问“烟台专项行动”，自动扩展“山东烟台、黑广播、公安、广电”。
- 多跳检索：先找到实体，再找到关联 chunk。
- 答案检查：答案中的实体是否来自证据。

不建议在本项目第一版中完整实现 GraphRAG 的社区检测、社区报告和全局 map-reduce，因为两个 PDF 的规模较小，成本和复杂度可能超过收益。可以在报告中把它作为扩展方案。

参考：

- Microsoft GraphRAG 文档：https://microsoft.github.io/graphrag/
- GraphRAG Query Overview：https://microsoft.github.io/graphrag/query/overview/
- GraphRAG Global Search：https://microsoft.github.io/graphrag/query/global_search/
- From Local to Global: A Graph RAG Approach to Query-Focused Summarization: https://arxiv.org/abs/2404.16130

### 17.5 CRAG / Self-RAG：检索质量判断与自我校验

CRAG 的核心思想是：RAG 不能假设检索结果一定正确，应使用轻量检索评估器判断检索质量，并在检索结果差时触发纠正动作。Self-RAG 则强调模型应判断是否需要检索、检索结果是否相关、生成内容是否被证据支持。

本项目可以借鉴其“检索后校验”思想，而不需要训练 Self-RAG 模型。

推荐增加三个判断器：

```text
need_retrieval：该问题是否需要检索文档？
is_relevant：检索片段是否与问题相关？
is_supported：答案是否被给定证据支持？
```

具体落地：

1. 所有任务默认需要检索，因为本项目是文档问答。
2. 对 top-k 证据做相关性评分。
3. 如果相关性不足，进入拒答。
4. 生成答案后，再做一次支持性检查。
5. 如果答案中的关键结论没有证据支持，则改写为拒答或保守回答。

这对不可回答问题尤其重要。

参考：

- CRAG: Corrective Retrieval Augmented Generation: https://huggingface.co/papers/2401.15884
- Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection: https://doi.org/10.48550/arXiv.2310.11511

### 17.6 HyDE：针对抽象需求题做查询扩展

HyDE 的思想是先让 LLM 根据问题生成一个“假想答案/假想文档”，再用这个假想文档去做向量检索。它主要解决 query 和 document 表达不一致的问题。

本项目中可以选择性使用 HyDE，尤其适合推荐类和抽象类问题：

- “某单位需要建设一套无线电安全保障系统，应该推荐什么产品？”
- “哪些案例体现了无线电干扰排查能力？”
- “哪些设备适合长期室外部署？”

这些问题和原文表达差异较大。原文可能写的是“工作温度 -25℃~+60℃”“测向定位”“告警时间 ≤1 分钟”，用户问题却写成“长期部署”“快速反馈”“定位来源”。

推荐做轻量 Query Expansion，而不是完整 HyDE：

```text
用户问题：
需要发现卫星导航异常、快速定位异常信号来源、短时间反馈、支持长期室外部署，推荐什么产品？

扩展查询：
GNSS 欺骗式 压制式 干扰检测 测向定位 告警时间 室外工作温度 长期部署 产品推荐
```

检索时同时使用：

- 原始问题。
- 关键词扩展问题。
- 可选的假想答案 embedding。

参考：

- HyDE 官方代码仓库：https://github.com/texttron/hyde
- HyDE 论文页：https://arxiv.org/abs/2212.10496

### 17.7 Lost in the Middle：生成上下文排序不能随意

长上下文模型并不总能均匀利用所有位置的信息。相关研究指出，模型可能对上下文开头和结尾的信息更敏感，中间位置的信息更容易被忽略。

对本项目的启发是：不要把一大堆证据按检索顺序随意塞进 prompt。

建议：

- 最关键证据放在上下文开头。
- 次关键证据放在结尾附近。
- 对比题按实体分组排列，而不是混排。
- 综合题先放摘要导航，再放原文证据。
- 控制进入生成的 chunk 数量，避免无关片段稀释注意力。

参考：

- Google Research, Found in the Middle: https://research.google/pubs/found-in-the-middle-calibrating-positional-attention-bias-improves-long-context-utilization/
- Microsoft Research, Make Your LLM Fully Utilize the Context: https://www.microsoft.com/en-us/research/publication/make-your-llm-fully-utilize-the-context/

## 18. 融合前沿方法后的推荐处理流程

结合上述研究，最终推荐把解析后的文本按以下流程逐步处理：

```text
MinerU 解析产物
  -> 读取 full.md / content_list / layout / block_list
  -> 修正页码、标题、表格、图片说明
  -> 生成基础 chunk
  -> 为每个 chunk 生成 contextual_prefix
  -> 得到 index_text / display_text / raw_text
  -> 构建 Level 0 原文 chunk 索引
  -> 构建 Level 1 页面/文章/产品摘要索引
  -> 可选构建轻量实体索引
  -> embedding 检索 + BM25 检索
  -> RRF 融合
  -> reranker 精排
  -> 检索质量判断
  -> 必要时 query expansion / HyDE-style expansion
  -> 组织生成上下文
  -> 答案生成
  -> 证据支持性检查
  -> 输出答案和引用
```

推荐优先级如下：

| 方法 | 是否建议做 | 对本项目的价值 | 实现成本 |
| --- | --- | --- | --- |
| MinerU 结构化解析 | 必做 | 决定文本质量和页码引用质量 | 中 |
| Contextual Retrieval | 强烈建议 | 显著缓解孤立 chunk 缺上下文问题 | 低-中 |
| Embedding + BM25 混合检索 | 强烈建议 | 兼顾语义检索和精确实体/参数匹配 | 中 |
| Reranker | 强烈建议 | 提升证据排序和最终答案准确性 | 中 |
| 检索质量判断 / 拒答 | 必做 | 控制幻觉和不可回答问题 | 中 |
| 轻量 RAPTOR 摘要索引 | 建议 | 改善综合归纳题 | 中 |
| Query Expansion / HyDE-style expansion | 可选加分 | 改善抽象需求题和推荐题 | 低-中 |
| 轻量实体索引 | 可选加分 | 改善实体关系、多案例问题 | 中 |
| 完整 GraphRAG | 不建议第一版做 | 适合更大语料的全局问题 | 高 |
| 完整 Late Chunking | 不建议第一版做 | 依赖长上下文 embedding 实现 | 中-高 |

因此，最稳的工程路线是：

1. 把 MinerU 解析结果作为高质量结构化输入。
2. 使用结构化 chunk，而不是简单固定长度切分。
3. 给每个 chunk 增加 contextual prefix，解决上下文缺失。
4. 建立向量索引和 BM25 索引，并使用 RRF 融合。
5. 使用 reranker 从候选证据中选出最可靠片段。
6. 对综合问题增加页面/文章/产品级摘要索引。
7. 对抽象推荐问题使用查询扩展。
8. 使用检索质量判断和答案支持性检查实现拒答。

这样既吸收了前沿 RAG 的有效思想，又不会让项目复杂度失控。
