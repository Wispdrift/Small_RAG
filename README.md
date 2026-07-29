# 数据检索与 RAG 方向实践项目

## 任务背景

本项目方向为数据检索与 RAG（Retrieval-Augmented Generation）。任务材料包含两个长篇幅 PDF 文档，围绕这些 PDF 设计并实现一个面向长文档的 RAG 问答系统。

## 文件说明

```text
.
├── 产品手册.pdf
├── 杂志.pdf
├── qa_pairs.jsonl
└── README.md
```

- `产品手册.pdf`：产品手册类长 PDF，共 49 页。
- `杂志.pdf`：杂志类长 PDF，共 132 页。
- `qa_pairs.jsonl`：样例问答数据，用于说明任务形式、答案风格和证据粒度。

## 任务目标

请基于给定 PDF 构建一个 RAG 问答系统。系统应能够：

1. 对长篇幅 PDF 进行解析、切分、索引和检索。
2. 根据用户问题从 PDF 中定位相关证据。
3. 基于检索结果生成准确、可追溯的中文答案。
4. 对无法从文档中回答的问题，明确回答“文档中没有提供相关信息”，避免幻觉。
5. 在答案中尽量给出引用依据，例如文件名、页码、章节或检索片段。

## 样例数据格式

`qa_pairs.jsonl` 中每一行是一个 JSON 对象，字段含义如下：

```json
{
  "q_id": "q_01",
  "source": "杂志.pdf",
  "query_type": "answerable",
  "question": "问题文本",
  "answer": "参考答案",
  "gold_chunks": [
    {
      "page": 111,
      "section": "章节或栏目名称",
      "content": "支持答案的证据片段"
    }
  ]
}
```

字段说明：

- `q_id`：问题编号。
- `source`：主要来源 PDF。
- `query_type`：问题类型，`answerable` 表示可由文档回答，`unanswerable` 表示文档中没有答案。
- `question`：用户问题。
- `answer`：参考答案。
- `gold_chunks`：参考证据片段，仅作为样例说明，不要求系统硬编码。

## 说明

- `qa_pairs.jsonl` 中提供的样例问答仅用于说明本任务的问题形式、答案风格和证据粒度，并非最终评测集合，也不代表评测问题的完整覆盖范围。参评同学可以利用样例数据理解任务要求、调试系统流程和检查输出格式，但不应围绕具体样例问题进行针对性规则设计或人工适配。
- 本任务的评价重点不在于系统是否只对少量样例问题取得较好效果，而在于其面向长篇幅 PDF 问答场景时，是否具备合理、可复现且有效的检索与 RAG 系统设计。参评同学应重点说明系统中的关键设计，以及各项设计的动机、解决的实际问题，并可结合实验或案例分析说明这些设计对检索效果和问答质量的作用。

## 当前实现

本仓库已经实现一个可运行的长 PDF RAG 原型，核心链路包括：

- 基于 MinerU 解析产物读取 PDF 文本、标题、表格、图注和页码结构。
- 构建正文 chunk、页级摘要、章节摘要、局部标题索引、轻量结构图索引和 LLM 摘要索引。
- 使用 BM25 + 本地 embedding 的混合检索，并加入查询变体、来源提示、证据质量权重和导航型问题扩展。
- 默认 embedding 为 `BAAI/bge-base-zh-v1.5`，CPU 可运行。
- 支持可选 reranker、可选 query rewrite / HyDE、可选 LLM verifier。
- LLM 使用 OpenAI-compatible Chat Completions 接口，通过环境变量配置，不在代码中保存密钥。

当前已生成全量 LLM 摘要块：

```text
total_chunks = 2236
llm_summary = 416
```

## 快速运行

安装依赖：

```powershell
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

首次运行如果需要自动下载 embedding 模型，保持 `.env` 中：

```text
HF_LOCAL_FILES_ONLY=0
```

模型下载完成后，可改为：

```text
HF_LOCAL_FILES_ONLY=1
```

检查运行状态：

```powershell
python .\scripts\check_runtime.py
```

评测检索：

```powershell
python .\scripts\evaluate_retrieval.py --top-k 5
```

提问：

```powershell
python .\scripts\ask.py "根据产品手册，欺骗式/压制式干扰检测设备适合什么场景？"
```

一键脚本：

```powershell
.\run.ps1 -Evaluate -Ask -TopK 5
```

从 MinerU 解析产物重新构建：

```powershell
.\scripts\build_all.ps1
```

包含 LLM 摘要块的完整构建：

```powershell
$env:LLM_BASE_URL="https://aihubmix.com/v1"
$env:LLM_MODEL="deepseek-v4-pro"
$env:LLM_API_KEY="your_key"
.\scripts\build_all.ps1 -WithLlmSummaries -SummaryWorkers 5
```

LLM 摘要构建脚本带有单实例锁和按 `chunk_id` 合并写入机制，避免多进程并发或旧 checkpoint 覆盖已经生成的摘要块。

提交前检查：

```powershell
.\scripts\check_submission.ps1
```

## 当前评测

默认配置：

```text
active_embedder: sentence-transformers:BAAI/bge-base-zh-v1.5
chunk_count: 2236
status: ready
```

样例 answerable 问题检索结果：

```text
Recall@5          = 6/6 = 1.000
MRR@5             = 0.611
SourceAccuracy@1  = 6/6 = 1.000

Recall@10         = 6/6 = 1.000
MRR@10            = 0.611
SourceAccuracy@1  = 6/6 = 1.000
```

说明：只使用 `BAAI/bge-base-zh-v1.5` 且未加入 LLM 摘要块时，当前小样例集 MRR@5 为 0.653；加入全量 LLM 摘要块后 Recall 保持满分，但少数精确页排序后移，MRR@5 为 0.611。摘要索引更适合作为宽泛主题问答和导航型问题的召回增强。

## 提交说明

详见 [SUBMISSION_GUIDE.md](SUBMISSION_GUIDE.md)。`.env`、模型权重、向量索引和缓存不应提交到 GitHub。
