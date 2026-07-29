# GitHub 提交与复现说明

## 推荐提交内容

建议提交源码、脚本、文档、样例问答、PDF 与 MinerU 解析产物。不要提交本地密钥、模型权重、向量索引和缓存。

已通过 `.gitignore` 忽略：

- `.env`
- `data/index/`
- `data/models/`
- `data/pycache/`
- Python 缓存和常见临时文件

## 复现流程

```powershell
python -m pip install -r requirements.txt
Copy-Item .env.example .env
python .\scripts\build_index.py
python .\scripts\check_runtime.py
python .\scripts\evaluate_retrieval.py --top-k 5
python .\scripts\ask.py "根据产品手册，欺骗式/压制式干扰检测设备适合什么场景？"
```

首次下载 `BAAI/bge-base-zh-v1.5` 时，`.env` 中 `HF_LOCAL_FILES_ONLY` 应设置为 `0`。模型下载完成后，可改为 `1`，保证离线运行。

## 一键脚本

只检查运行环境：

```powershell
.\run.ps1
```

评测并问答：

```powershell
.\run.ps1 -Evaluate -Ask -TopK 5
```

从 MinerU 解析产物重新构建 chunks 和索引：

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

摘要生成脚本默认使用单实例锁，并在 checkpoint/final 阶段合并最新 `chunks.jsonl`，避免重复启动任务时覆盖已有摘要。

提交前检查：

```powershell
.\scripts\check_submission.ps1
```

## 当前实测状态

当前本地默认 embedding：

```text
BAAI/bge-base-zh-v1.5
```

全量 LLM 摘要索引已生成：

```text
llm_summary = 416
total_chunks = 2236
```

当前检索评测：

```text
Recall@5 = 6/6 = 1.000
MRR@5    = 0.611
Recall@10 = 6/6 = 1.000
MRR@10    = 0.611
```

说明：全量摘要块提高了语义覆盖和展示能力，但在当前 6 条小样例上会让少数精确正文页排序后移，因此 MRR 低于只使用 `bge-base-zh-v1.5` 且无 LLM 摘要块时的 0.653。答辩时应强调这是召回增强，需要用更大验证集综合判断是否默认启用。
