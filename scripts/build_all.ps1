param(
    [switch]$WithLlmSummaries,
    [int]$SummaryWorkers = 5,
    [int]$TopK = 5
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

if (-not $env:LOCAL_EMBEDDING_MODEL) {
    $env:LOCAL_EMBEDDING_MODEL = "BAAI/bge-base-zh-v1.5"
}
if (-not $env:EMBEDDING_DEVICE) {
    $env:EMBEDDING_DEVICE = "cpu"
}

Write-Host "== Build chunks from MinerU outputs =="
python .\scripts\build_chunks.py

if ($WithLlmSummaries) {
    if (-not $env:LLM_API_KEY) {
        throw "LLM_API_KEY is required when -WithLlmSummaries is used."
    }
    Write-Host "== Build LLM summary chunks =="
    python .\scripts\build_llm_summary_index.py --workers $SummaryWorkers --retries 2 --checkpoint-every 20
}

Write-Host "== Build retrieval index =="
python .\scripts\build_index.py

Write-Host "== Check runtime =="
python .\scripts\check_runtime.py

Write-Host "== Evaluate retrieval =="
python .\scripts\evaluate_retrieval.py --top-k $TopK
