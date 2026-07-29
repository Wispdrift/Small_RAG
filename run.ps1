param(
    [string]$Question = "根据产品手册，欺骗式/压制式干扰检测设备适合什么场景？",
    [int]$TopK = 5,
    [switch]$RebuildChunks,
    [switch]$RebuildIndex,
    [switch]$Evaluate,
    [switch]$Ask
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

if (-not $env:LOCAL_EMBEDDING_MODEL) {
    $env:LOCAL_EMBEDDING_MODEL = "BAAI/bge-base-zh-v1.5"
}
if (-not $env:EMBEDDING_DEVICE) {
    $env:EMBEDDING_DEVICE = "cpu"
}
if (-not $env:HF_LOCAL_FILES_ONLY) {
    $env:HF_LOCAL_FILES_ONLY = "1"
}

Write-Host "== Runtime =="
python .\scripts\check_runtime.py

if ($RebuildChunks) {
    Write-Host "== Build chunks =="
    python .\scripts\build_chunks.py
}

if ($RebuildIndex) {
    Write-Host "== Build index =="
    python .\scripts\build_index.py
}

if ($Evaluate) {
    Write-Host "== Evaluate retrieval =="
    python .\scripts\evaluate_retrieval.py --top-k $TopK
}

if ($Ask) {
    Write-Host "== Ask =="
    python .\scripts\ask.py $Question --top-k $TopK
}

if (-not ($RebuildChunks -or $RebuildIndex -or $Evaluate -or $Ask)) {
    Write-Host "No action requested. Example:"
    Write-Host ".\run.ps1 -Evaluate -Ask -TopK 5"
}
