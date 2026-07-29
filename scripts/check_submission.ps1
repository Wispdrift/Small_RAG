$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

Write-Host "== Compile =="
$env:PYTHONPYCACHEPREFIX = ".\data\pycache"
python -m compileall .\src .\scripts

Write-Host "== Runtime =="
python .\scripts\check_runtime.py

Write-Host "== Retrieval eval =="
python .\scripts\evaluate_retrieval.py --top-k 5

Write-Host "== Path and secret scan =="
$ScanOutput = rg -n "D:\\|C:\\|sk-[A-Za-z0-9]{20,}" .\src .\scripts .\README.md .\TASK_SOLUTION.md .\PROJECT_STRUCTURE_AND_STEPS.md .\.env.example .\pyproject.toml .\requirements.txt .\run.ps1 --glob "!scripts/check_submission.ps1"
if ($LASTEXITCODE -eq 0) {
    Write-Host "Potential hard-coded path or secret found:"
    $ScanOutput
} else {
    Write-Host "No hard-coded absolute paths or real-looking API keys found in tracked source/docs."
    $global:LASTEXITCODE = 0
}
