$projectRoot = Split-Path -Parent $PSScriptRoot
$environmentPath = Join-Path $projectRoot ".venv"

if (-not (Test-Path -LiteralPath (Join-Path $environmentPath "conda-meta"))) {
    throw "Project Conda environment not found: $environmentPath"
}

$env:CONDA_PREFIX = $environmentPath
$env:CONDA_DEFAULT_ENV = $environmentPath
$env:PATH = "$environmentPath;$environmentPath\Scripts;$environmentPath\Library\bin;$env:PATH"
Write-Host "SMP 2026 virtual environment activated." -ForegroundColor Green
