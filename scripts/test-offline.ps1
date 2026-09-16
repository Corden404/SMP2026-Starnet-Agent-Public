$projectRoot = Split-Path -Parent $PSScriptRoot
$pythonPath = Join-Path $projectRoot ".venv\python.exe"

if (-not (Test-Path -LiteralPath $pythonPath)) {
    throw "Project Python not found: $pythonPath"
}

$env:ANONYMIZED_TELEMETRY = "False"
& $pythonPath -m unittest discover -s (Join-Path $projectRoot "tests") -p "test_*.py" -v
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
