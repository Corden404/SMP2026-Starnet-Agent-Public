$projectRoot = Split-Path -Parent $PSScriptRoot
$keyFile = Join-Path $projectRoot ".local\GLM-APIKEY.txt"

if (-not (Test-Path -LiteralPath $keyFile)) {
    throw "Private API key file not found: $keyFile"
}

$keyValue = (Get-Content -Raw -Encoding UTF8 -LiteralPath $keyFile).Trim()
if ([string]::IsNullOrWhiteSpace($keyValue)) {
    throw "Private API key file is empty."
}

$env:ZHIPUAI_API_KEY = $keyValue
Write-Host "ZHIPUAI_API_KEY loaded for this PowerShell process." -ForegroundColor Green
