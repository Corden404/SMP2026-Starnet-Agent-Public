$projectRoot = Split-Path -Parent $PSScriptRoot
$submissionRoot = Join-Path $projectRoot "workspace\team_submission"
$distRoot = Join-Path $projectRoot "dist"
$timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
$archivePath = Join-Path $distRoot "team_submission-$timestamp.zip"

$requiredItems = @(
    (Join-Path $submissionRoot "config.json"),
    (Join-Path $submissionRoot "prompt"),
    (Join-Path $submissionRoot "starnet_model.py")
)

foreach ($item in $requiredItems) {
    if (-not (Test-Path -LiteralPath $item)) {
        throw "Submission item not found: $item"
    }
}

New-Item -ItemType Directory -Force -Path $distRoot | Out-Null
Compress-Archive -LiteralPath $requiredItems -DestinationPath $archivePath
Write-Host "Submission package created: $archivePath" -ForegroundColor Green
