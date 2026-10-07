# Writes a short scouting report for a fighter with a local Ollama model.
# ufc.exe's profile is the source of truth; the model only turns those numbers into prose.
#   .\describe.ps1 "Islam Makhachev"
#   .\describe.ps1 adesanya -Model llama3.1:8b
param(
    [Parameter(Mandatory)] [string] $Fighter,
    [string] $Model = "llama3.2:3b"
)
$ErrorActionPreference = "Stop"

$exe = Join-Path $PSScriptRoot "ufc.exe"
if (-not (Test-Path $exe)) { throw "ufc.exe not found. Build it first: g++ -std=c++20 -O2 main.cpp -o ufc.exe" }

# ufc.exe prints UTF-8 (accented names) and finds data\ relative to the working directory.
$prevEncoding = [Console]::OutputEncoding
[Console]::OutputEncoding = [Text.Encoding]::UTF8
Push-Location $PSScriptRoot
try { $stats = & $exe profile $Fighter | Out-String; $code = $LASTEXITCODE }
finally { Pop-Location; [Console]::OutputEncoding = $prevEncoding }
if ($code -ne 0) { Write-Host $stats.TrimEnd(); exit $code }  # unknown or ambiguous name: show the suggestions

# Small models mix up the fighter / league / percentile columns, so spell each metric row out as a sentence.
$stats = $stats -replace '(?m)^  (\S.*?)\s{2,}(-?[\d.]+%?)\s+(-?[\d.]+%?)\s+(\d+)\s*$',
    '  $1: fighter $2, league average $3, percentile $4 (higher = more than other fighters)'

$prompt = @"
You are an MMA analyst. Write a 120-150 word scouting report on this fighter as one paragraph.
Use ONLY the facts and numbers below. Do not invent fights, titles, nicknames, or events. The data ends March 2024.
Refer to the fighter by last name rather than pronouns. Quote the fighter's own numbers, never the league average.
High percentiles in defense and low percentiles in "absorbed" are good. Explain how the fighter fights,
their main strengths, and their main vulnerability.

$stats
"@
$body = @{ model = $Model; prompt = $prompt; stream = $false; options = @{ temperature = 0.3; num_ctx = 4096 } } | ConvertTo-Json
try {
    $r = Invoke-RestMethod -Uri "http://localhost:11434/api/generate" -Method Post -TimeoutSec 600 `
        -ContentType "application/json; charset=utf-8" -Body ([Text.Encoding]::UTF8.GetBytes($body))
} catch {
    if (-not $_.Exception.Response) { throw "Can't reach Ollama on localhost:11434. Start the Ollama app and retry." }
    throw "Ollama error: $($_.ErrorDetails.Message) (model missing? run: ollama pull $Model)"
}
$r.response.Trim()
