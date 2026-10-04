param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$BenchlabArgs
)

$ErrorActionPreference = "Stop"
if ([string]::IsNullOrWhiteSpace($env:BENCHLAB_PROJECT)) {
    throw "Set BENCHLAB_PROJECT to the external local-llm-serving-lab project directory (see scripts/README.md)"
}
if (-not (Test-Path -LiteralPath (Join-Path $env:BENCHLAB_PROJECT "pyproject.toml") -PathType Leaf)) {
    throw "BENCHLAB_PROJECT must contain pyproject.toml"
}
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    throw "Install uv and run uv sync --project with your BENCHLAB_PROJECT directory first"
}
uv run --project $env:BENCHLAB_PROJECT benchlab @BenchlabArgs
exit $LASTEXITCODE
