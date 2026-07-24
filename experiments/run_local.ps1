#Requires -Version 5.1

<#
.SYNOPSIS
    Runs the resumable local GraphFeedback experiment.
.PARAMETER Mode
    Check, Validate, Smoke, Demo, FinalSmoke, Final, Confirmatory, or Aggregate.
.PARAMETER RunId
    Output run identifier. Reusing it resumes completed JSONL records.
.EXAMPLE
    powershell -NoProfile -ExecutionPolicy Bypass -File .\experiments\run_local.ps1 -Mode Smoke -RunId smoke3
.EXAMPLE
    powershell -NoProfile -ExecutionPolicy Bypass -File .\experiments\run_local.ps1 -Mode Demo -RunId demo30
.EXAMPLE
    powershell -NoProfile -ExecutionPolicy Bypass -File .\experiments\run_local.ps1 -Mode Final -RunId graphfeedback30
#>

[CmdletBinding()]
param(
    [ValidateSet('Check', 'Download', 'Validate', 'Smoke', 'Demo', 'FinalSmoke', 'Final', 'Confirmatory', 'Aggregate')]
    [string]$Mode = 'Check',

    [ValidatePattern('^[A-Za-z0-9._-]+$')]
    [string]$RunId = 'demo30',

    [ValidateRange(0, 10000)]
    [int]$SampleSize = 0,

    [string]$ConfigPath = ''
)

$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = if ($env:GRAPHFEEDBACK_PYTHON) {
    $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($env:GRAPHFEEDBACK_PYTHON)
}
else {
    $pythonCommand = Get-Command python -ErrorAction SilentlyContinue
    if (-not $pythonCommand) {
        throw 'Python was not found on PATH. Activate the experiment environment or set GRAPHFEEDBACK_PYTHON.'
    }
    $pythonCommand.Source
}
$Pipeline = Join-Path $PSScriptRoot 'scripts\pipeline.py'
$Config = if ($ConfigPath) {
    $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($ConfigPath)
}
else {
    Join-Path $PSScriptRoot 'config.local.yaml'
}

if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Python environment not found: $Python"
}
if (-not (Test-Path -LiteralPath $Config -PathType Leaf)) {
    throw "Experiment config not found: $Config"
}

function Invoke-PipelineStep {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)]
        [string]$Command,

        [int]$SampleSize = 0
    )

    $arguments = @($Pipeline, $Command, '--config', $Config, '--run-id', $RunId)
    if ($SampleSize -gt 0) {
        $arguments += @('--sample-size', $SampleSize)
    }
    Write-Host "[$(Get-Date -Format s)] $Command" -ForegroundColor Cyan
    & $Python @arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Pipeline step '$Command' failed with exit code $LASTEXITCODE"
    }
}

Push-Location -LiteralPath $ProjectRoot
try {
    switch ($Mode) {
        'Check' {
            Invoke-PipelineStep -Command 'check'
            Invoke-PipelineStep -Command 'prepare-checkpoint'
        }
        'Download' {
            Invoke-PipelineStep -Command 'download-models'
        }
        'Validate' {
            Invoke-PipelineStep -Command 'check'
            Invoke-PipelineStep -Command 'prepare-checkpoint'
            Invoke-PipelineStep -Command 'validate'
        }
        'Smoke' {
            Invoke-PipelineStep -Command 'validate'
            Invoke-PipelineStep -Command 'select' -SampleSize $(if ($SampleSize -gt 0) { $SampleSize } else { 12 })
            Invoke-PipelineStep -Command 'generate'
            Invoke-PipelineStep -Command 'filter'
            Invoke-PipelineStep -Command 'evaluate'
            Invoke-PipelineStep -Command 'aggregate'
            $SummaryPath = Join-Path $ProjectRoot "experiments\outputs\$RunId\citeseer\summary.csv"
            $GraphResult = Import-Csv -LiteralPath $SummaryPath | Where-Object { $_.method -eq 'graph_prompt_attack' }
            if (-not $GraphResult -or [int]$GraphResult.successful_nodes -lt 1) {
                throw "Stress smoke gate failed: graph_prompt_attack produced no classification flip"
            }
        }
        'Demo' {
            Invoke-PipelineStep -Command 'validate'
            Invoke-PipelineStep -Command 'select' -SampleSize $(if ($SampleSize -gt 0) { $SampleSize } else { 30 })
            Invoke-PipelineStep -Command 'generate'
            Invoke-PipelineStep -Command 'filter'
            Invoke-PipelineStep -Command 'evaluate'
            Invoke-PipelineStep -Command 'aggregate'
        }
        'FinalSmoke' {
            Invoke-PipelineStep -Command 'validate'
            Invoke-PipelineStep -Command 'select' -SampleSize $(if ($SampleSize -gt 0) { $SampleSize } else { 3 })
            Invoke-PipelineStep -Command 'generate'
            Invoke-PipelineStep -Command 'filter'
            Invoke-PipelineStep -Command 'evaluate'
            Invoke-PipelineStep -Command 'feedback-generate'
            Invoke-PipelineStep -Command 'feedback-filter'
            Invoke-PipelineStep -Command 'feedback-evaluate'
            Invoke-PipelineStep -Command 'feedback-finalize'
            Invoke-PipelineStep -Command 'aggregate'
        }
        'Final' {
            Invoke-PipelineStep -Command 'validate'
            Invoke-PipelineStep -Command 'select' -SampleSize $(if ($SampleSize -gt 0) { $SampleSize } else { 30 })
            Invoke-PipelineStep -Command 'generate'
            Invoke-PipelineStep -Command 'filter'
            Invoke-PipelineStep -Command 'evaluate'
            Invoke-PipelineStep -Command 'feedback-generate'
            Invoke-PipelineStep -Command 'feedback-filter'
            Invoke-PipelineStep -Command 'feedback-evaluate'
            Invoke-PipelineStep -Command 'feedback-finalize'
            Invoke-PipelineStep -Command 'aggregate'
        }
        'Confirmatory' {
            Invoke-PipelineStep -Command 'validate'
            Invoke-PipelineStep -Command 'select' -SampleSize $(if ($SampleSize -gt 0) { $SampleSize } else { 60 })
            Invoke-PipelineStep -Command 'generate'
            Invoke-PipelineStep -Command 'filter'
            Invoke-PipelineStep -Command 'evaluate'
            Invoke-PipelineStep -Command 'feedback-generate'
            Invoke-PipelineStep -Command 'feedback-filter'
            Invoke-PipelineStep -Command 'feedback-evaluate'
            Invoke-PipelineStep -Command 'feedback-finalize'
            Invoke-PipelineStep -Command 'aggregate'
        }
        'Aggregate' {
            Invoke-PipelineStep -Command 'aggregate'
        }
    }
}
finally {
    Pop-Location
}
