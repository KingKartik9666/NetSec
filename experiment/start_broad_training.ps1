param(
    [string]$Study = 'artifacts/training/broad_live_models_20261028_v2',
    [switch]$Resume
)
$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$PythonPath = Join-Path $ProjectRoot '.venv/Scripts/python.exe'
$StudyPath = [System.IO.Path]::GetFullPath((Join-Path $ProjectRoot $Study))
$TrainingRoot = [System.IO.Path]::GetFullPath((Join-Path $ProjectRoot 'artifacts/training'))
if (-not $StudyPath.StartsWith($TrainingRoot + [System.IO.Path]::DirectorySeparatorChar, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw 'The broad study must stay inside artifacts/training.'
}
$CliPath = Join-Path $ProjectRoot 'experiment/train_broad_live.py'
if (-not (Test-Path -LiteralPath (Join-Path $StudyPath 'study.json'))) {
    throw 'Initialize and verify the study before starting it.'
}
$Operation = if ($Resume) { 'resume' } else { 'supervise' }
$Arguments = '"{0}" {1} --study "{2}"' -f $CliPath, $Operation, $StudyPath
$RunId = [Guid]::NewGuid().ToString('N')
$Process = Start-Process -FilePath $PythonPath -ArgumentList $Arguments -WorkingDirectory $ProjectRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $StudyPath "launcher-$RunId.log") -RedirectStandardError (Join-Path $StudyPath "launcher-$RunId.error.log")
Write-Output "Started hidden broad-study supervisor launcher PID $($Process.Id). Check status and heartbeat to verify work."
