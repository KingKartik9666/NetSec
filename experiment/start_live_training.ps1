param(
    [string]$Study = "artifacts/training/live_models_20261025_v1",
    [switch]$Resume
)
$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$PythonPath = Join-Path $ProjectRoot '.venv/Scripts/python.exe'
$StudyPath = [System.IO.Path]::GetFullPath((Join-Path $ProjectRoot $Study))
$CliPath = Join-Path $ProjectRoot 'experiment/train_live_models.py'
if (-not (Test-Path -LiteralPath (Join-Path $StudyPath 'study.json'))) {
    throw 'Initialize and verify the study before starting it.'
}
$Operation = if ($Resume) { 'resume' } else { 'supervise' }
$Arguments = '"{0}" {1} --study "{2}"' -f $CliPath, $Operation, $StudyPath
$RunId = [Guid]::NewGuid().ToString('N')
$Process = Start-Process -FilePath $PythonPath -ArgumentList $Arguments -WorkingDirectory $ProjectRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $StudyPath "launcher-$RunId.log") -RedirectStandardError (Join-Path $StudyPath "launcher-$RunId.error.log")
Write-Output "Started hidden study supervisor PID $($Process.Id)."
