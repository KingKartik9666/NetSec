param([switch]$CheckOnly)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$studyDirectory = Join-Path $projectRoot 'dataset\experiments\xgboost_followup_20261013_v1'
$fourModelEvidence = Join-Path $projectRoot 'dataset\experiments\xgboost_data_audit_20261012_v1\deep_clean_v2\verification\completion_calibration_20261014.json'
$watchState = Join-Path $studyDirectory 'completion_watch.json'

function Get-CompletionStatus {
    $trainingState = Get-Content -Raw -LiteralPath (Join-Path $studyDirectory 'state.json') | ConvertFrom-Json
    $fourModels = Get-Content -Raw -LiteralPath $fourModelEvidence | ConvertFrom-Json
    $requiredFiles = @('summary.json', 'RESULTS.md', 'metric_verification.json', 'calibration\assessment.json', 'frozen_candidate\bundle.json')
    $reportsReady = $true
    foreach ($relativeFile in $requiredFiles) {
        if (-not (Test-Path -LiteralPath (Join-Path $studyDirectory $relativeFile))) { $reportsReady = $false }
    }
    $complete = $trainingState.status -eq 'completed' -and $reportsReady -and $fourModels.models_completed -eq 4 -and $fourModels.calibrators_completed -eq 4
    $controllerAlive = $null -ne (Get-Process -Id $trainingState.pid -ErrorAction SilentlyContinue)
    [pscustomobject]@{ Complete = $complete; ControllerAlive = $controllerAlive; Training = $trainingState }
}

function Show-TrainingNotification([string]$Title, [string]$Message) {
    Add-Type -AssemblyName System.Windows.Forms
    Add-Type -AssemblyName System.Drawing
    $notification = New-Object System.Windows.Forms.NotifyIcon
    try {
        $notification.Icon = [System.Drawing.SystemIcons]::Information
        $notification.Text = 'IMDAI training and calibration'
        $notification.Visible = $true
        $notification.BalloonTipTitle = $Title
        $notification.BalloonTipText = $Message
        $notification.ShowBalloonTip(15000)
        Start-Sleep -Seconds 20
    } finally {
        $notification.Dispose()
    }
}

if ($CheckOnly) {
    Get-CompletionStatus | ConvertTo-Json -Depth 5
    exit 0
}

$watchLock = $null
try {
    $watchLock = [System.IO.File]::Open((Join-Path $studyDirectory '.completion-watch.lock'), 'OpenOrCreate', 'ReadWrite', 'None')
    $missingControllerChecks = 0
    while ($true) {
        $completionStatus = Get-CompletionStatus
        $message = $null
        $title = $null
        if ($completionStatus.Complete) {
            $title = 'IMDAI training and calibration finished'
            $message = 'All four optimized-dataset models and calibration checks are complete. XGBoost follow-up calibration is finished too. Results: dataset/experiments/xgboost_followup_20261013_v1/RESULTS.md.'
        } elseif ($completionStatus.Training.status -eq 'failed') {
            $title = 'IMDAI calibration needs attention'
            $message = 'The remaining calibration stopped: ' + $completionStatus.Training.error
        } elseif (-not $completionStatus.ControllerAlive) {
            $missingControllerChecks++
            if ($missingControllerChecks -ge 3) {
                $title = 'IMDAI calibration stopped unexpectedly'
                $message = 'The training controller is no longer running. Completed fits remain saved. Ask Codex to resume the XGBoost follow-up calibration.'
            }
        } else {
            $missingControllerChecks = 0
        }
        $watchRecord = @{ watcher_pid = $PID; updated_at = [DateTimeOffset]::UtcNow.ToString('o'); status = 'watching'; training_status = $completionStatus.Training.status; completed_fits = $completionStatus.Training.completed_count; notification_channel = 'Windows desktop'; notify_only_on_completion_or_failure = $true }
        if ($message) { $watchRecord.status = 'notifying'; $watchRecord.title = $title }
        $watchRecord | ConvertTo-Json | Set-Content -LiteralPath $watchState -Encoding UTF8
        if ($message) {
            Show-TrainingNotification $title $message
            $watchRecord.status = 'notification_requested'
            $watchRecord | ConvertTo-Json | Set-Content -LiteralPath $watchState -Encoding UTF8
            break
        }
        Start-Sleep -Seconds 30
    }
} finally {
    if ($watchLock) { $watchLock.Dispose() }
}
