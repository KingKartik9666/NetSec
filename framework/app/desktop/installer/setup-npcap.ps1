[CmdletBinding()]
param([switch]$Install, [switch]$VerifyDownload)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

# Intentionally interactive and optional. Npcap's free license does not permit
# bundling its installer or silent installation. This file contains no driver.
$driverDll = Join-Path $env:WINDIR 'System32\Npcap\wpcap.dll'
if ((Test-Path -LiteralPath $driverDll) -and -not $VerifyDownload) {
    Write-Output 'Npcap is already installed. Existing installation retained.'
    exit 0
}
try {
    [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
    $downloadDirectory = Join-Path ([IO.Path]::GetTempPath()) ('IMDAI-Npcap-' + [guid]::NewGuid().ToString('N'))
    New-Item -ItemType Directory -Path $downloadDirectory | Out-Null
    $downloadPath = Join-Path $downloadDirectory 'npcap-1.89.exe'
    Invoke-WebRequest -UseBasicParsing -Uri 'https://npcap.com/dist/npcap-1.89.exe' -OutFile $downloadPath -TimeoutSec 120
    $actualHash = (Get-FileHash -LiteralPath $downloadPath -Algorithm SHA256).Hash
    if ($actualHash -ne '8AED85E900D783D1308506E919587D3E540451947AF8A82F2D04F819E44305CC') {
        throw 'Npcap download checksum differs from the verified release. No installer was run.'
    }
    $signature = Get-AuthenticodeSignature -LiteralPath $downloadPath
    if ($signature.Status -ne 'Valid' -or $signature.SignerCertificate.Subject -notmatch '(?:^|,\s*)CN=Nmap Software LLC(?:,|$)') {
        throw 'Npcap does not have a valid Nmap Software LLC Authenticode signature. No installer was run.'
    }
    Write-Output "Verified official Npcap 1.89: $downloadPath"
    if ($Install) {
        # ShellExecute lets the vendor's own driver installer display its normal
        # elevation/license/options UI. Never add /S or bypass system policy.
        $startInfo = New-Object System.Diagnostics.ProcessStartInfo
        $startInfo.FileName = $downloadPath
        $startInfo.UseShellExecute = $true
        $installerProcess = [Diagnostics.Process]::Start($startInfo)
        $installerProcess.WaitForExit()
        if ($installerProcess.ExitCode -notin @(0, 3010)) { throw "Npcap installer exited with code $($installerProcess.ExitCode)." }
        if (-not (Test-Path -LiteralPath $driverDll)) { throw 'Npcap driver files are not available yet. Complete installation or restart Windows as requested by its installer.' }
        Write-Output 'Npcap driver files are installed. Restart IMDAI to refresh packet capture.'
    }
    exit 0
} catch {
    Write-Error $_
    exit 1
}
