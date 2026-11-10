param([string]$Python = 'python')

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Push-Location -LiteralPath $projectRoot
try {
    & $Python -m PyInstaller --noconfirm Observatory.spec
    if ($LASTEXITCODE -ne 0) { throw 'Windows executable build failed.' }
    $executable = Join-Path $projectRoot 'dist\Observatory.exe'
    if (-not (Test-Path -LiteralPath $executable)) { throw 'Build did not produce Observatory.exe.' }
    Get-Item -LiteralPath $executable | Select-Object FullName, Length, LastWriteTime
} finally {
    Pop-Location
}
