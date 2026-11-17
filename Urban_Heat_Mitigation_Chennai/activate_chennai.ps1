$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$activateScript = Join-Path $projectRoot '.venv312\Scripts\Activate.ps1'
$qgisLauncher = 'C:\Program Files\QGIS 3.44.15\bin\python-qgis-ltr.bat'

if (-not (Test-Path -LiteralPath $activateScript)) {
    throw "Python 3.12 environment not found. Create it with: & C:\Users\SURYA\AppData\Local\Programs\Python\Python312\python.exe -m venv .venv312"
}
if (-not (Test-Path -LiteralPath $qgisLauncher)) {
    throw "QGIS Python launcher not found: $qgisLauncher"
}

. $activateScript
$env:QGIS_PYTHON = $qgisLauncher
Set-Location -LiteralPath $projectRoot
Write-Host "Chennai project environment active: $(& python --version)"
Write-Host "QGIS launcher: $env:QGIS_PYTHON"
