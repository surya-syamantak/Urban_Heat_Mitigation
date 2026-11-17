$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$desktop = Split-Path -Parent $projectRoot
$pythonCandidates = @(
    (Join-Path $desktop 'Urban_Heat_Mitigation_Chennai\.venv312\Scripts\python.exe'),
    (Join-Path $desktop 'Urban_Heat_Mitigation_Nagpur\.venv\Scripts\python.exe')
)
$python = $pythonCandidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1

if (-not $python) {
    throw 'No project Python environment was found. Run compare_projects.py with Python 3.12 and install this folder\requirements.txt first.'
}

Set-Location -LiteralPath $projectRoot
& $python (Join-Path $projectRoot 'compare_projects.py')
if ($LASTEXITCODE -ne 0) {
    throw "Comparison script failed with exit code $LASTEXITCODE."
}
