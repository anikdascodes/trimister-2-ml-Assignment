$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projectRoot

$env:MPLCONFIGDIR = (Join-Path $projectRoot ".mplconfig")
$env:JUPYTER_DATA_DIR = (Join-Path $projectRoot ".jupyter-data")
$env:JUPYTER_CONFIG_DIR = (Join-Path $projectRoot ".jupyter-config")
$env:JUPYTER_RUNTIME_DIR = (Join-Path $projectRoot ".jupyter-runtime")
$env:IPYTHONDIR = (Join-Path $projectRoot ".ipython")
$env:TEMP = (Join-Path $projectRoot ".tmp")
$env:TMP = $env:TEMP
$env:JUPYTER_ALLOW_INSECURE_WRITES = "true"

New-Item -ItemType Directory -Force `
    $env:MPLCONFIGDIR, `
    $env:JUPYTER_DATA_DIR, `
    $env:JUPYTER_CONFIG_DIR, `
    $env:JUPYTER_RUNTIME_DIR, `
    $env:IPYTHONDIR, `
    $env:TEMP | Out-Null

& ".\.venv\Scripts\Activate.ps1"
& ".\.venv\Scripts\jupyter-nbconvert.exe" --to notebook --execute --inplace ".\assignment_notebook.ipynb"
