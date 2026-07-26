$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = $scriptDir
$venvPython = Join-Path $projectRoot '.venv\Scripts\python.exe'

if ([string]::IsNullOrWhiteSpace($env:F1_DATA_MODE)) {
    $env:F1_DATA_MODE = 'OFFLINE'
}

if ([string]::IsNullOrWhiteSpace($env:FASTF1_CACHE_DIR)) {
    $env:FASTF1_CACHE_DIR = Join-Path $projectRoot '.fastf1_cache'
}

if (Test-Path $venvPython) {
    & $venvPython -m streamlit run (Join-Path $projectRoot 'frontend\dashboard.py')
} else {
    streamlit run (Join-Path $projectRoot 'frontend\dashboard.py')
}