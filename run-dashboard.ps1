$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = $scriptDir
$venvPython = Join-Path (Split-Path -Parent $projectRoot) '.venv\Scripts\python.exe'

if (Test-Path $venvPython) {
    & $venvPython -m streamlit run (Join-Path $projectRoot 'frontend\dashboard.py')
} else {
    streamlit run (Join-Path $projectRoot 'frontend\dashboard.py')
}