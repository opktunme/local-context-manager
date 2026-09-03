$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
& "$ProjectRoot\.venv\Scripts\python.exe" "$ProjectRoot\run.py" --dev
