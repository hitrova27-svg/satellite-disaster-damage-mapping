param(
    [Parameter(Mandatory = $true)]
    [int]$DownloadPid
)

$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot

Write-Output "Waiting for xBD downloader PID $DownloadPid"
Wait-Process -Id $DownloadPid

$archive = Join-Path $PSScriptRoot 'xbd-dataset.zip'
if (-not (Test-Path -LiteralPath $archive -PathType Leaf)) {
    throw "Downloader exited without creating $archive"
}

Write-Output 'Checking ZIP central directory'
tar.exe -tf $archive | Out-Null

Write-Output 'Extracting xBD archive'
tar.exe -xf $archive

Write-Output 'Validating dataset layout and split assignment'
& .\.venv\Scripts\python.exe .\make_splits.py --src .\xbd --out . --dry-run
if ($LASTEXITCODE -ne 0) {
    throw "make_splits.py validation failed with exit code $LASTEXITCODE"
}

Write-Output 'Executing local CUDA smoke notebook'
& .\.venv\Scripts\jupyter.exe nbconvert `
    --to notebook `
    --execute .\artur_xbd_LOCAL_SMOKE.ipynb `
    --output artur_xbd_LOCAL_SMOKE.executed.ipynb `
    --ExecutePreprocessor.kernel_name=xbd-local `
    --ExecutePreprocessor.timeout=-1
if ($LASTEXITCODE -ne 0) {
    throw "Smoke notebook failed with exit code $LASTEXITCODE"
}

Write-Output 'LOCAL SETUP COMPLETE: split validation and smoke notebook both succeeded.'
