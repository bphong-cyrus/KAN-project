# Chạy trên Windows PowerShell từ bất kỳ thư mục nào:
# powershell -ExecutionPolicy Bypass -File report/build.ps1
$ErrorActionPreference = "Stop"
$ReportDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Push-Location $ReportDir
try {
    xelatex -interaction=nonstopmode -halt-on-error -file-line-error main.tex
    if ($LASTEXITCODE -ne 0) { throw "xelatex pass 1 failed" }

    bibtex main
    if ($LASTEXITCODE -ne 0) { throw "bibtex failed" }

    xelatex -interaction=nonstopmode -halt-on-error -file-line-error main.tex
    if ($LASTEXITCODE -ne 0) { throw "xelatex pass 2 failed" }

    xelatex -interaction=nonstopmode -halt-on-error -file-line-error main.tex
    if ($LASTEXITCODE -ne 0) { throw "xelatex pass 3 failed" }

    Copy-Item -Path (Join-Path $ReportDir "main.pdf") -Destination (Join-Path $ReportDir "Report.pdf") -Force
    Write-Host "Built Report.pdf successfully at $(Join-Path $ReportDir 'Report.pdf')"
}
finally {
    Pop-Location
}
