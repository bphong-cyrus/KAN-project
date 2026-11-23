#!/usr/bin/env bash
# Chạy từ thư mục bất kỳ: bash /duong/dan/report/build.sh
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
if command -v latexmk >/dev/null 2>&1; then
  latexmk -xelatex -interaction=nonstopmode -halt-on-error -file-line-error main.tex
else
  xelatex -interaction=nonstopmode -halt-on-error -file-line-error main.tex
  bibtex main
  xelatex -interaction=nonstopmode -halt-on-error -file-line-error main.tex
  xelatex -interaction=nonstopmode -halt-on-error -file-line-error main.tex
fi
cp main.pdf Report.pdf
printf 'Built Report.pdf\n'
