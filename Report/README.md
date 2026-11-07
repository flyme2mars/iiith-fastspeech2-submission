# Report

The graded file is `Technical_Report.pdf`.

The source is `Technical_Report.md` in this folder. `TECHNICAL_REPORT.md` is a symlink to that file. `report/REPORT.md` only points here so the old draft is not a second copy of the numbers.

Rebuild:

```bash
cd Report
pandoc Technical_Report.md -o Technical_Report.pdf \
  --embed-resources \
  --pdf-engine=wkhtmltopdf \
  --pdf-engine-opt=--enable-local-file-access \
  --lua-filter=telugu-font.lua \
  --metadata title="FastSpeech 2 for English (Svarah) and Telugu (IndicTTS)" \
  -c report.css \
  -f markdown-smart
```

Run it from this directory so `../Results/` resolves. `telugu-font.lua` marks Telugu letters for Noto Sans Telugu. The loss curves and variance plots are image links. Do not replace them with guessed loss numbers.
