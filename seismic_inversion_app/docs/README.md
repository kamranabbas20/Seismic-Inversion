# User guide (EN / RU)

`build_guide.py` renders a bilingual A4 user guide covering all eleven workflow
steps, the four inversion methods, the common mistakes and a glossary. English
and Russian sit side by side, column for column, so a mixed team can work from
one page and point at the same paragraph.

```bash
pip install reportlab
python build_guide.py Seismic_Inversion_Guide_EN_RU.pdf
```

The wording lives in `guide_content.py` and the layout in `build_guide.py`, so
the text can be revised without touching the PDF machinery.

DejaVu Sans is registered explicitly: ReportLab's built-in fonts carry no
Cyrillic, and with Helvetica every Russian character renders as a black box.
The font is expected at `/usr/share/fonts/truetype/dejavu`; on Windows or macOS
point `FONT_DIR` at any DejaVu (or other Cyrillic-capable) installation.

Numbers quoted in the guide come from `scripts/validate_penobscot.py`. If that
script is re-run and the results change, update `RESULT_ROWS` in
`build_guide.py` and the Penobscot table in the main README together.
