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

## Screenshots

`shots/` holds the figures the guide embeds: one per workflow step, captured by
driving the running app with Playwright, plus `penobscot.png` from
`scripts/validate_penobscot.py`. Streamlit's main column is as tall as the
viewport whatever it contains, so a short page captures with a field of white
under it; trim those rows before embedding, or the PDF scales the blank space
along with the content and the figure lands a third of its useful size.

A missing image is a skipped figure and a note, not a build failure, so the
guide still builds on a machine that has never run the app.

To refresh them, start the app and re-run the capture against it. Two things
that cost time when this was first written: Streamlit's `networkidle` fires
while the page is still being drawn over the websocket, so wait for a button to
have a real label rather than for a fixed timeout; and Streamlit dataframes
render to canvas, so their `inner_text` is empty and they can only be told apart
by counting how many appear after an action.

## The mark

The logo — impedance layers of unequal thickness with a single trace running
through them — is drawn as vector by `draw_mark()` in `build_guide.py`, on the
cover at 26 mm and in every running header at 4.4 mm. It is code rather than an
embedded image because the header instance is small enough that a raster would
have to be absurdly oversampled to stay clean, and because the cover and the
header are then guaranteed to be the same shape.

On the navy cover the four bars separate by lightness rather than hue: two
blues at similar opacity collapsed into one grey mass at 26 mm.

```bash
python build_guide.py --svg      # writes brand/logo.svg and brand/logo-dark.svg
```

Those SVGs are generated from the same geometry, so a slide or a poster can use
the logo without anyone redrawing it by eye. Edit the shape in `build_guide.py`
and re-export; do not hand-edit the SVGs.

## Terminology

`terminology_ru.tsv` records the Russian term agreed for each English one,
reviewed term by term with the project owner. Rows marked PROPOSED are the
exception: the multi-attribute vocabulary was written with this release and has
not been through that review yet.

Several of the settled terms are deliberate departures from the literal
translation — *post-stack* is «суммарные данные» not
«постстековые», *well tie* is «привязка» not «увязка», and English's
tops/markers and misfit/residual pairs each collapse to one Russian word. The
low-frequency model is never abbreviated. Check that file before changing any
Russian wording, so a settled decision is not quietly reversed.

DejaVu Sans is registered explicitly: ReportLab's built-in fonts carry no
Cyrillic, and with Helvetica every Russian character renders as a black box.
The font is expected at `/usr/share/fonts/truetype/dejavu`; on Windows or macOS
point `FONT_DIR` at any DejaVu (or other Cyrillic-capable) installation.

Numbers quoted in the guide come from `scripts/validate_penobscot.py`. If that
script is re-run and the results change, update `RESULT_ROWS` in
`build_guide.py` and the Penobscot table in the main README together.
