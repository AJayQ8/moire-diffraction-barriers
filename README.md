# Interlayer relaxation and diffraction-based sliding-barrier inference in twisted bilayer MoS₂

**Release:** v022 (2026-11-23)

**Repository:** https://github.com/AJayQ8/moire-diffraction-barriers

**Version DOI:** https://doi.org/10.5281/zenodo.24456343

This release contains the continuum relaxation implementation, archived analysis
and replay code, checksummed result records, the fitted wall measurements,
and scripts to reproduce the numerical comparisons and figures for all 14 maps.

## Scientific scope and saved results

The periodic model implements the published continuum mechanics for twisted
bilayer MoS₂ and compares its diffraction-derived wall-width predictions with
an isolated-wall model.

- Across all ten maps with closely spaced walls (272 matched wall segments), the
  mean absolute error in the map-averaged first-minus-second-order width
  difference is **1.366 nm** for isolated walls and **0.183 nm** for the relaxed
  network. These ten comprise the six maps below and DS5, DS7, DS8 and DS11.
- On 168 matched wall segments in six closer-spaced maps (DS3, DS4, DS6, DS9,
  DS10, DS12), the mean map-level absolute error in the first-minus-second
  fitted-width difference is **1.342 nm** for the isolated wall and **0.156 nm**
  for the periodic network.
- On 74 segments in four wider-spaced maps (DS14, DS1, DS2, DS13), the
  corresponding errors are **0.377 nm** and **0.365 nm**.
- In the saved synthetic test at 8.5 nm nominal spacing, with variable probe
  and energy landscape, median barrier-recovery error is **49.7%** for the
  isolated model and **11.5%** for the periodic model across 36 realizations.
  This is a controlled synthetic recovery result.
- In the bounded saved-score comparison that omits DS13 and DS14, the eight
  retained maps select **14 meV per primitive cell** at the nominal 1.25 nm
  probe for both the fixed-AA-energy and fixed-AA-curvature families, both intensity
  fits, and both frame branches. The archived and rotated branches are exactly
  equal on the retained set across 15,488 fit-width arrays. The selected estimate
  is conditional on the maps, fit-qualified common sections, saved probe, and
  material assumptions. Its paired
  within-map spatial resampling range for the reselected minimum error is about
  0.14–0.22 nm across those conditions; the reselected grid barrier spans
  12.5–14 meV per cell. It does not quantify between-map, device-level, or
  broader material-parameter uncertainty.

The exact saved values and cohort definitions are in evidence/. The exported
CSV tables are derived from those records by
analysis/export_release_tables.py.

## Quick reproduction from saved results

The documented regeneration environment was tested with Python 3.12.8.
Use Python 3.12 for the pinned NumPy 1.26.4 environment. From the repository root:

    python3 -m venv .venv
    . .venv/bin/activate
    python3 -m pip install -r requirements.txt
    shasum -a 256 -c SHA256SUMS
    (cd evidence && shasum -a 256 -c SHA256SUMS)
    python3 analysis/export_release_tables.py
    python3 analysis/summarize_mos2_spacing_and_weighting.py
    python3 analysis/summarize_mos2_additional_widths.py --tex additional-map-widths.tex
    sh analysis/rebuild_figures.sh

The commands use the archived fits to reproduce the comparisons, the supplementary
table of four additional maps and all 11 main and supplementary figures. The
current figures and their source are in `manuscript/current_figures/`. The
[width-analysis instructions](analysis/width-analysis.md) also describe how to
check the original diffraction image against the 14 published scan archives.
The older figures remain available under `manuscript/figures/` and can be
rebuilt with `sh analysis/rebuild_v021_figures.sh`.

SHA256SUMS checks the complete release tree. The file
replay/ARCHIVED_INPUT_HASHES.json links replay inputs and scripts to saved
record hashes. evidence/SHA256SUMS verifies the portable evidence files;
provenance/SOURCE_ARCHIVE_SHA256SUMS retains the original v021 source-bundle
hashes. provenance/portable_path_redactions.json records the evidence metadata
replacement; provenance/release_adaptations.json lists release-only source
changes. replay/graphene/ contains processed graphene intensity excerpts and
receipts. The representative width CSV is written to
results/representative_mos2_wall_widths.csv from the evidence record.

## Replaying fits from public scan archives

The included saved records are sufficient for the quick reproduction above.
Re-running MoS₂ width extraction and forward profile comparisons requires the
processed scan ZIPs and pinned author reader, which are not bundled:

1. Download the processed per-scan data from Van Winkle et al.,
   https://doi.org/10.5281/zenodo.7865970, and place the original DS*.zip
   files in a directory.
2. Download the pinned pyInterferometry release from
   https://doi.org/10.5281/zenodo.7863860. Keep the downloaded archive named
   release.zip and extract it in the same directory; use the extracted src/
   directory. The replay checks the archive MD5 and source commit.
3. Install the reader dependencies alongside requirements.txt: gekko==1.3.0,
   pathos, scikit-image, shapely, and networkx.
4. Set the three paths, add the standalone file-writing helper to PYTHONPATH,
   and invoke an archived entry point from the package root. For example:

    DATA=/path/to/processed_scan_zips
    READER=/path/to/pyInterferometry/src
    DEPS=/path/to/reader_dependencies
    export PYTHONPATH="$PWD/replay/standalone_support"
    OUT=mos2_wall_width_transfer_replay.json
    python3 replay/run_mos2_wall_width_transfer.py \
      "$DATA" "$READER" "$DEPS" "$OUT"

The frozen entry points, parameters, and evidence files are in replay/ and
analysis/. Replays that regenerate physical model solutions are separate from
the saved-result workflow above.

## Licensing and attribution

Original software code is MIT licensed. Original result tables, figures, and
documentation are CC BY 4.0. Some bundled measurement-derived files adapt
CC BY 4.0 upstream datasets. The full source citations and license boundaries
are in LICENSES.md and THIRD_PARTY_NOTICES.md. Upstream software and data retain
their own terms; this release makes no license claim over them.

Cite the repository using CITATION.cff.
