# Reproducing the wall-width and orientation results

## Reproduce the spacing and weighting comparisons

The wall-width and barrier summaries use Python 3.11 or later with the standard library and run offline.

From the repository root, run

    python3 analysis/summarize_mos2_spacing_and_weighting.py

The calculation writes `evidence/mos2_spacing_and_weighting_v01.json`. It reports the prediction for each omitted map, the map-average errors, the effect of dropping each map from the physical-model comparison, and the barrier-error curves under each weighting rule. The script verifies the shared wall segments and reproduces the original map-averaged scores. The reference copy includes the SHA-256 hashes of the inputs and script.

## Reproduce the four additional maps and combined error

Run `python3 analysis/summarize_mos2_additional_widths.py` from the extracted directory. The script selects identical valid walls for the measured, isolated and periodic fits. It fits the spacing relation on the original six maps and checks the detector calibration membership and measured and predicted mean width differences for each of the four additional maps. It reports the four-map and combined ten-map errors. Use `--tex rows.tex` to save the values underlying Supplementary Table S34.

The full replay of the four added maps begins with the public scan ZIPs and the
author's reader. After following the reader setup in the main README, run

    python3 replay/run_additional_parallel_maps.py replay "$DATA" "$READER" "$DEPS" evidence evidence four-map-widths.json

It first checks the original DS10 wall fits, then computes the fitted widths
for DS5, DS7, DS8 and DS11. Both evidence arguments point to the same archive
directory in this combined release.

## Compare the published diffraction image with the archived scans

This check asks whether Fig. 1b of Van Winkle et al. provides an independent crystal-angle reference for DS14. Place the original DS1.zip through DS14.zip files from the [public diffraction archive](https://doi.org/10.5281/zenodo.7865970) in one directory. With Python 3.11 or later, NumPy and SciPy installed, run

    python3 analysis/check_mos2_diffraction_figure.py --data-dir /path/to/scan-archives

The script registers the 12 MoS₂ spot centers in the source figure to the archived coordinates and measures the separations of five hBN spots in source-image pixels. It starts from the saved intensity-weighted centroids at the default segmentation threshold of 135. Use --threshold 115 or --threshold 155 for the two sensitivity checks. The accompanying evidence record identifies the source image, the segmented centers and the archived scan results.

For DS14 the MoS₂ registration has a root-mean-square residual of 0.44 pixels, while the hBN spots differ by 10.02 pixels. Ten other archived scans give an hBN residual below one pixel. The image therefore cannot assign DS14 an independent orientation.
