# Exported result tables

Run python3 analysis/export_release_tables.py from the package root to rebuild
these CSVs from the checksummed evidence.

- matched_width_errors.csv: map-weighted width-gap errors for close, wider,
  and combined MoS₂ map groups.
- synthetic_barrier_recovery.csv: saved synthetic recovery errors by spacing,
  controlled synthetic condition, and model.
- eight_map_saddle_sensitivity.csv: saved-score minima, reselected-minimum
  spatial-block ranges, and selected-barrier ranges for each material family,
  intensity fit, and registry frame.
- representative_mos2_wall_widths.csv: qualified measured and predicted
  first- and second-order widths for representative DS10 and DS14 sections.

These are derived numerical results, not raw scan data.
