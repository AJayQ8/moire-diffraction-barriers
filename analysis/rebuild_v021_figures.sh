#!/usr/bin/env sh
set -eu
PYTHON_BIN=${PYTHON_BIN:-python3}
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$ROOT"
"$PYTHON_BIN" manuscript/figures/plot_v021_overview.py
"$PYTHON_BIN" manuscript/figures/plot_mos2_matched_wall_models.py --rigid
"$PYTHON_BIN" manuscript/figures/plot_mos2_saddle_sweep.py \
  evidence/mos2_energy_family_summary_v01.json \
  manuscript/figures/saddle_sweep_v016.pdf
"$PYTHON_BIN" manuscript/figures/plot_measured_wall_examples.py \
  --input evidence/mos2_native_wall_examples_v01.json \
  --output manuscript/figures/measured_wall_examples_v015.pdf \
  --scan-order DS10 DS14
"$PYTHON_BIN" manuscript/figures/plot_ring_discrepancy.py
"$PYTHON_BIN" manuscript/figures/plot_moire_barrier_revision.py
"$PYTHON_BIN" manuscript/figures/plot_graphene_crosscheck.py
"$PYTHON_BIN" manuscript/figures/plot_barrier_recovery.py
"$PYTHON_BIN" manuscript/figures/plot_moire_geometry_probe.py \
  evidence/mos2_geometry_probe_summary_v005.json \
  evidence/mos2_geometry_probe_v005.json \
  --saddle evidence/mos2_saddle_control_summary_v01.json
