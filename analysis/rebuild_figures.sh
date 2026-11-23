#!/usr/bin/env sh
set -eu

PYTHON_BIN=${PYTHON_BIN:-python3}
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$ROOT"
figures=manuscript/current_figures
"$PYTHON_BIN" "$figures/plot_wall_observations_v027.py" --edition v028
"$PYTHON_BIN" "$figures/plot_mos2_saddle_sweep.py" \
    evidence/mos2_energy_family_summary_v01.json "$figures/saddle_sweep_v027.pdf"
"$PYTHON_BIN" "$figures/plot_ring_discrepancy.py"
"$PYTHON_BIN" "$figures/plot_barrier_recovery.py"
"$PYTHON_BIN" "$figures/plot_moire_geometry_probe.py" \
    evidence/mos2_geometry_probe_summary_v005.json evidence/mos2_geometry_probe_v005.json \
    --saddle evidence/mos2_saddle_control_summary_v01.json --output "$figures" --edition v028
