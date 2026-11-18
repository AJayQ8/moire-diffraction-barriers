"""Score the frozen periodic and isolated-wall models on a new graphene acquisition."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.spatial import QhullError

from bilayer_relaxation import make_grid
from extract_graphene_zip_intensities import PROTOCOL
from run_graphene_wall_signal import assess_wall, gap, reconstruct_geometry, summarize
from run_heterobilayer_transfer import digest
from run_mos2_nonlinear_controls import minimize_all
from run_mos2_wall_width_transfer import DISTANCE
from shared_compute import atomic_json, checkpoint_guard


def repeatability(values, available):
    signals = values.reshape(-1, 12)
    output = {}
    for start, name in ((0, "first"), (6, "second")):
        pairs = [(i, i+3) for i in range(start, start+3) if available[i] and available[i+3]]
        output[name] = [float(np.corrcoef(signals[:, i], signals[:, j])[0, 1])
                        for i, j in pairs if min(np.std(signals[:, (i, j)], axis=0)) > 0]
    return output


def native_bounds(bounds_fraction, native_shape, stride):
    bounds = np.asarray(bounds_fraction) * np.asarray(native_shape)[:2, None]
    bounds = np.rint(bounds).astype(int)
    bounds[0] = stride * ((bounds[0] + stride - 1) // stride)
    return bounds.tolist()


def repeated_chord(first, second):
    separation = np.linalg.norm(first["midpoint_nm"] - second["midpoint_nm"])
    alignment = abs(first["unit_normal"] @ second["unit_normal"])
    return separation <= 1.5 and alignment >= np.cos(np.deg2rad(20))


def geometry_windows(values, available, protocol, scan, native_shape):
    inputs, resolution = protocol["inputs"], protocol["qualification"]
    pixel, stride = inputs["raster_pixel_nm"], scan["row_stride"]
    records, all_chords = {}, []
    for rule in scan["geometry_windows"]:
        (r0, r1), (c0, c1) = native_bounds(rule["bounds_fraction"], native_shape, stride)
        crop = values[:, r0//stride:r1//stride, c0:c1]
        evidence = repeatability(crop[0], available)
        record = {"native_bounds_rc": [[r0, r1], [c0, c1]], "opposite_intensity_correlation": evidence}
        records[rule["id"]] = record
        minimum = resolution["opposite_intensity_median_correlation_min"]
        weak = [ring for ring, pairs in evidence.items() if len(pairs) < 2 or np.median(pairs) < minimum]
        if weak:
            record["qualification_failure"] = f"Weak repeatability in the {' and '.join(weak)} ring"
            continue
        try:
            context, sections, total = reconstruct_geometry(crop[0], pixel, inputs["lattice_nm"],
                                                             scan["published_twist_degrees"], available)
        except (ValueError, QhullError) as error:
            record["qualification_failure"] = str(error)
            continue
        record["first_ring_geometry"] = context["receipt"]
        record["geometry_candidates"] = total
        if not sections:
            record["qualification_failure"] = "No interior AA-to-AA chord"
        for section in sections:
            coords = section["coordinates"]
            midpoint = coords[1, len(DISTANCE)//2] + np.array((r0, c0))
            edge_margin = np.minimum(coords[:, section["keep"]], np.array(context["support"].shape) -
                                     1 - coords[:, section["keep"]]).min() * pixel
            normal = coords[1, -1] - coords[1, 0]
            all_chords.append({"window": rule["id"], "midpoint_nm": midpoint*pixel, "margin_nm": edge_margin,
                               "unit_normal": normal/np.linalg.norm(normal), "section": section,
                               "context": context, "intensities": crop})

    selected = []
    for chord in sorted(all_chords, key=lambda v: (-v["margin_nm"], v["window"], tuple(v["midpoint_nm"]))):
        if not any(repeated_chord(chord, prior) for prior in selected):
            selected.append(chord)
    for name, record in records.items():
        record["selected_unique_chords"] = sum(chord["window"] == name for chord in selected)
    return records, selected


def evaluate(rows, qualification):
    report = summarize(rows)
    pairs = report.get("common_model_pairs", 0)
    verdict = {"minimum_common_pairs": qualification["minimum_width_pairs_per_group"],
               "observed_common_pairs": pairs}
    if pairs < verdict["minimum_common_pairs"]:
        verdict["status"] = "too few qualified walls"
        return report, verdict
    observed = report["observed_gap_on_common_pairs_nm_median"]
    predicted = report["models"]["periodic"]["gap_nm_median"]
    isolated = report["models"]["isolated"]
    cross_ring = report["models"]["periodic"]["mean_cross_ring_errors"]
    reference = isolated["mean_cross_ring_errors"]
    discrepancy = report["opposite_gap_med_abs_difference_nm"]
    allowed = max(.5, discrepancy or 0.)
    error = abs(observed - predicted)
    validity = {
        "positive_gap": observed > 0.,
        "periodic_closer_than_isolated": error < abs(observed - isolated["gap_nm_median"]),
        "wins_on_full_profiles_both_directions": all(x < y for x, y in zip(cross_ring, reference, strict=True)),
        "opposite_spot_repeatability_available": report["opposite_triplet_pairs"] >= 3,
        "inside_half_nm": error <= .5,
        "inside_repeatability_limit": error <= allowed,
    }
    verdict.update({"checks": validity, "median_gap_error_nm": error, "tolerance_nm": allowed,
                    "status": "passes at measured repeatability" if all(value for name, value in validity.items()
                                                                        if name != "inside_half_nm")
                    else "criteria unmet"})
    return report, verdict


def results_for_scan(protocol, scan, directory):
    receipt = json.loads((directory / "receipt.json").read_text())
    expected_url = f'https://zenodo.org/records/{scan["record"]}/files/{scan["file"]}?download=1'
    if receipt["archive_md5"] != scan["archive_md5"] or receipt["hdf5_member"] != scan["member"] or (
        receipt["input_url"] != expected_url or receipt["protocol_sha256"] != digest(PROTOCOL)
        or receipt["row_stride"] != scan["row_stride"] or receipt["integrated_sha256"] != digest(directory/"intensity.npz")
    ):
        raise ValueError("The raw diffraction lineage does not match the frozen test")
    with np.load(directory / "intensity.npz") as data:
        intensities = data["intensity"]
    available = np.array(receipt["usable"], dtype=bool)
    records, selected = geometry_windows(intensities, available, protocol, scan, receipt["HDF5_shape"])
    inputs = protocol["inputs"]
    adhesion = np.r_[inputs["adhesion_even_coefficients_meV_per_cell"], 0., 0.]
    moduli = np.asarray(inputs["single_layer_bulk_shear_eV_per_cell"]) / 2
    solved, aperture_walls = {}, [[] for _ in intensities]
    for entry in selected:
        checkpoint_guard()
        label, section, crop, context = (entry[key] for key in ("window", "section", "intensities", "context"))
        if label not in solved:
            grid = make_grid(context["gradient"], context["basis"], 96, *moduli)
            motion, records[label]["equilibrium"] = minimize_all(grid, adhesion)
            solved[label] = grid, motion
        grid, motion = solved[label]
        for i, aperture in enumerate(crop):
            row = assess_wall(section, aperture, available, context, grid, motion, adhesion, moduli,
                              scan["probe_fwhm_nm"], scan["row_stride"])
            aperture_walls[i].append({**row, "window": label, "midpoint_nm": entry["midpoint_nm"].tolist()})
    qualification = protocol["qualification"]
    outputs = []
    for walls in aperture_walls:
        close, middle, far = [], [], []
        for row in walls:
            if row["spacing_nm"] <= qualification["high_twist_spacing_nm_max"]:
                close.append(row)
            elif row["spacing_nm"] >= qualification["low_twist_spacing_nm_min"]:
                far.append(row)
            else:
                middle.append(row)
        ordered = sorted(walls, key=lambda row: (row["spacing_nm"], row["midpoint_nm"]))
        thirds = len(ordered)//3
        groups = {"closer_walls": close, "middle": middle, "farther_walls": far, "all": walls,
                  "spacing_short_third": ordered[:thirds],
                  "spacing_long_third": ordered[-thirds:] if thirds else []}
        outputs.append({name: dict(zip(("summary", "numeric_check"), evaluate(rows, qualification), strict=True))
                        for name, rows in groups.items()})
    return {"source": receipt, "windows": records, "unique_wall_chords": len(selected),
            "apertures": outputs, "walls": aperture_walls}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("intensity_directory", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    started = perf_counter()
    protocol = json.loads(PROTOCOL.read_text())
    scan = protocol["scans"]["s1-6"]
    output = results_for_scan(protocol, scan, args.intensity_directory)
    output["inputs_sha256"] = {path.name: digest(path) for path in (Path(__file__), PROTOCOL,
                              *(PROTOCOL.with_name(name) for name in ("extract_graphene_zip_intensities.py",
                               "run_graphene_wall_signal.py", "run_mos2_wall_width_transfer.py",
                               "bilayer_relaxation.py", "graphene_wall_signal_protocol_v01.json")))}
    output["elapsed_s"] = perf_counter() - started
    atomic_json(args.output, output)
    print("Scored", output["unique_wall_chords"], "unique candidate walls.", flush=True)


if __name__ == "__main__":
    main()
