"""Compare a straight-wall theory with the stretch response of the full network."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.linalg import polar

from author_release import load_author_release
from bilayer_relaxation import MONOLAYERS, STACKING
from observe_bilayers import load_dataset, read_input_metadata
from run_heterobilayer_transfer import PROJECT, digest
from run_moire_common_intensity import BENCHMARK, averages, calibrate_and_score, centers_and_context
from run_moire_regime_benchmark import SETTINGS
from run_mos2_intensity_only_replay import disk_images
from run_mos2_local_wall_transfer import predictions
from run_mos2_soliton_approximation import soliton_motion
from run_wall_heterostrain import PROTOCOL, SOURCE, wall_scores
from shared_compute import atomic_json, checkpoint_guard


INTERFACE = ("MoS2", "MoS2", "P")
REFERENCE = PROJECT / "wall_heterostrain_v03.json"
TRIPLET = np.array(((1, 0), (0, 1), (-1, -1)))


def directional_solitons(rigid, basis, gradient, effective, bulk_over_shear):
    """Preserve the displacement slip while changing each soliton's physical width."""
    lattice = np.linalg.norm(basis[:,0])
    q = TRIPLET @ np.linalg.inv(basis)
    turn, deformation = polar(np.eye(2)+gradient)
    isotropic = np.sqrt(np.linalg.det(deformation))*turn-np.eye(2)
    twist = abs((gradient[1,0]-gradient[0,1])/2)
    shear_reference = np.linalg.norm(q * twist, axis=-1)
    wavevector = q @ gradient
    q_norm = np.linalg.norm(q, axis=-1)
    k_norm = np.linalg.norm(wavevector, axis=-1)
    slip = q/q_norm[:,None]
    normal = wavevector/k_norm[:,None]
    line_stiffness = 1.+bulk_over_shear*np.sum(normal*slip, axis=-1)**2
    # Fourier decay is proportional to k times the physical isolated-wall
    # width. Keep each physical slip fixed across the three scan directions.
    original = 2*twist/(3*np.sqrt(effective))
    damping = {
        "fixed_physical_width": original*k_norm/shear_reference,
        "orientation_stiffness": original*k_norm*np.sqrt(line_stiffness)/shear_reference,
        "no_stretch_wall_law": original*np.linalg.norm(q@isotropic,axis=-1)/shear_reference,
        "fixed_isotropic_angles": np.full(3, original),
    }
    result = {}
    phase = 2*np.pi*(rigid @ q.T)
    for name, decay in damping.items():
        moving = np.zeros_like(rigid)
        nmax = int(np.ceil(15/min(decay)))
        for n in range(1, nmax+1):
            weight = (2/3)/(n*np.cosh(n*decay)**2)
            moving += (np.sin(n*phase)*weight) @ slip
        result[name] = rigid + np.sqrt(3)*lattice/(2*np.pi)*moving
    reference, _ = soliton_motion(rigid,basis,twist,effective)
    if np.max(np.abs(result["fixed_isotropic_angles"]-rigid-reference)) > 1e-12:
        raise ValueError("The published isotropic soliton observer changed")
    return result, {"decay": {k: v.tolist() for k,v in damping.items()},
                    "line_stiffness_over_shear": line_stiffness.tolist(),
                    "physical_frequency_relative_to_twist": (k_norm/shear_reference).tolist()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("release", type=Path)
    parser.add_argument("dependencies", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    files = (PROTOCOL, SOURCE, BENCHMARK, SETTINGS, REFERENCE)
    protocol, source, benchmark, settings, old = (json.loads(p.read_text()) for p in files)
    author = load_author_release(args.release,args.dependencies)
    _, c1, c2, c3, _, _ = STACKING[INTERFACE]
    _, bulk, shear = MONOLAYERS["MoS2"]
    effective = (c1-8*c2+9*c3)/(2000*shear)
    result = {"scope": "Extend the published triangular soliton Fourier formula by using the actual spatial frequency of each wall direction to retain a common physical width, or applying the additional isolated-wall line stiffness. This sets one decay length per reciprocal direction and fixes the displacement step. Compare with the original isotropic formula and full periodic equilibrium on the common shear and diffraction endpoints.",
              "limits": "These changes to the published soliton prediction approximate a network as three superposed wall families. They retain no explicit wall curvature and inter-wall interactions. The construction is only tested on the previously seen parallel maps and uses their first-ring geometry and nominal microscope calibration.",
              "input_sha256": {p.name:digest(p) for p in (Path(__file__), *files)}, "scans":{}}
    for group in ("parallel_development","parallel_comparison"):
        for scan in protocol["cases"][group]:
            checkpoint_guard()
            reference = source["scans"][scan]
            rec = benchmark["scans"][scan]
            meta = read_input_metadata(scan,args.archive)
            images,modes,fingerprint = disk_images(args.archive,scan,author)
            if fingerprint != reference["diskset_sha256"] or meta["archive_md5"] != rec["archive_md5"]:
                raise ValueError("The original archive changed")
            ctx = centers_and_context(images,modes,meta["metadata_fields"],settings,INTERFACE)
            if ctx["receipt"] != reference["geometry"]:
                raise ValueError("The first-ring geometry changed")
            basis, rigid, grad = (ctx[k] for k in ("basis","rigid","gradient"))
            fields, approximation = directional_solitons(rigid,basis,grad,effective,bulk/shear)
            controls = predictions({"rigid":rigid},grad,basis,INTERFACE)["rigid"]
            modeled = fields | controls
            data = load_dataset(scan,args.archive,allowed_ids=(scan,),allow_author_crop=True)
            wall = wall_scores(data,rec,ctx,modeled,settings)
            previous = old["scans"][scan]["wall_scores"]["models"]["full"]["directional_rmse_pp"]
            if abs(wall["models"]["full"]["directional_rmse_pp"]-previous) > 1e-12:
                raise ValueError("The frozen full wall score changed")
            intensity = averages(calibrate_and_score(modeled,ctx,images,ctx["support"]))
            result["scans"][scan] = {"group":group, "intensity":intensity,"wall_scores":wall,"geometry_controls":approximation}
            atomic_json(args.output,result)
            print(scan,{k:round(v["directional_rmse_pp"],3) for k,v in wall["models"].items()},flush=True)


if __name__ == "__main__":
    main()
