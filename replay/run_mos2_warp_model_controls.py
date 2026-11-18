"""Give rigid and relaxed patterns identical freedom to correct scan positions."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from author_release import load_author_release
from bilayer_relaxation import MODES, MONOLAYERS, STACKING
from observe_bilayers import read_input_metadata
from run_heterobilayer_transfer import PROJECT, digest
from run_moire_common_intensity import BENCHMARK, centers_and_context
from run_moire_regime_benchmark import SETTINGS
from run_mos2_directional_soliton import directional_solitons
from run_mos2_intensity_only_replay import disk_images
from run_mos2_local_intensity_geometry import geometric_folds
from run_mos2_local_wall_transfer import predictions
from run_mos2_phase_warp_transfer import outputs
from run_mos2_reduced_equilibrium import FourierModel
from run_mos2_slow_readout_transfer import smooth_basis
from run_registry_prediction import rotation
from run_wall_heterostrain import PROTOCOL, SOURCE
from shared_compute import atomic_json, checkpoint_guard


WARP_REFERENCE = PROJECT/"mos2_phase_warp_transfer_v01.json"


def fields(background, gradient, basis, interface):
    physics = predictions({"background":background},gradient,basis,interface)["background"]
    model = FourierModel.from_geometry(gradient,basis,interface,MODES)
    params, _ = model.solve()
    origin = basis @ np.full(2,2/3 if interface[-1] == "AP" else 0.)
    compact = background+model.displacement(background+origin,params)
    physics["compact"] = compact
    if interface[-1]=="P":
        _, c1, c2, c3, _, _ = STACKING[interface]
        _, bulk, shear = MONOLAYERS["MoS2"]
        effective = (c1-8*c2+9*c3)/(2000*shear)
        walls, _ = directional_solitons(background,basis,gradient,effective,bulk/shear)
        physics["constant_width_soliton"] = walls["fixed_physical_width"]
    return physics


def evaluate(images, ctx, interface, metadata, reference):
    r = ctx["receipt"]
    pixel, basis = r["pixel_nm"], ctx["basis"]
    theta = np.deg2rad(float(metadata["SampleRotation"]))
    fit = np.asarray(r["first_ring_affine_fit"])
    y,x = np.indices(images.shape[:2],dtype=float)
    query = pixel*np.stack((x,y),axis=-1)
    fitted = list(geometric_folds(np.asarray(r["centers_rc_px"]),query,pixel,basis,fit,ctx["support"]))
    work = [("affine",ctx["rigid"],ctx["gradient"],ctx["support"],fit,None)]
    if fitted:
        group, _, local, affine_fit, mask, _ = max(fitted,key=lambda row:row[4].sum())
        work.append(("local",local,affine_fit[:2].T@rotation(-theta),mask,affine_fit,group))
    result = {}
    for name, background, gradient, mask, geometry, group in work:
        previous = reference[name]
        if group!=previous["unused_center_group"]:
            raise ValueError("The earlier local geometry selection changed")
        period = np.sqrt(abs(np.linalg.det(np.linalg.solve(gradient,basis))))/pixel
        controls, _, _ = smooth_basis(mask.shape,mask,period)
        cooked = fields(background,gradient,basis,interface)
        result[name] = {
            "unused_center_group":group, "control_count":controls.shape[-1],
            "models": {
                key: outputs(field,images,ctx,mask,controls,geometry,"first","second")
                for key,field in cooked.items()
            },
        }
        check = result[name]["models"]["full"]["scores"]
        for a,b in zip(check, previous["rings"]["second"]["scores"],strict=True):
            if a["pixels"]!=b["pixels"] or max(abs(a["rmse"][k]-b["rmse"][k]) for k in a["rmse"])>1e-12:
                raise ValueError("The previously measured effect of coordinate warping changed")
        checkpoint_guard()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive",type=Path)
    parser.add_argument("release",type=Path)
    parser.add_argument("dependencies",type=Path)
    parser.add_argument("output",type=Path)
    args=parser.parse_args()
    started=perf_counter()
    paths=PROTOCOL,SOURCE,BENCHMARK,SETTINGS,WARP_REFERENCE
    protocol,source,benchmark,settings,reference=[json.loads(p.read_text()) for p in paths]
    author=load_author_release(args.release,args.dependencies)
    result={
        "scope":"Allow each fixed physical field an identical smooth scan-plane warp or layer shift. Fit the correction only to the first ring, with the same number of control points and equivalent strain-scale bound as for the full periodic field. Use the same second-ring training halves for three constant detector parameters. Compare the test halves to see which physical predictions survive this additional information about the local measured geometry.",
        "limits":"Only the first-ring intensity feeds the correction, including intensity at equivalent second-ring score locations. No microscope reference independently assigns the warp to sample or scanner. A large warp can trade with the physical field. This is a sensitivity analysis on the examined acquisitions.",
        "input_sha256":{p.name:digest(p) for p in (Path(__file__),*paths)},
        "scans":{},
    }
    for group,names in protocol["cases"].items():
        for scan in names:
            checkpoint_guard()
            saved=source["scans"][scan]
            metadata=read_input_metadata(scan,args.archive)
            images,modes,checksum=disk_images(args.archive,scan,author)
            if checksum!=saved["diskset_sha256"] or metadata["archive_md5"]!=benchmark["scans"][scan]["archive_md5"]:
                raise ValueError("The diffraction inputs changed")
            interface=tuple(saved["interface"])
            context=centers_and_context(images,modes,metadata["metadata_fields"],settings,interface)
            if context["receipt"]!=saved["geometry"]:
                raise ValueError("The first-ring geometry changed")
            row=evaluate(images,context,interface,metadata["metadata_fields"],reference["scans"][scan]["geometry"])
            result["scans"][scan]={"group":group,"geometry":row}
            result["elapsed_seconds"]=perf_counter()-started
            atomic_json(args.output,result)
            print(scan,{geo:{k:round(float(np.mean([f["rmse"]["coordinate_warp"] for f in v["scores"]])),5)
                             for k,v in entry["models"].items()}
                        for geo,entry in row.items() if entry["models"]["full"]["scores"]},flush=True)


if __name__=="__main__":
    main()
