"""Test wall diffraction with position corrections fitted away from walls."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter
from scipy.optimize import lsq_linear

from author_release import load_author_release
from observe_bilayers import read_input_metadata
from run_heterobilayer_transfer import PROJECT, digest
from run_moire_common_intensity import BENCHMARK, centers_and_context
from run_moire_regime_benchmark import SETTINGS
from run_mos2_directional_diffraction import wall_sectors
from run_mos2_frame_sensitivity import fold_masks
from run_mos2_intensity_only_replay import disk_images
from run_mos2_local_intensity_geometry import geometric_folds
from run_mos2_phase_warp_transfer import BOUND, DAMPING, SCALE_NM, motion_transforms, warp_field
from run_mos2_slow_readout_transfer import calibrate, render, rms, smooth_basis
from run_mos2_warp_model_controls import fields
from run_registry_prediction import rotation
from run_wall_heterostrain import PROTOCOL, SOURCE
from shared_compute import atomic_json, checkpoint_guard


def fit_coordinate_motion(field, fit, controls, probe, values, q, train, base, pixel):
    """Fit the position correction with variable detector gains on source pixels."""
    normalized, coefficient = calibrate(base,values,train)
    inverse, coords = motion_transforms(field,fit,pixel)
    direction = np.einsum("xyij,kj->xyki",coords,q)
    phase = 2*np.pi*(field@q.T)
    response = np.pi*(-coefficient[:,0]*np.sin(phase)+coefficient[:,1]*np.cos(phase))
    y = (normalized-np.einsum("xyci,ci->xyc",base,coefficient))[train].T.ravel()
    columns = []
    for channel in range(len(q)):
        slope = SCALE_NM*response[...,channel,None]*direction[...,channel,:]
        spatial = (slope[...,None,:]*controls[...,None]).reshape(*train.shape,-1)
        observed = gaussian_filter(spatial,(probe,probe,0),mode="nearest")[train]
        camera = base[...,channel,:][train]
        columns.append(observed-camera@np.linalg.lstsq(camera,observed,rcond=None)[0])
    x = np.concatenate(columns)
    width = x.shape[1]
    regularizer = DAMPING*np.sqrt(len(y)/width)
    factor = np.linalg.cholesky(x.T@x+regularizer**2*np.eye(width))
    solution = lsq_linear(factor.T,np.linalg.solve(factor,x.T@y),bounds=(-BOUND,BOUND))
    if not solution.success:
        raise ValueError("The source-only registration did not qualify")
    shift = SCALE_NM*(controls@solution.x.reshape(-1,2))@inverse
    return shift, float(np.mean(abs(solution.x)>BOUND-1e-7))


def wall_readout(observed, signals, train, validate):
    scores, sizes = {}, {}
    calibrated = {name:calibrate(response,observed,train) for name,response in signals.items()}
    for i, mask in enumerate(validate):
        if mask.sum()<15:
            continue
        part = str(i)
        scores[part] = {}
        sizes[part] = int(mask.sum())
        for name, response in signals.items():
            norm, camera = calibrated[name]
            estimate = np.einsum("ncp,cp->nc",response[mask],camera)
            scores[part][name]=rms(estimate-norm[mask])
    return {"wall_pixels":sizes,"rmse":scores}


def score(values, ctx, candidates, fit, support, wall_masks, period):
    probe, q, pixel = ctx["probe"],ctx["reciprocal"],ctx["receipt"]["pixel_nm"]
    first, second = [ctx["rings"][name] for name in ("first","second")]
    walls = np.logical_or.reduce(wall_masks)
    distance = distance_transform_edt(~walls)
    # A guard around the union excludes the PSF around all three target
    # wall sectors from every displacement and detector calibration here.
    source = support & (distance > int(np.ceil(2*probe)))
    if source.sum()<300:
        return {"source_pixels":int(source.sum()),"wall_pixels":int(walls.sum()),"folds":[],"fits":{}}
    controls,_,_ = smooth_basis(support.shape,source,period)
    signals, receipts = {}, {}
    for name, field in candidates.items():
        checkpoint_guard()
        baseline = render(field,np.ones(support.shape),q,probe)
        position, on_bound = fit_coordinate_motion(field,fit,controls,probe,values[...,first],q[first],
                                                    source,baseline[...,first,:],pixel)
        signals[name] = baseline[...,second,:]
        corrected = warp_field(field,position)
        signals[name+"_warped"] = render(corrected,np.ones(support.shape),q[second],probe)
        receipts[name] = {"position_rms_px":rms(np.linalg.norm(position[support],axis=-1)),
                          "control_bound_fraction":on_bound}
    results = []
    for train,test in fold_masks(support,probe):
        calibration = train & source
        wall = test & walls
        if calibration.sum()<200 or wall.sum()<60:
            continue
        split = wall_readout(values[...,second],signals,calibration,(wall, *(test & s for s in wall_masks)))
        results.append({"calibration_pixels":int(calibration.sum()),**split})
    return {"source_pixels":int(source.sum()),"wall_pixels":int(walls.sum()),"fits":receipts,"folds":results}


def compare(values, context, interface, meta, selected_group):
    rec = context["receipt"]
    basis, pixel = context["basis"],rec["pixel_nm"]
    theta = np.deg2rad(float(meta["SampleRotation"]))
    y,x = np.indices(values.shape[:2],dtype=float)
    xy = np.stack((x,y),axis=-1)*pixel
    fit = np.asarray(rec["first_ring_affine_fit"])
    choices = [("affine", context["rigid"], context["gradient"], context["support"], fit)]
    local = list(geometric_folds(np.asarray(rec["centers_rc_px"]),xy,pixel,basis,fit,context["support"]))
    if local:
        group, _, warped, local_fit, mask, _ = max(local,key=lambda row:row[4].sum())
        if group!=selected_group:
            raise ValueError("The local center group changed")
        choices.append(("local",warped,local_fit[:2].T@rotation(-theta),mask,local_fit))
    results = {}
    for name, background, gradient, mask, geometry in choices:
        observed = fields(background,gradient,basis,interface)
        wall = tuple(mask & part for part in wall_sectors(background,basis).values())
        period = np.sqrt(abs(np.linalg.det(np.linalg.solve(gradient,basis))))/pixel
        results[name]=score(values,context,observed,geometry,mask,wall,period)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive",type=Path)
    parser.add_argument("release",type=Path)
    parser.add_argument("dependencies",type=Path)
    parser.add_argument("output",type=Path)
    options = parser.parse_args()
    start=perf_counter()
    baseline=PROJECT/"mos2_warp_model_controls_v01.json"
    paths=(PROTOCOL,SOURCE,BENCHMARK,SETTINGS,baseline)
    protocol,source,benchmark,settings,reference = [json.loads(p.read_text()) for p in paths]
    author=load_author_release(options.release,options.dependencies)
    result={
        "scope":"Use all fixed first-ring centers for the affine geometry and the previous most supported disjoint center-fold warp for the local geometry. Before fitting any position correction, exclude all three saddle-point sectors and a two-probe-sigma buffer from the source ring. Exclude the same pixels when fitting constant camera parameters on buffered halves of the second ring. Score only the second-ring wall sectors in the opposite halves, using common geometry-defined pixels for all fields. The ring-to-ring position correction extrapolates across the omitted wall sectors.",
        "limits":"The original centers are measured across the image and the masks are nominal midpoint regions defined with the same source geometry, not traced walls. This is a retrospective direct-diffraction sensitivity check, mainly suitable for parallel MoS2. The corrected predictions inherit information from the opposite ring away from the scored positions.",
        "input_sha256":{f.name:digest(f) for f in (Path(__file__),*paths)},
        "scans":{},
    }
    for group,names in protocol["cases"].items():
        for scan in names:
            checkpoint_guard()
            old=source["scans"][scan]
            meta=read_input_metadata(scan,options.archive)
            images,modes,checksum=disk_images(options.archive,scan,author)
            if checksum!=old["diskset_sha256"] or meta["archive_md5"]!=benchmark["scans"][scan]["archive_md5"]:
                raise ValueError("The archived diffraction inputs changed")
            interface=tuple(old["interface"])
            context=centers_and_context(images,modes,meta["metadata_fields"],settings,interface)
            if context["receipt"]!=old["geometry"]:
                raise ValueError("The fixed first-ring geometry changed")
            prior=reference["scans"][scan]["geometry"].get("local",{})
            compared=compare(images,context,interface,meta["metadata_fields"],prior.get("unused_center_group"))
            result["scans"][scan]={"group":group,"geometry":compared}
            result["elapsed_seconds"]=perf_counter()-start
            atomic_json(options.output,result)
            print(scan,{key:[part["rmse"]["0"] for part in row["folds"]] for key,row in compared.items()},flush=True)


if __name__=="__main__":
    main()
