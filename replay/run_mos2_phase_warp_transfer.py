"""Distinguish a shift of the image coordinates from a shift of registry."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates
from scipy.optimize import lsq_linear

from author_release import load_author_release
from observe_bilayers import read_input_metadata
from run_heterobilayer_transfer import PROJECT, digest
from run_moire_common_intensity import BENCHMARK, centers_and_context
from run_moire_regime_benchmark import SETTINGS
from run_mos2_frame_sensitivity import fold_masks
from run_mos2_intensity_only_replay import disk_images
from run_mos2_local_intensity_geometry import geometric_folds
from run_mos2_local_wall_transfer import predictions
from run_mos2_slow_readout_transfer import calibrate, render, rms, smooth_basis
from run_registry_prediction import rotation
from run_wall_heterostrain import PROTOCOL, SOURCE
from shared_compute import atomic_json, checkpoint_guard


RINGS = ("first", "second")
SCALE_NM = .012
BOUND = 1.5
DAMPING = .012
PILOT = {"DS1", "DS2", "DS7", "DS10", "DS13", "DS16", "DS23", "DS25"}


def warp_field(field, pixels):
    y, x = np.indices(field.shape[:2], dtype=float)
    offset = [y+pixels[...,1], x+pixels[...,0]]
    return np.stack([map_coordinates(plane,offset,order=3,mode="nearest")
                     for plane in np.moveaxis(field,-1,0)],axis=-1)


def motion_transforms(field, fitted, pixel):
    # Both alternatives use the same control coefficients in nm of affine
    # registry shift. Convert to actual pixel offsets using the affine
    # geometry, then apply the spatial gradient of the relaxed field.
    inverse = np.linalg.inv(fitted[:2]*pixel)
    dy, dx = np.gradient(field, axis=(0,1), edge_order=2)
    jacobian = np.stack((dx,dy),axis=-2)
    return inverse, inverse@jacobian


def systems(field, values, mask, q, probe, base, controls, coordinate_transform):
    normalized, camera = calibrate(base,values,mask)
    phase = 2*np.pi*(field@q.T)
    coefficient = np.pi*(-camera[:,0]*np.sin(phase)+camera[:,1]*np.cos(phase))
    initial = np.einsum("xyci,ci->xyc",base,camera)
    target = (normalized-initial)[mask].T.ravel()
    directions = {"registry_shift":np.broadcast_to(q, (*mask.shape,*q.shape)),
                  "coordinate_warp":np.einsum("xyij,kj->xyki",coordinate_transform,q)}
    lhs = {}
    for name, direction in directions.items():
        responses = []
        for j in range(len(q)):
            slope = SCALE_NM*coefficient[...,j,None]*direction[...,j,:]
            terms = (slope[...,None,:]*controls[...,None]).reshape(*mask.shape,-1)
            columns = gaussian_filter(terms,(probe,probe,0),mode="nearest")[mask]
            design = base[...,j,:][mask]
            responses.append(columns-design@np.linalg.lstsq(design,columns,rcond=None)[0])
        lhs[name] = np.concatenate(responses)
    return lhs, target


def outputs(field, values, context, mask, controls, fitted, source, target, synthetic=False):
    inverse, coordinate = motion_transforms(field,fitted,context["receipt"]["pixel_nm"])
    q = context["reciprocal"]
    probe = context["probe"]
    base = render(field,np.ones(mask.shape),q,probe)
    i, j = [context["rings"][name] for name in (source,target)]
    matrix, observation = systems(field,values[...,i],mask,q[i],probe,base[...,i,:],controls,coordinate)
    target_values = values[...,j]
    camera = {"fixed":base[...,j,:]}
    estimates = {}
    for name, design in matrix.items():
        width = design.shape[1]
        damping = DAMPING*np.sqrt(len(observation)/width)
        factor = np.linalg.cholesky(design.T@design+damping**2*np.eye(width))
        rhs = np.linalg.solve(factor,design.T@observation)
        fit = lsq_linear(factor.T,rhs,bounds=(-BOUND,BOUND))
        if not fit.success:
            raise ValueError("The phase or coordinate fit did not converge")
        coefficient = fit.x.reshape(controls.shape[-1],2)
        motion = SCALE_NM*(controls@coefficient)
        if name == "registry_shift":
            corrected = field+motion
        else:
            corrected = warp_field(field,motion@inverse)
        camera[name] = render(corrected,np.ones(mask.shape),q[j],probe)
        estimates[name] = {"rms_shift_nm":rms(np.linalg.norm((corrected-field)[mask],axis=-1)),
                           "rms_coordinate_displacement_px":rms(np.linalg.norm((motion@inverse)[mask],axis=-1))
                           if name == "coordinate_warp" else None,
                           "fraction_coefficients_at_bound":float(np.mean(abs(fit.x)>BOUND-1e-7)),
                           "source_linear_fit_rmse":rms(design@fit.x-observation)}
    rows = []
    folds = ((mask,mask),) if synthetic else fold_masks(mask,probe)
    for train,test in folds:
        if min(train.sum(),test.sum())<300:
            continue
        score = {}
        for name, image in camera.items():
            normalized, params = calibrate(image,target_values,train)
            prediction = np.einsum("nci,ci->nc",image[test],params)
            score[name] = rms(prediction-normalized[test])
        rows.append({"pixels":int(test.sum()),"rmse":score})
    return {"source_ring":source,"target_ring":target,"fits":estimates,"scores":rows}


def qualify(field, values, context, mask, controls, nodes, fitted):
    q, probe = context["reciprocal"], context["probe"]
    base = render(field,np.ones(mask.shape),q,probe)
    detector = calibrate(base,values,mask)[1]
    label = nodes[:,0]+1.28*nodes[:,1]
    motion = .7*SCALE_NM*(controls@np.column_stack((np.sin(label),np.cos(label*.79))))
    inverse, _ = motion_transforms(field,fitted,context["receipt"]["pixel_nm"])
    result = {}
    for name, corrected in (("registry_shift",field+motion),("coordinate_warp",warp_field(field,motion@inverse))):
        synthetic = np.einsum("xyci,ci->xyc",render(corrected,np.ones(mask.shape),q,probe),detector)
        out = outputs(field,synthetic,context,mask,controls,fitted,"first","second",synthetic=True)
        result[name] = out["scores"][0]["rmse"]
        if result[name][name] >= result[name]["fixed"]:
            raise ValueError("The synthetic correction did not improve its own injected change")
    return result


def evaluate(image, ctx, interface, metadata, recorded):
    receipt = ctx["receipt"]
    pixel, basis = receipt["pixel_nm"], ctx["basis"]
    fit = np.asarray(receipt["first_ring_affine_fit"])
    angle = np.deg2rad(float(metadata["SampleRotation"]))
    yy, xx = np.indices(image.shape[:2], dtype=float)
    xy = pixel*np.stack((xx,yy),axis=-1)
    options = list(geometric_folds(np.asarray(receipt["centers_rc_px"]),xy,pixel,basis,fit,ctx["support"]))
    candidates = [("affine",ctx["rigid"],ctx["gradient"],ctx["support"],fit,None)]
    if options:
        group, _, warp, local, extra, _ = max(options,key=lambda row:row[4].sum())
        candidates.append(("local",warp,local[:2].T@rotation(-angle),extra,local,group))
    report = {}
    for name, warp, gradient, mask, fitted, holdout in candidates:
        checkpoint_guard()
        past = recorded[name]
        if holdout != past["unused_center_group"]:
            raise ValueError("The most supported local geometry changed")
        period = np.sqrt(abs(np.linalg.det(np.linalg.solve(gradient,basis))))/pixel
        controls,nodes,_ = smooth_basis(mask.shape,mask,period)
        solution = predictions({name:warp},gradient,basis,interface)[name]["full"]
        result = {target:outputs(solution,image,ctx,mask,controls,fitted,source,target)
                  for source,target in (RINGS,RINGS[::-1])}
        for target,row in result.items():
            for new, old in zip(row["scores"],past["rings"][target]["folds"],strict=True):
                if new["pixels"] != old["test_pixels"] or abs(new["rmse"]["fixed"]-old["rmse"]["fixed"])>1e-12:
                    raise ValueError("The earlier slow-shift calibration or mask changed")
        report[name]={"unused_center_group":holdout, "control_nodes":len(nodes), "rings":result,
                      "synthetic":qualify(solution,image,ctx,mask,controls,nodes,fitted)}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive",type=Path)
    parser.add_argument("release",type=Path)
    parser.add_argument("dependencies",type=Path)
    parser.add_argument("output",type=Path)
    parser.add_argument("--pilot",action="store_true")
    parser.add_argument("--resume",action="store_true")
    args = parser.parse_args()
    start = perf_counter()
    reference = PROJECT/"mos2_slow_readout_transfer_v01.json"
    paths = PROTOCOL,SOURCE,BENCHMARK,SETTINGS,reference
    protocol,source,benchmark,settings,prior = [json.loads(p.read_text()) for p in paths]
    author = load_author_release(args.release,args.dependencies)
    report = {
        "scope": "Compare smooth corrections learned on one ring. Registry shift adds a local vector directly to the layer displacement. Coordinate warp moves the image positions before evaluating the same relaxed displacement field. Both perturbations have the same number of controls and the same bound in equivalent affine disregistry. Compare directly against integrated intensities of the other ring and include simulated perturbations as qualification.",
        "limits": "This uses the opposite diffraction ring at the same locations. Either transformation can include physical and instrumental variation. The two shape changes can coincide locally if the original displacement is affine. The coordinate observer maps the whole relative-displacement field and then applies the same Gaussian probe. It may miss some raster-specific microscope distortions. The result compares sparse smooth perturbations and is retrospective.",
        "input_sha256": {p.name:digest(p) for p in (Path(__file__),*paths)},
        "scans": {},
    }
    if args.resume:
        saved = json.loads(args.output.read_text())
        known = dict(saved["input_sha256"])
        known.pop(Path(__file__).name)
        if known != {p.name:digest(p) for p in paths} or saved["scope"] != report["scope"]:
            raise ValueError("The transfer inputs changed since the checkpoint")
        report["resumed_source_sha256"] = saved["input_sha256"]
        report["resumed_checkpoint_sha256"] = digest(args.output)
        report["scans"] = saved["scans"]
        start -= saved["elapsed_seconds"]
    for group,names in protocol["cases"].items():
        for scan in names:
            if scan in report["scans"]:
                continue
            if args.pilot and scan not in PILOT:
                continue
            checkpoint_guard()
            saved = source["scans"][scan]
            meta = read_input_metadata(scan,args.archive)
            images,modes,fingerprint = disk_images(args.archive,scan,author)
            if fingerprint!=saved["diskset_sha256"] or meta["archive_md5"]!=benchmark["scans"][scan]["archive_md5"]:
                raise ValueError("The input diffraction changed")
            interface = tuple(saved["interface"])
            context = centers_and_context(images,modes,meta["metadata_fields"],settings,interface)
            if context["receipt"] != saved["geometry"]:
                raise ValueError("The fixed first-ring geometry changed")
            row = evaluate(images,context,interface,meta["metadata_fields"],prior["scans"][scan]["geometry"])
            report["scans"][scan]={"group":group,"geometry":row}
            report["elapsed_seconds"]=perf_counter()-start
            atomic_json(args.output,report)
            print(scan,{key:[f["rmse"] for f in value["rings"]["second"]["scores"]]
                        for key,value in row.items()},flush=True)


if __name__ == "__main__":
    main()
