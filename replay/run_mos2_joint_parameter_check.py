"""Check whether the two published parameter sets transfer in diffraction and shear."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter

from author_release import load_author_release
from observe_bilayers import load_dataset, read_input_metadata
from run_heterobilayer_transfer import PROJECT, blur, digest
from run_moire_common_intensity import BENCHMARK
from run_moire_regime_benchmark import SETTINGS
from run_mos2_constrained_equilibrium import score_fields
from run_mos2_cross_ring import TEMPLATES
from run_mos2_directional_diffraction import wall_sectors
from run_mos2_intensity_only_replay import disk_images, geometry
from run_mos2_local_baselines import predict_fields
from run_mos2_local_intensity_geometry import geometric_folds
from run_mos2_shear_readout_comparison import shear
from run_registry_prediction import rotation
from shared_compute import atomic_json, checkpoint_guard


SOURCE = PROJECT / "moire_common_intensity_replay_v01.json"
MODELS = ("full","alternate_inputs","energy_scaled_first_order","soliton_effective_adhesion","rigid")


def measure(fields, data, reference, context, settings, support, rigid):
    angle = np.deg2rad(data.sample_rotation_degrees)
    pixel = data.pixel_nm
    y,x = data.metadata.get("author_unwrap_crop_origin_row_col",(0,0))
    height,width = data.u_nm.shape[:2]
    crop = np.s_[y:y+height,x:x+width]
    valid = data.source_valid_mask & np.isfinite(data.u_nm).all(axis=-1)
    sigma = settings["smoothing_sigma_pixels"]
    weights = gaussian_filter(valid.astype(float),sigma,mode="constant")
    keep = support[crop] & valid & (weights > 0.97)
    archive = np.asarray(reference["affine_geometry"]["gradient_crystal"])
    transform = archive@np.linalg.inv(context["gradient"])
    nearest = min((rotation(k*np.pi/3) for k in range(6)),key=lambda m:np.linalg.norm(m-transform))
    probe = context["probe"]
    sectors = wall_sectors(rigid[crop],context["basis"])
    masks = {name:keep&value for name,value in sectors.items()}
    if min(v.sum() for v in masks.values()) < 20:
        return {}
    observed = shear(blur(data.u_nm,valid,weights,sigma),pixel,angle)
    target = np.array([observed[m].mean() for m in masks.values()])
    output = {}
    for name,field in fields.items():
        correction = gaussian_filter(field-rigid,(probe,probe,0),mode="nearest")
        measured = shear(blur((rigid+correction)[crop]@nearest.T,valid,weights,sigma),pixel,angle)
        values = np.array([measured[m].mean() for m in masks.values()])
        delta = values-target
        output[name] = {"mean_bias_pp":float(delta.mean()),"directional_rmse_pp":float(delta.std()),
                        "absolute_rmse_pp":float(np.sqrt(np.mean(delta**2)))}
    return {"target_percent":target.tolist(),"sector_pixels":[int(v.sum()) for v in masks.values()],"models":output}


def evaluate(images,context,data,templates,benchmark,reference,settings):
    receipt = context["receipt"]
    pixel = receipt["pixel_nm"]
    angle = np.deg2rad(data.sample_rotation_degrees)
    yy,xx = np.indices(images.shape[:2])
    points = np.stack((xx,yy),axis=-1)*pixel
    fit = np.asarray(receipt["first_ring_affine_fit"])
    centers = np.asarray(receipt["centers_rc_px"])
    basis = context["basis"]
    rows = []
    for group,_,warp,local,mask,_ in geometric_folds(centers,points,pixel,basis,fit,context["support"]):
        checkpoint_guard()
        earlier = next((r for r in reference["local_scores"] if r["center_group"] == group),None)
        if earlier is None:
            continue
        raw,_ = predict_fields(warp,local[:2].T@rotation(-angle),basis,templates)
        fields = {key:raw[key] for key in MODELS}
        scores = score_fields(fields,context,images,mask)
        for r,s in zip(scores,earlier["scores"],strict=True):
            if abs(r["rmse"]["full"]-s["rmse"]["full"]) > 1e-12:
                raise ValueError("The common diffraction comparison changed")
        rows.append({"center_group":group,"diffraction":scores,
                     "shear":measure(fields,data,benchmark,context,settings,mask,warp)})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive",type=Path)
    parser.add_argument("release",type=Path)
    parser.add_argument("dependencies",type=Path)
    parser.add_argument("output",type=Path)
    args = parser.parse_args()
    paths = (SOURCE,BENCHMARK,SETTINGS,TEMPLATES)
    source,benchmark,settings,templates = (json.loads(p.read_text()) for p in paths)
    author = load_author_release(args.release,args.dependencies)
    result = {"scope":"Compare the two published MoS2 input sets with identical first-ring center warps. Score disjoint second-ring pixels and common geometry-defined wall sectors in the archived phase-gradient map.",
              "limits":"Shear is a differentiated and smoothed displacement readout. Both targets use the same experiment and support. The alternate parameter set has already been compared on these data.",
              "input_sha256":{p.name:digest(p) for p in (Path(__file__),*paths)},"scans":{}}
    for scan,reference in source["scans"].items():
        if reference["interface"] != ["MoS2","MoS2","P"]:
            continue
        checkpoint_guard()
        meta = read_input_metadata(scan,args.archive)
        images,modes,fingerprint = disk_images(args.archive,scan,author)
        record = benchmark["scans"][scan]
        if fingerprint != reference["diskset_sha256"] or meta["archive_md5"] != record["archive_md5"]:
            raise ValueError("The diffraction source changed")
        context = geometry(images,modes,meta["metadata_fields"],settings)
        if context["receipt"] != reference["geometry"]:
            raise ValueError("The first-ring geometry changed")
        data = load_dataset(scan,args.archive,allowed_ids=(scan,),allow_author_crop=True)
        scores = evaluate(images,context,data,templates["templates"]["P"],record,reference,settings)
        result["scans"][scan] = scores
        atomic_json(args.output,result)
        print(scan,len(scores),sum(bool(r["shear"]) for r in scores),flush=True)


if __name__ == "__main__":
    main()
