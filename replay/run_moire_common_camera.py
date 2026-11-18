"""Fit one microscope response shared by all physical candidates."""

import argparse
import json
from pathlib import Path

import numpy as np

from author_detector_replay import rms
from author_release import load_author_release
from observe_bilayers import read_input_metadata
from run_heterobilayer_transfer import PROJECT, digest
from run_moire_common_intensity import BENCHMARK, MIXED_TEMPLATES, centers_and_context, model_fields
from run_moire_regime_benchmark import SETTINGS
from run_mos2_camera_spatial_replay import phase_camera
from run_mos2_cross_ring import TEMPLATES
from run_mos2_frame_sensitivity import fold_masks
from run_mos2_intensity_only_replay import disk_images
from run_mos2_local_baselines import predict_fields
from run_mos2_local_intensity_geometry import geometric_folds
from run_registry_prediction import rotation
from shared_compute import atomic_json, checkpoint_guard


REFERENCE = PROJECT / "moire_common_intensity_replay_v01.json"


def normalize_camera(observed, train):
    low, high = np.quantile(observed[train], (0.05, 0.95), axis=0)
    if np.any(high <= low):
        raise ValueError("The camera calibration pixels have no intensity contrast")
    # Let the two quantiles represent the nominal dark and bright response.
    # This same response is applied to every physical prediction.
    return (observed-low)/(high-low)


def compare(fields, context, images, support):
    second = context["rings"]["second"]
    measured = images[..., second]
    predictions = {name: phase_camera(v, context["reciprocal"][second], context["probe"])[0]
                   for name, v in fields.items()}
    output = []
    for train, test in fold_masks(support, context["probe"]):
        if min(train.sum(), test.sum()) < 300:
            continue
        target = normalize_camera(measured, train)[test]
        output.append({"pixels": int(test.sum()), "rmse": {name: rms(target, signal[test])
                       for name,signal in predictions.items()}})
    return output


def evaluate(values, context, meta, interface, templates, reference):
    basis = context["basis"]

    def fields(background, gradient):
        generated,_ = model_fields(background,gradient,basis,interface,templates)
        if interface == ("MoS2","MoS2","P"):
            added,_ = predict_fields(background,gradient,basis,templates)
            generated.update({k:v for k,v in added.items() if k.startswith("soliton")})
        return generated

    affine = compare(fields(context["rigid"],context["gradient"]),context,values,context["support"])
    receipt = context["receipt"]
    pixel = receipt["pixel_nm"]
    angle = np.deg2rad(float(meta["SampleRotation"]))
    yy,xx = np.indices(values.shape[:2])
    points = np.stack((xx,yy),axis=-1)*pixel
    centers = np.asarray(receipt["centers_rc_px"])
    fit = np.asarray(receipt["first_ring_affine_fit"])
    local = []
    usable = {r["center_group"] for r in reference["local_scores"]}
    for group,_,warp,fit_local,mask,_ in geometric_folds(centers,points,pixel,basis,fit,context["support"]):
        if group not in usable:
            continue
        checkpoint_guard()
        model = fields(warp,fit_local[:2].T@rotation(-angle))
        local.append({"group":group,"scores":compare(model,context,values,mask)})

    def average(rows):
        return {name:float(np.mean([r["rmse"][name] for r in rows])) for name in rows[0]["rmse"]} if rows else {}

    return {"affine":affine,"local":local,"mean_affine":average(affine),
            "mean_local":average([r for f in local for r in f["scores"]])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive",type=Path)
    parser.add_argument("release",type=Path)
    parser.add_argument("dependencies",type=Path)
    parser.add_argument("output",type=Path)
    args = parser.parse_args()
    files = (REFERENCE,BENCHMARK,SETTINGS,TEMPLATES,MIXED_TEMPLATES)
    source,benchmark,settings,templates,mixed = (json.loads(p.read_text()) for p in files)
    author = load_author_release(args.release,args.dependencies)
    result = {"scope":"Use the fifth and ninety-fifth percentiles on the detector-training half to set the same channel scale and background for all models. Hold the first-ring geometry, even cosine response, nominal probe blur and test pixels fixed. Predict experimental intensity on the other half.",
              "limits":"The nominal dark and bright calibration uses measured intensity ranges and assumes the ideal even response. It neither estimates the detector gain independently nor models spatially varying background. The archive maps have been examined previously.",
              "input_sha256":{p.name:digest(p) for p in (Path(__file__),*files)},"scans":{}}
    for scan,reference in source["scans"].items():
        checkpoint_guard()
        meta = read_input_metadata(scan,args.archive)
        images,modes,fingerprint = disk_images(args.archive,scan,author)
        if fingerprint != reference["diskset_sha256"] or meta["archive_md5"] != benchmark["scans"][scan]["archive_md5"]:
            raise ValueError("The source map changed")
        interface = tuple(reference["interface"])
        context = centers_and_context(images,modes,meta["metadata_fields"],settings,interface)
        if context["receipt"] != reference["geometry"]:
            raise ValueError("The first-ring geometry changed")
        learned = templates["templates"][interface[-1]] if interface[0] == interface[1] else mixed["templates"]
        score = evaluate(images,context,meta["metadata_fields"],interface,learned,reference)
        result["scans"][scan] = {"interface":list(interface),**score}
        atomic_json(args.output,result)
        print(scan,score["mean_affine"],flush=True)


if __name__ == "__main__":
    main()
