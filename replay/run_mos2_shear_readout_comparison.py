"""Check whether the diffraction predictions preserve measurable wall shear."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter

from author_release import load_author_release
from bilayer_relaxation import projected_observables
from observe_bilayers import load_dataset, read_input_metadata
from run_heterobilayer_transfer import PROJECT, blur, digest
from run_moire_regime_benchmark import SETTINGS
from run_mos2_cross_ring import TEMPLATES
from run_mos2_directional_diffraction import wall_sectors
from run_mos2_intensity_only_replay import disk_images, geometry
from run_mos2_local_baselines import predict_fields
from run_registry_prediction import rotation
from shared_compute import atomic_json, checkpoint_guard


SOURCE = PROJECT / "mos2_intensity_only_replay_v01.json"


def shear(field, pixel, angle):
    dy, dx = np.gradient(field, pixel, axis=(0, 1))
    gradient = np.stack((dx, dy), axis=-1) @ rotation(-angle)
    _, _, maximum_shear = projected_observables(gradient)
    return maximum_shear


def assess(data, reference, context, templates, settings):
    row,col = data.metadata.get("author_unwrap_crop_origin_row_col",(0,0))
    height,width = data.u_nm.shape[:2]
    region = np.s_[row:row+height,col:col+width]
    if context["rigid"][region].shape != data.u_nm.shape:
        raise ValueError("The archive phase and detector raster dimensions disagree")
    pixel = data.pixel_nm
    angle = np.deg2rad(data.sample_rotation_degrees)
    basis = context["basis"]
    valid = data.source_valid_mask & np.isfinite(data.u_nm).all(axis=-1)
    sigma = settings["smoothing_sigma_pixels"]
    weights = gaussian_filter(valid.astype(float),sigma,mode="constant")
    common = context["support"][region] & valid & (weights > 0.97)
    rigid = context["rigid"][region]
    fit = np.asarray(context["receipt"]["first_ring_affine_fit"])
    nominal = np.asarray(reference["affine_geometry"]["fit_scan"])
    g_archive = nominal[:2].T @ rotation(-angle)
    branch = g_archive @ np.linalg.inv(context["gradient"])
    # Match lattice branches using the archived global geometry alone.
    angles = [rotation(k*np.pi/3) for k in range(6)]
    nearest = min(angles,key=lambda matrix:np.linalg.norm(matrix-branch))
    fields,_ = predict_fields(rigid, context["gradient"], basis, templates)
    probe = context["probe"]
    sectors = wall_sectors(rigid,basis)
    measured = shear(blur(data.u_nm,valid,weights,sigma),pixel,angle)
    observed = {key:float(measured[common&mask].mean())
                for key,mask in sectors.items() if np.count_nonzero(common&mask)>=20}
    if len(observed) != 3:
        raise ValueError("The comparison needs all three wall sectors")
    scores = {}
    for name,field in fields.items():
        motion = gaussian_filter(field-rigid,(probe,probe,0),mode="nearest")
        values = shear(blur((rigid+motion)@nearest.T,valid,weights,sigma),pixel,angle)
        sector_values = {key:float(values[common&mask].mean()) for key,mask in sectors.items()}
        target = np.array(list(observed.values()))
        prediction = np.array(list(sector_values.values()))
        delta = prediction-target
        scores[name] = {"sectors_percent":sector_values,"mean_bias_pp":float(delta.mean()),
                        "directional_rmse_pp":float(delta.std()),
                        "absolute_rmse_pp":float(np.sqrt(np.mean(delta**2)))}
    return {"original_to_archived_rotation":nearest.tolist(),"first_ring_fit_scan":fit.tolist(),
            "frame_fit_residual":float(np.linalg.norm(branch-nearest)),
            "observed_sectors_percent":observed,"models":scores}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive",type=Path)
    parser.add_argument("release",type=Path)
    parser.add_argument("dependencies",type=Path)
    parser.add_argument("output",type=Path)
    args = parser.parse_args()
    benchmark_file = PROJECT/"moire_regime_benchmark_v02.json"
    paths = (SOURCE,benchmark_file,SETTINGS,TEMPLATES)
    source,benchmark,settings,templates = (json.loads(p.read_text()) for p in paths)
    author = load_author_release(args.release,args.dependencies)
    output = {"scope":"Predict relative displacement from first-ring geometry. Compare the full, soliton and simple models' wall-sector shear with the archived processed displacement using a common first-ring mask. Align the relative lattice indexing with archived center geometry before taking the shear tensor.",
              "limits":"The processed displacement comes from the same data. The derived strain depends on the author's branch and smoothing. The common geometry wall masks are different from masks defined separately by each field. Previously examined scans.",
              "input_sha256":{p.name:digest(p) for p in (Path(__file__),*paths)},"scans":{},"failed_scans":{}}
    for scan,row in source["scans"].items():
        checkpoint_guard()
        meta = read_input_metadata(scan,args.archive)
        values,modes,fingerprint = disk_images(args.archive,scan,author)
        if meta["archive_md5"] != row["archive_md5"] or fingerprint != row["diskset_sha256"]:
            raise ValueError("The source map changed")
        context = geometry(values,modes,meta["metadata_fields"],settings)
        if context["receipt"] != row["geometry"]:
            raise ValueError("The first-ring geometry changed")
        data = load_dataset(scan,args.archive,allowed_ids=(scan,),allow_author_crop=True)
        try:
            report = assess(data,benchmark["scans"][scan],context,templates["templates"]["P"],settings)
        except (ValueError,np.linalg.LinAlgError) as exc:
            output["failed_scans"][scan] = str(exc)
            print(scan,str(exc),flush=True)
        else:
            output["scans"][scan] = report
            print(scan,report["models"]["full"],flush=True)
        atomic_json(args.output,output)


if __name__ == "__main__":
    main()
