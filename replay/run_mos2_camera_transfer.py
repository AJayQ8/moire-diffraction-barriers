"""Transfer fixed normalized detector coefficients between parallel MoS2 scans."""

import argparse
import json
from pathlib import Path

import numpy as np

from author_detector_replay import rms
from author_release import load_author_release
from observe_bilayers import read_input_metadata
from run_heterobilayer_transfer import PROJECT, digest
from run_moire_common_intensity import BENCHMARK, centers_and_context, model_fields
from run_moire_common_camera import normalize_camera
from run_moire_regime_benchmark import SETTINGS
from run_mos2_camera_spatial_replay import phase_camera
from run_mos2_cross_ring import TEMPLATES
from run_mos2_frame_sensitivity import fold_masks
from run_mos2_intensity_only_replay import disk_images
from run_mos2_local_baselines import predict_fields
from shared_compute import atomic_json, checkpoint_guard


DEVELOPMENT = PROJECT / "mos2_intensity_only_development_v01.json"
REPLAY = PROJECT / "mos2_intensity_only_replay_v01.json"
COMMON = PROJECT / "moire_common_intensity_replay_v01.json"


def records(archive, author, rows, benchmark, common, settings, templates):
    for scan, reference in rows.items():
        checkpoint_guard()
        meta = read_input_metadata(scan, archive)
        values, modes, fingerprint = disk_images(archive, scan, author)
        if fingerprint != reference["diskset_sha256"] or meta["archive_md5"] != benchmark["scans"][scan]["archive_md5"]:
            raise ValueError("A diffraction source changed")
        interface = tuple(benchmark["scans"][scan]["interface"])
        context = centers_and_context(values,modes,meta["metadata_fields"],settings,interface)
        if context["receipt"] != common["scans"][scan]["geometry"]:
            raise ValueError("The first-ring geometry changed")
        basis = context["basis"]
        fields,_ = model_fields(context["rigid"],context["gradient"],basis,interface,templates)
        other,_ = predict_fields(context["rigid"],context["gradient"],basis,templates)
        fields.update({name:v for name,v in other.items() if name.startswith("soliton")})
        second = context["rings"]["second"]
        signal = {key:phase_camera(value,context["reciprocal"][second],context["probe"])[0]
                  for key,value in fields.items()}
        yield scan, values[...,second], signal, list(fold_masks(context["support"],context["probe"]))


def train_coefficients(rows):
    summaries = {}
    for scan,observed,signals,folds in rows:
        train = folds[0][0] | folds[0][1]
        y = normalize_camera(observed,train)[train]
        estimates = {}
        for name,image in signals.items():
            x = image[train].ravel()
            coefficients = np.linalg.lstsq(np.column_stack((x,np.ones(len(x)))),y.ravel(),rcond=None)[0]
            if coefficients[0] < 0:
                raise ValueError("The development contrast inverted")
            estimates[name] = coefficients.tolist()
        summaries[scan] = estimates
    models = next(iter(summaries.values()))
    by_model = {name:np.median([r[name] for r in summaries.values()],axis=0) for name in models}
    pooled = np.median(list(by_model.values()),axis=0)
    return summaries, by_model, pooled


def evaluate(rows,coefficients,common):
    output = {}
    for scan,observed,signals,folds in rows:
        checks = []
        for train,test in folds:
            y = normalize_camera(observed,train)[test]
            row = {}
            for name,predicted in signals.items():
                x = predicted[test]
                gain,offset = coefficients[name]
                row[name] = {"model_specific_from_development":rms(y,gain*x+offset),
                             "same_for_all_from_development":rms(y,common[0]*x+common[1]),
                             "ideal_even_response":rms(y,x)}
            checks.append(row)
        output[scan] = checks
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive",type=Path)
    parser.add_argument("release",type=Path)
    parser.add_argument("dependencies",type=Path)
    parser.add_argument("output",type=Path)
    args = parser.parse_args()
    paths = (DEVELOPMENT,REPLAY,COMMON,BENCHMARK,SETTINGS,TEMPLATES)
    development,replay,common,benchmark,settings,templates = (json.loads(p.read_text()) for p in paths)
    author = load_author_release(args.release,args.dependencies)
    arguments = (benchmark,common,settings,templates["templates"]["P"])
    source = records(args.archive,author,development["scans"],*arguments)
    fits,by_model,shared = train_coefficients(source)
    test = records(args.archive,author,replay["scans"],*arguments)
    scored = evaluate(test,by_model,shared)
    result = {"scope":"Estimate a single even gain and offset across all channels of each of four development maps after training-range normalization. Use medians to transfer those parameters to ten other maps. Include one median response for all models and the ideal response. The ten targets supply geometry and training-pixel intensity quantiles; the gain and offset receive no target-model fit.",
              "limits":"Retrospective after the target maps were examined. The development and target map identities do not establish independent specimens. Detector normalization still uses training pixels from each target map.",
              "input_sha256":{p.name:digest(p) for p in (Path(__file__),*paths)},
              "development_map_coefficients":fits,"frozen_model_coefficients":{k:v.tolist() for k,v in by_model.items()},
              "frozen_common_coefficients":shared.tolist(),"scans":scored}
    atomic_json(args.output,result)
    print("scored",len(scored),flush=True)


if __name__ == "__main__":
    main()
