"""Separate slow common brightness changes from slow layer-shift errors."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.optimize import lsq_linear

from author_release import load_author_release
from observe_bilayers import read_input_metadata
from run_heterobilayer_transfer import PROJECT, digest
from run_moire_common_intensity import BENCHMARK, centers_and_context
from run_moire_regime_benchmark import SETTINGS
from run_mos2_camera_spatial_replay import phase_camera
from run_mos2_frame_sensitivity import fold_masks
from run_mos2_intensity_only_replay import camera_fit, disk_images
from run_mos2_local_intensity_geometry import geometric_folds
from run_mos2_local_wall_transfer import predictions
from run_registry_prediction import rotation
from run_wall_heterostrain import PROTOCOL, SOURCE
from shared_compute import atomic_json, checkpoint_guard


PILOT = {"DS1", "DS2", "DS7", "DS10", "DS13", "DS16", "DS23", "DS25"}
CORRECTIONS = ("brightness", "displacement", "joint")
GAIN = .10
SHIFT_NM = .012


def smooth_basis(shape, support, period):
    """Use piecewise-bilinear controls roughly one moiré period apart."""
    y, x = np.indices(shape)
    # Limit conditioning and memory on very small or large moiré cells.
    counts = [int(np.clip(np.ceil((n-1)/period),1,6)) for n in shape]
    y = y*counts[0]/(shape[0]-1)
    x = x*counts[1]/(shape[1]-1)
    positions = np.stack((x, y), axis=-1)
    nodes = np.array([(i,j) for j in range(counts[0]+1) for i in range(counts[1]+1)])
    tent = np.maximum(1-np.abs(positions[...,None,:]-nodes),0.).prod(axis=-1)
    usable = np.max(tent[support],axis=0) > 0.2
    return tent[...,usable], nodes[usable], counts


def rms(values):
    return float(np.sqrt(np.mean(values**2)))


def render(field, flux, reciprocal, probe):
    phase = np.pi*(field@reciprocal.T)
    cos, sin = np.cos(phase), np.sin(phase)
    # Include a smoothly varying multiplicative factor for the constant
    # intensity component as well as for the even and odd interference terms.
    unblurred = np.stack((cos**2, sin*cos, np.ones_like(cos)),axis=-1)*flux[...,None,None]
    return gaussian_filter(unblurred,(probe,probe,0,0),mode="nearest")


def calibrate(signals, values, pixels):
    normalized = values / np.ptp(values[pixels],axis=0)
    matrices = signals[pixels].transpose(1,0,2)
    target = normalized[pixels].T[...,None]
    coefficients = np.linalg.solve(matrices.transpose(0,2,1)@matrices,
                                   matrices.transpose(0,2,1)@target)[...,0]
    return normalized, coefficients


def columns(field, reciprocal, values, signals, probe, controls, support):
    """Project slow perturbations away from the three detector coefficients."""
    normalized, coefficients = calibrate(signals,values,support)
    amplitude, odd, offset = coefficients.T
    phase = 2*np.pi*(field@reciprocal.T)
    cosine, sine = np.cos(phase), np.sin(phase)
    image = amplitude*(1+cosine)/2 + odd*sine/2 + offset
    derivative = np.pi*(-amplitude*sine+odd*cosine)[...,None]*reciprocal
    residual = normalized-np.einsum("xycp,cp->xyc",signals,coefficients)
    count = controls.shape[-1]
    block = []
    for i in range(len(reciprocal)):
        grad = np.concatenate((GAIN*image[...,i,None], SHIFT_NM*derivative[...,i,:]),axis=-1)
        raw = (grad[...,None,:]*controls[...,None]).reshape(*support.shape,3*count)
        low = gaussian_filter(raw,(probe,probe,0),mode="nearest")[support]
        camera = signals[...,i,:][support]
        low -= camera @ np.linalg.lstsq(camera,low,rcond=None)[0]
        block.append(low)
    return np.concatenate(block), residual[support].T.ravel()


def estimate(lhs, rhs, controls, method):
    width = controls.shape[-1]
    indices = {"brightness":np.arange(0,3*width,3),
               "displacement":np.flatnonzero(np.tile((False,True,True),width)),
               "joint":np.arange(3*width)}[method]
    chosen = lhs[:,indices]
    damping = .012*np.sqrt(len(rhs)/width)
    # Factor the damped normal matrix so the bounded solve stays independent
    # of the number of scan pixels.
    triangular = np.linalg.cholesky(chosen.T@chosen+damping**2*np.eye(len(indices)))
    projected = np.linalg.solve(triangular,chosen.T@rhs)
    result = lsq_linear(triangular.T,projected,bounds=(-1.5,1.5))
    if not result.success:
        raise ValueError("The bounded smooth-transfer fit failed")
    output = np.zeros(3*width)
    output[indices] = result.x
    return components(controls,output.reshape(width,3))


def components(controls, coefficients):
    change = controls@coefficients
    return GAIN*change[...,0], SHIFT_NM*change[...,1:]


def qualify(field, context, values, controls, nodes, support, signals):
    q, probe = context["reciprocal"], context["probe"]
    coefficients = calibrate(signals,values,support)[1]
    coefficients = .55*np.column_stack((np.ones(len(q)),
                                           .18*np.where(np.arange(len(q))%2==0,1.,-1.),
                                           .65*np.ones(len(q))))+ .45*coefficients
    design = nodes.sum(axis=-1)
    injected_gain, injected_shift = components(controls, np.column_stack((.6*np.sin(design),
                          .55*np.cos(.73*design), -.45*np.sin(1.3*design))))
    synthetic = render(field+injected_shift,1+injected_gain,q,probe)
    synthetic = np.einsum("xycp,cp->xyc",synthetic,coefficients)
    choose = context["rings"]["first"]
    system = columns(field,q[choose],synthetic[...,choose],signals[...,choose,:],
                     probe,controls,support)
    gain,shift = estimate(*system,controls,"joint")
    tested = context["rings"]["second"]
    repaired = render(field+shift,1+gain,q[tested],probe)
    normalized, fit = calibrate(repaired,synthetic[...,tested],support)
    original = signals[...,tested,:]
    before = calibrate(original,synthetic[...,tested],support)[1]
    raw_residue = normalized[support]-np.einsum("ncp,cp->nc",original[support],before)
    corrected = normalized[support]-np.einsum("ncp,cp->nc",repaired[support],fit)
    # Rank-deficient locations must not define the scale of validation. This
    # task is a discrimination check, so report the known injection recovery.
    return {"initial_rmse":rms(raw_residue),"corrected_rmse":rms(corrected),
            "brightness_truth_rms_fraction":rms(injected_gain[support]),
            "brightness_error_after_centering":rms((gain-injected_gain)[support]-(gain-injected_gain)[support].mean()),
            "displacement_truth_rms_nm":rms(injected_shift[support]),
            "displacement_error_after_centering_nm":rms((shift-injected_shift)[support]-
                                                        (shift-injected_shift)[support].mean(axis=0))}


def score(values, context, field, support, controls, nodes):
    q, probe = context["reciprocal"], context["probe"]
    background = render(field,np.ones(support.shape),q,probe)
    even, odd = phase_camera(field,q,probe)
    if not np.allclose(background[...,:2],np.stack((even,odd),axis=-1),rtol=0,atol=1e-12):
        raise ValueError("The slow correction does not reproduce the constant camera")
    choices = (("first","second"),("second","first"))
    output = {"control_nodes":len(nodes),"supported_pixels":int(support.sum()),
              "synthetic":qualify(field,context,values,controls,nodes,support,background),"rings":{}}
    for origin, target in choices:
        checkpoint_guard()
        first,second = (context["rings"][k] for k in (origin,target))
        x,y = values[...,first],values[...,second]
        args = columns(field,q[first],x,background[...,first,:],probe,controls,support)
        prepared = {"fixed":background[...,second,:]}
        diagnostics = {}
        for correction in CORRECTIONS:
            gain,shift = estimate(*args,controls,correction)
            prepared[correction]=render(field+shift,1+gain,q[second],probe)
            diagnostics[correction]={"brightness_rms_fraction":rms(gain[support]),
                                     "displacement_vector_rmse_nm":rms(np.linalg.norm(shift[support],axis=-1))}
        rows = []
        for train,test in fold_masks(support,probe):
            if min(train.sum(),test.sum())<300:
                continue
            norm = y/np.ptp(y[train],axis=0)
            errors = {name:rms(np.einsum("ncp,cp->nc",s[test],calibrate(s,y,train)[1])-norm[test])
                      for name,s in prepared.items()}
            earlier, truth, _ = camera_fit(even[...,second],odd[...,second],y,train,test,allow_odd=True)
            if abs(errors["fixed"]-rms(earlier-truth))>1e-12:
                raise ValueError("The nominal camera score changed")
            rows.append({"test_pixels":int(test.sum()),"rmse":errors})
        output["rings"][target]={"source":origin,"fits":diagnostics,"folds":rows}
    return output


def evaluate(images, context, interface, meta):
    basis, g, support = (context[k] for k in ("basis","gradient","support"))
    ref = context["receipt"]
    pixel = ref["pixel_nm"]
    centers, fit = [np.asarray(ref[key]) for key in ("centers_rc_px","first_ring_affine_fit")]
    y,x = np.indices(support.shape,dtype=float)
    xy = pixel*np.stack((x,y),axis=-1)
    angle = np.deg2rad(float(meta["SampleRotation"]))
    options = list(geometric_folds(centers,xy,pixel,basis,fit,support))
    entries = [("affine",context["rigid"],g,support,None)]
    if options:
        # The choice uses only the geometry and image extent, never an
        # intensity residual from the validation ring.
        group, _, warp, local, mask, _ = max(options,key=lambda row:row[4].sum())
        entries.append(("local",warp,local[:2].T@rotation(-angle),mask,group))
    rows = {}
    for name, warp, strain, mask, group in entries:
        checkpoint_guard()
        period = np.sqrt(abs(np.linalg.det(np.linalg.solve(strain,basis))))/pixel
        controls,nodes,limits = smooth_basis(mask.shape,mask,period)
        solution=predictions({name:warp},strain,basis,interface)[name]["full"]
        rows[name]={"moiré_period_nm":float(period*pixel),"maximum_blocks":limits,"unused_center_group":group,
                    **score(images,context,solution,mask,controls,nodes)}
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive",type=Path)
    parser.add_argument("release",type=Path)
    parser.add_argument("dependencies",type=Path)
    parser.add_argument("output",type=Path)
    parser.add_argument("--pilot",action="store_true")
    args = parser.parse_args()
    began = perf_counter()
    paths = PROTOCOL,SOURCE,BENCHMARK,SETTINGS
    protocol,source,benchmark,settings = [json.loads(p.read_text()) for p in paths]
    author = load_author_release(args.release,args.dependencies)
    report = {
        "scope": "At the fixed physical relaxation, fit a spatially slowly varying multiplicative camera brightness, a shift in local registry, or both on only one diffraction ring. Represent all three corrections at roughly the length of a moiré cell, using bounded bilinear control fields. Transfer the corrections without refitting to the other ring, where calibrating only three constant camera coefficients spatially precedes the usual disjoint test. Repeat the same procedure under affine geometry and the first-ring local center fold with the most usable pixels. Use an injected slow gain and shift to test how well the model separates them.",
        "limits": "This is an error diagnosis within previously opened experimental scans, using source ring readings at the scored locations of the other ring. A common gain could be caused by several beam or specimen effects. A transferred local registry shift could be physical variability or scan geometry. The corrections are bounded linearized approximations, and a weak result does not exclude more complex imaging.",
        "input_sha256": {f.name:digest(f) for f in (Path(__file__),*paths)},
        "scans": {},
    }
    for group,names in protocol["cases"].items():
        for scan in names:
            if args.pilot and scan not in PILOT:
                continue
            checkpoint_guard()
            record = source["scans"][scan]
            meta = read_input_metadata(scan,args.archive)
            values,modes,checksum = disk_images(args.archive,scan,author)
            if checksum!=record["diskset_sha256"] or meta["archive_md5"]!=benchmark["scans"][scan]["archive_md5"]:
                raise ValueError("The source disks changed")
            interface = tuple(record["interface"])
            context = centers_and_context(values,modes,meta["metadata_fields"],settings,interface)
            if context["receipt"]!=record["geometry"]:
                raise ValueError("The first-ring geometry changed")
            answer = evaluate(values,context,interface,meta["metadata_fields"])
            report["scans"][scan]={"group":group,"geometry":answer}
            report["elapsed_seconds"]=perf_counter()-began
            atomic_json(args.output,report)
            print(scan,{k:[r["rmse"] for r in v["rings"]["second"]["folds"]] for k,v in answer.items()},
                  flush=True)


if __name__ == "__main__":
    main()
