"""Compare Gaussian averaging with a coherent two-disk overlap observer."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates
from scipy.special import expit

from author_release import load_author_release
from bilayer_relaxation import make_grid, sample_disregistry
from observe_bilayers import read_input_metadata
from run_ds31_phase_wavevector_check import electron_wavelength_nm
from run_ds31_pupil_transfer import bands
from run_heterobilayer_sparse_geometry import solve_from_geometry
from run_heterobilayer_transfer import PROJECT, digest
from run_moire_common_intensity import BENCHMARK, centers_and_context
from run_moire_regime_benchmark import SETTINGS
from run_mos2_camera_spatial_replay import phase_camera
from run_mos2_frame_sensitivity import fold_masks
from run_mos2_intensity_only_replay import camera_fit, disk_images
from run_wall_heterostrain import PROTOCOL, SOURCE
from shared_compute import atomic_json, checkpoint_guard


CONVERGENCE = .00171
WAVELENGTH = electron_wavelength_nm(80000)
APERTURE = CONVERGENCE/WAVELENGTH
PILOT = {"DS1", "DS7", "DS10", "DS16", "DS23", "DS25"}
TOLERANCE = .003


def tilted_window_coefficients(motion, cell, q, gradient, width, interior, tolerance=TOLERANCE, quadrature=61):
    """Form the detector terms from two shifted pupils and the periodic phase factors."""
    delta = q@gradient
    axis = np.arange(quadrature)-(quadrature-1)/2
    sweep = 2*APERTURE*axis/(quadrature-1)
    y,x = np.meshgrid(sweep,sweep,indexing="ij")
    raster = np.stack((x,y),axis=-1).reshape(-1,2)
    detector_coordinates = raster[:,None,:] + np.array((-1,1))[None,:,None]*delta/2
    overlap = np.all(np.linalg.norm(detector_coordinates, axis=-1)<=interior*APERTURE, axis=1)
    detector = detector_coordinates[overlap]
    if len(detector)<15:
        raise ValueError("The assumed overlap window has too little support")
    n = len(motion)
    t = np.arange(n)/n
    fractional = np.stack(np.meshgrid(t,t,indexing="ij"),axis=-1).reshape(-1,2)
    inverse = np.linalg.inv(cell)
    planes = []
    for i, sign in enumerate((-1,1)):
        modes, amplitude, tail = bands(motion, sign*q/2, tolerance)
        sideband = modes@inverse
        p = detector[:,i,None,:]-sideband
        transmission = np.sqrt(expit((APERTURE-np.linalg.norm(p,axis=-1))/width))
        phases = np.exp(2j*np.pi*fractional@modes.T)*amplitude
        planes.append((phases,transmission,tail))
    first, second = planes
    def integral(a,b):
        wave_a, pupil_a, _ = a
        wave_b, pupil_b, _ = b
        overlap_matrix = (pupil_a.T @ pupil_b) / len(detector)
        return np.sum((wave_a.conj() @ overlap_matrix) * wave_b,axis=1)

    cross = integral(first,second)
    terms = np.column_stack((integral(first,first).real,integral(second,second).real,cross.real,cross.imag))
    idx = np.arange(0,n*n,max(1,n*n//6))
    check_a,check_b = [wave[idx] @ pupil.T for wave,pupil,_ in planes]
    if np.max(abs(cross[idx]-np.mean(check_a.conj()*check_b,axis=1)))>1e-11:
        raise ValueError("Reciprocal overlap matrix failed its direct integration control")
    if np.min(terms[:,:2])<-1e-11 or np.max(abs(cross)**2-(terms[:,0]*terms[:,1]))>1e-11:
        raise ValueError("The magnitudes violate the coherent overlap bounds")
    return terms.reshape(n,n,4),{"aperture_overlap_fraction_of_grid":float(overlap.mean()),
                                 "sideband_counts": [int(row[1].shape[1]) for row in planes],
                                 "discarded_sideband_amplitude": [float(row[-1]) for row in planes]}


def observer(phase,origin,grid,shift,reciprocal,physical_gradient,probe,width,interior):
    n = len(shift)
    # The scatterer in each layer carries half the periodic relative motion.
    # For rigid layers their separation in reciprocal space is q @ gradient.
    # Include the phase of that separation at the probe scan coordinates.
    signal = []
    gates = []
    center = (phase+origin) @ np.linalg.inv(grid.basis).T
    coordinate = n*np.moveaxis(center,-1,0)
    for q in reciprocal:
        checkpoint_guard()
        response, receipt = tilted_window_coefficients(shift,grid.cell,q,physical_gradient,width,interior)
        if len(response)!=n:
            raise ValueError("The periodic observation mesh changed")
        panel = np.stack([map_coordinates(response[...,i],coordinate,order=3,mode="grid-wrap")
                          for i in range(4)],axis=-1)
        phi = 2*np.pi*(phase@q)
        # Combine the real and imaginary probe cross terms with the relative
        # lattice phase at each scan point. Single-layer intensity adds.
        interference = (panel[...,2]+1j*panel[...,3])*np.exp(-1j*phi)
        contrast = np.stack((.25*np.sum(panel[...,:2],axis=-1)+.5*interference.real,
                             -.5*interference.imag),axis=-1)
        # Source-size blur completes the 1.25 nm FWHM of the incoming probe.
        # 0.514/R is the Airy intensity FWHM for a circular aperture.
        coherent = .514/APERTURE
        sigma = probe*np.sqrt(max(1-(coherent/1.25)**2,0.))
        signal.append(gaussian_filter(contrast,(sigma,sigma,0),mode="nearest"))
        gates.append(receipt)
    array = np.stack(signal,axis=-2)
    return (array[...,0],array[...,1]),gates


def simple_receipt(context, candidate, values):
    rows = []
    for train,test in fold_masks(context["support"], context["probe"]):
        if min(train.sum(),test.sum())<300:
            continue
        row = {}
        for name,(even,odd) in candidate.items():
            prediction, observed, _ = camera_fit(even,odd,values,train,test,allow_odd=True)
            row[name] = float(np.sqrt(np.mean((prediction-observed)**2)))
        rows.append(row)
    return rows


def simulate(context, interface, values):
    rigid,basis,gradient = (context[k] for k in ("rigid","basis","gradient"))
    grid, shift, receipt = solve_from_geometry(gradient,basis,interface)
    origin = basis @ np.full(2,2/3 if interface[-1]=="AP" else 0.)
    reference = sample_disregistry(grid,shift,rigid+origin,grid.cell)-origin
    if len(shift)>96:
        n = 96
        small_grid = make_grid(gradient,basis,n,1.,1.)
        shift = sample_disregistry(grid,shift,small_grid.fractional@basis.T,small_grid.cell)-small_grid.fractional@basis.T
        grid = small_grid
    rho = sample_disregistry(grid,shift,rigid+origin,grid.cell)-origin
    output = {"solver":receipt,"optical_grid":len(shift),"rings":{},
              "resampling_maximum_displacement_error_nm":float(np.max(abs(reference-rho)))}
    for name in ("first","second"):
        selected=context["rings"][name]
        q=context["reciprocal"][selected]
        y=values[...,selected]
        probe=context["probe"]
        fields={"gaussian_relaxed": phase_camera(reference,q,probe),"gaussian_rigid":phase_camera(rigid,q,probe)}
        diagnostics={}
        for aperture in (.85,1.):
            for key,w in (("rigid",np.zeros_like(shift)),("relaxed",shift)):
                model=f"coherent_{key}_{aperture}"
                fields[model],diagnostics[model]=observer(rigid,origin,grid,w,q,gradient,probe,.012,aperture)
        for aperture in (.85,1.):
            a,b=fields[f"coherent_rigid_{aperture}"]
            ref=phase_camera(rigid,q,probe*np.sqrt(max(1-(.514/APERTURE/1.25)**2,0.)))
            for j in range(a.shape[-1]):
                design=np.column_stack((ref[0][...,j].ravel(),ref[1][...,j].ravel(),np.ones(a.shape[0]*a.shape[1])))
                observed=np.column_stack((a[...,j].ravel(),b[...,j].ravel()))
                factors=np.linalg.lstsq(design,observed,rcond=None)[0]
                if np.max(abs(design @ factors-observed))>1e-10:
                    raise ValueError("The coherent observer failed its zero-motion limit")
        output["rings"][name]={"scores":simple_receipt(context,fields,y),"camera":diagnostics}
    return output


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive",type=Path)
    parser.add_argument("release",type=Path)
    parser.add_argument("dependencies",type=Path)
    parser.add_argument("output",type=Path)
    args=parser.parse_args()
    start=perf_counter()
    files=(PROTOCOL,SOURCE,BENCHMARK,SETTINGS)
    protocol,source,benchmark,settings=(json.loads(f.read_text()) for f in files)
    author=load_author_release(args.release,args.dependencies)
    result={"scope":"Pilot weak-phase two-layer probe propagation for the same measured integrated diffraction as the Gaussian observer. Separate the two reciprocal disk centers by the measured relative gradient, sum each layer's continuum sidebands through the stated 80 kV and 1.71 mrad pupil, then integrate the two single-layer intensities and their complex cross term over two assumed centered overlap apertures. The same signed per-channel calibration freedom and spatial holdouts apply to every predictor.",
            "limits":"The exact virtual detector masks, defocus, layer scattering amplitudes and mutual coherence are unverified for these processed maps. The model assumes balanced single-layer scattering and equal partition of relative relaxation. Apertures are geometry-derived sensitivity cases. This is a diagnostic for probe effects and retains the measured moire geometry.",
            "aperture_cycles_per_nm":APERTURE,"beam_wavelength_nm":WAVELENGTH,
            "input_sha256":{f.name:digest(f) for f in (Path(__file__), *files)},"scans":{}}
    for group,scans in protocol["cases"].items():
        for scan in scans:
            if scan not in PILOT:
                continue
            checkpoint_guard()
            record=source["scans"][scan]
            meta=read_input_metadata(scan,args.archive)
            images,modes,checksum=disk_images(args.archive,scan,author)
            if checksum!=record["diskset_sha256"] or meta["archive_md5"]!=benchmark["scans"][scan]["archive_md5"]:
                raise ValueError("The archived diffraction input changed")
            interface=tuple(record["interface"])
            context=centers_and_context(images,modes,meta["metadata_fields"],settings,interface)
            if context["receipt"]!=record["geometry"]:
                raise ValueError("The measured geometry changed")
            result["scans"][scan]={"group":group,**simulate(context,interface,images)}
            result["elapsed_seconds"]=perf_counter()-start
            atomic_json(args.output,result)
            print(scan,{name:[x["gaussian_relaxed"]-x["coherent_relaxed_1.0"] for x in r["scores"]]
                        for name,r in result["scans"][scan]["rings"].items()},flush=True)


if __name__=="__main__":
    main()
