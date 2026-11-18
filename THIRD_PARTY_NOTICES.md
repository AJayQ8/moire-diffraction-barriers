# Third-party data and software notices

## MoS₂ 4D-STEM data

The paper's measured wall profiles and extracted or cropped diffraction-count
examples are derived from the public data of Madeline Van Winkle, Isaac Craig,
Karen Bustillo, Jim Ciston, and D. Kwabena Bediako, *Rotational and Dilational
Reconstruction in Transition Metal Dichalcogenide Moiré Bilayers* (Zenodo,
2023), https://doi.org/10.5281/zenodo.7865970. The record identifies the data
as CC BY 4.0. The paper dataset is also described in Van Winkle et al.,
*Nature Communications* 14, 2989 (2023),
https://doi.org/10.1038/s41467-023-38504-7.

Files in evidence/ contain processed or derived measurements and model
comparisons, including selected wall profiles, crops, fits, and summaries.
They are adaptations of the cited data; the complete upstream scan archives
are not redistributed here. The corresponding source hashes and processing
settings are retained in the result records.

## Graphene integrated-intensity arrays

The files under replay/graphene/ are derived integrated-intensity arrays and
receipts from the public datasets of Nathanael P. Kazmierczak, Madeline Van
Winkle, Colin Ophus, Karen C. Bustillo, Hamish G. Brown, Stephen Carr, Jim
Ciston, Takashi Taniguchi, Kenji Watanabe, and D. Kwabena Bediako, *Strain
fields in twisted bilayer graphene*, *Nature Materials* 20, 956–963 (2021),
https://doi.org/10.1038/s41563-021-00973-w. Upstream data records:
dataset 2, https://doi.org/10.5281/zenodo.4459674, and dataset 19,
https://doi.org/10.5281/zenodo.4463612. Both Zenodo records identify the
data as CC BY 4.0. The bundled arrays are processed excerpts, not the complete
source scans; adjacent receipt.json files identify source URLs, selections,
and hashes.

## External analysis software

The replay code can load the pinned author release of pyInterferometry:
https://doi.org/10.5281/zenodo.7863859, source commit
85f21dd9134b600000e517ad1b22acc725571376. That software is not included in
this release. Obtain it from its source record and follow its own license and
attribution terms. This release does not make or imply a license claim over
pyInterferometry or any other dependency.

## Attribution and changes

The derived profiles and array excerpts above preserve upstream attribution
and source links. They have been cropped, integrated, fit, normalized, and/or
summarized for the analyses described by the replay settings and evidence
records. The complete upstream datasets remain available from the cited
records under their stated terms.
