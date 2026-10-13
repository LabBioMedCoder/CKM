#!/usr/bin/env python3
"""Download the public reference atlases used for the Figure 3 brain renders into ./atlas.
fsaverage 164k pial / sulc / Desikan2006 labels from TemplateFlow; Harvard-Oxford subcortical
(25% max-probability, 1 mm) atlas and the MNI152 1 mm brain mask from nilearn.
Usage (from the Figure3_Editorial_Redesign directory): python3 code/get_atlas.py
"""
import shutil
from pathlib import Path
import nibabel as nib
import pandas as pd
from templateflow import api as tf
from nilearn import datasets

A = Path(__file__).resolve().parents[1] / 'atlas'
A.mkdir(exist_ok=True)
for h in 'LR':
    for kw, name in [(dict(density='164k', suffix='pial', extension='.surf.gii'), f'{h}_pial_164k.surf.gii'),
                     (dict(density='164k', suffix='sulc', extension='.shape.gii'), f'{h}_sulc_164k.shape.gii'),
                     (dict(density='164k', atlas='Desikan2006', desc='original', suffix='dseg', extension='.label.gii'),
                      f'{h}_DK_164k_dseg.label.gii')]:
        p = tf.get('fsaverage', hemi=h, **kw)
        p = p[0] if isinstance(p, list) else p
        shutil.copy(p, A / name)
        print('saved', name)
ho = datasets.fetch_atlas_harvard_oxford('sub-maxprob-thr25-1mm')
img = ho.maps if hasattr(ho.maps, 'get_fdata') else nib.load(ho.maps)
nib.save(img, A / 'HarvardOxford_sub_maxprob_thr25_1mm.nii.gz')
pd.Series(ho.labels).to_csv(A / 'HarvardOxford_sub_labels.csv', header=['label'])
nib.save(datasets.load_mni152_brain_mask(resolution=1), A / 'MNI152_brain_mask_1mm.nii.gz')
print('atlas files ready in', A)
