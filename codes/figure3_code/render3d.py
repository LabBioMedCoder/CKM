#!/usr/bin/env python3
"""High-resolution 3-D reference-atlas renders for Figure 3c (cortex) and 3d (subcortex).

All colours are taken from the M1 primary model in the source workbook:
  * cortex: 66 Desikan-Killiany cortical-volume IDPs (33 per hemisphere), q = p_fdr_idp159
  * subcortex: 14 FIRST nuclei + 2 lateral ventricles, q = p_fdr_idp159
Parcels/structures with q >= 0.05 are rendered in neutral grey (cortex: sulcal-depth shading).
Renders are reference-atlas projections of region-level coefficients, not participant MRI.

Usage: xvfb-run -a python3 code/render3d.py
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import nibabel as nib
import pyvista as pv
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm, to_rgb
from scipy.ndimage import gaussian_filter
from skimage.measure import marching_cubes
from PIL import Image

import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent))
ROOT = Path(__file__).resolve().parents[1]
ATL = ROOT / 'atlas'
OUT = ROOT / 'renders'
OUT.mkdir(exist_ok=True)
SRC = Path('/home/ubuntu/upload/CKM脑区关联分析_完整结果.xlsx')

from f3style import BLUE_ARM, RED_ARM, VMAX, NS_SUB as _NS_SUB, beta_rgb
CONTRASTS = {1: 'Stage1 vs Stage0', 2: 'Stage2 vs Stage0', 3: 'Stage3 vs Stage0'}
NS_SUB = np.array(to_rgb(_NS_SUB))
pv.OFF_SCREEN = True


def load_results():
    x = pd.read_excel(SRC, sheet_name='完整分类结果')
    x = x[x.model.eq('M1')].copy()
    assert x.n.eq(35508).all()
    return x


# ----------------------------------------------------------------------------- cortex
def cortex_mesh(code):
    g = nib.load(ATL / f'{code}_pial_164k.surf.gii')
    v = np.asarray(g.darrays[0].data, float)
    f = np.asarray(g.darrays[1].data, np.int64)
    seg = nib.load(ATL / f'{code}_DK_164k_dseg.label.gii')
    lab = np.asarray(seg.darrays[0].data, np.int64)
    names = seg.labeltable.get_labels_as_dict()
    sulc = np.asarray(nib.load(ATL / f'{code}_sulc_164k.shape.gii').darrays[0].data, float)
    # boundary vertices: any edge to a vertex with a different label
    e = np.vstack([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
    diff = lab[e[:, 0]] != lab[e[:, 1]]
    bnd = np.zeros(len(v), bool)
    bnd[e[diff, 0]] = True
    bnd[e[diff, 1]] = True
    faces = np.hstack([np.full((len(f), 1), 3), f]).ravel()
    return pv.PolyData(v, faces), lab, names, sulc, bnd


def cortex_colours(res, hemi, stage, lab, names, sulc, bnd):
    d = res[(res.phenotype_family.eq('Cortical volume (Desikan-Killiany)')) & res.hemisphere.eq(hemi)
            & res.contrast.eq(CONTRASTS[stage])].set_index('region')
    assert len(d) == 33
    assert not (set(d.index) - set(names.values()))
    # neutral sulcal-depth shading (gyri light, sulci darker)
    s = np.tanh(sulc / 4.0)
    grey = 0.915 - 0.125 * (s + 1) / 2
    col = np.repeat(grey[:, None], 3, axis=1) * np.array([0.985, 1.0, 1.012])[None, :]
    rows = []
    for key, name in names.items():
        if name not in d.index:
            continue
        b, q = float(d.loc[name, 'beta_std']), float(d.loc[name, 'p_fdr_idp159'])
        rows.append(dict(hemisphere=hemi, stage=stage, region=name, beta=b, q=q, coloured=q < .05))
        if q < .05:
            m = lab == key
            base = beta_rgb(b)
            # keep hue; add a faint sulcal modulation (<= 6%) for depth
            col[m] = base[None, :] * (1.0 - 0.06 * ((s[m] + 1) / 2))[:, None]
    # thin white parcel outlines
    col[bnd] = col[bnd] * 0.25 + 0.75 * np.array([1, 1, 1])
    return np.clip(col, 0, 1), rows


def render_cortex(res):
    views = {('left', 'lateral'): (-1, 0, 0), ('left', 'medial'): (1, 0, 0),
             ('right', 'medial'): (-1, 0, 0), ('right', 'lateral'): (1, 0, 0)}
    meshes = {h: cortex_mesh('L' if h == 'left' else 'R') for h in ('left', 'right')}
    allrows, files = [], []
    for stage in (1, 2, 3):
        for (hemi, view), vec in views.items():
            mesh, lab, names, sulc, bnd = meshes[hemi]
            col, rows = cortex_colours(res, hemi, stage, lab, names, sulc, bnd)
            if view == 'lateral':
                allrows += rows
            m = mesh.copy()
            m.point_data['rgb'] = (col * 255).astype(np.uint8)
            pl = pv.Plotter(off_screen=True, window_size=(1800, 1240))
            pl.set_background('white')
            pl.add_mesh(m, scalars='rgb', rgb=True, smooth_shading=True, ambient=0.32, diffuse=0.72,
                        specular=0.10, specular_power=18)
            c = np.asarray(mesh.center)
            pl.camera.focal_point = c
            pl.camera.position = c + 600 * np.asarray(vec, float)
            pl.camera.up = (0, 0, 1)
            pl.enable_parallel_projection()
            pl.camera.parallel_scale = 66
            fn = OUT / f'cortex_{hemi}_{view}_stage{stage}.png'
            pl.screenshot(str(fn), transparent_background=True)
            pl.close()
            files.append(fn)
    crop_common(files)
    pd.DataFrame(allrows).to_csv(OUT / 'cortex_render_values.csv', index=False)
    return files


# ----------------------------------------------------------------------------- subcortex
SUB = [  # (HO label index L, R), workbook region, display name
    ((4, 15), 'thalamus', 'Thalamus'), ((5, 16), 'caudate', 'Caudate'), ((6, 17), 'putamen', 'Putamen'),
    ((7, 18), 'pallidum', 'Pallidum'), ((9, 19), 'hippocampus', 'Hippocampus'), ((10, 20), 'amygdala', 'Amygdala'),
    ((11, 21), 'accumbens', 'Accumbens'), ((3, 14), 'Lateral-Ventricle', 'Lateral ventricle')]


def iso_mesh(mask, affine, sigma=0.9, level=0.5, smooth=40, step=1):
    vol = gaussian_filter(mask.astype(float), sigma)
    vv, ff, _, _ = marching_cubes(vol, level=level, step_size=step)
    vv = nib.affines.apply_affine(affine, vv)
    faces = np.hstack([np.full((len(ff), 1), 3), ff]).ravel()
    m = pv.PolyData(vv, faces).clean().triangulate()
    return m.smooth_taubin(n_iter=smooth, pass_band=0.05)


def build_subcortex():
    img = nib.load(ATL / 'HarvardOxford_sub_maxprob_thr25_1mm.nii.gz')
    a = np.asarray(img.dataobj).astype(int)
    out = {}
    for (li, ri), reg, disp in SUB:
        for hemi, idx in (('left', li), ('right', ri)):
            out[(reg, hemi)] = iso_mesh(a == idx, img.affine, sigma=0.8)
    mimg = nib.load(ATL / 'MNI152_brain_mask_1mm.nii.gz')
    glass = iso_mesh(np.asarray(mimg.dataobj) > 0.5, mimg.affine, sigma=2.0, smooth=60, step=2)
    return out, glass


# Inferior view (camera below, slightly posterior; anterior = image top). All eight structure
# classes, including pallidum and accumbens, are visible. Radiological left-right display:
# the participant's right hemisphere appears on the image left.
SUB_CAMERA = dict(dir=(0, -0.22, -1), up=(0, 1, -0.22), scale=46, focal=(0, -8, 0))


def sub_colours(res, stage):
    fam = ['Subcortical volume (FIRST)', 'Ventricle/CSF/cerebellar volume (aseg)']
    d = res[res.phenotype_family.isin(fam) & res.contrast.eq(CONTRASTS[stage])]
    vals = {}
    for _, reg, _ in SUB:
        for hemi in ('left', 'right'):
            r = d[d.region.eq(reg) & d.hemisphere.eq(hemi)]
            assert len(r) == 1, (reg, hemi)
            r = r.iloc[0]
            vals[(reg, hemi)] = (float(r.beta_std), float(r.p_fdr_idp159), r.outcome)
    return vals


def sub_camera(pl):
    f = np.asarray(SUB_CAMERA['focal'], float)
    dvec = np.asarray(SUB_CAMERA['dir'], float)
    dvec /= np.linalg.norm(dvec)
    pl.camera.focal_point = f
    pl.camera.position = f + 500 * dvec
    pl.camera.up = SUB_CAMERA['up']
    pl.enable_parallel_projection()
    pl.camera.parallel_scale = SUB_CAMERA['scale']


def render_subcortex(res):
    from scipy.ndimage import distance_transform_edt, label as cc_label
    meshes, _ = build_subcortex()
    files, rows = [], []
    W, H = 1500, 1500
    keys = list(meshes)
    for stage in (0, 1, 2, 3):
        pl = pv.Plotter(off_screen=True, window_size=(W, H))
        pl.set_background('white')
        vals = None if stage == 0 else sub_colours(res, stage)
        for k, m in meshes.items():
            if vals is None:
                colr = np.array(to_rgb('#CBD2D8'))
            else:
                b, q, oid = vals[k]
                colr = beta_rgb(b) if q < .05 else NS_SUB
                rows.append(dict(stage=stage, outcome=oid, region=k[0], hemisphere=k[1], beta=b, q=q, coloured=q < .05))
            pl.add_mesh(m, color=colr, smooth_shading=True, ambient=0.30, diffuse=0.75, specular=0.12, specular_power=16,
                        silhouette=dict(color='white', line_width=5.0))
        sub_camera(pl)
        fn = OUT / (f'subcortex_stage{stage}.png' if stage else 'subcortex_key.png')
        pl.screenshot(str(fn), transparent_background=True)
        pl.close()
        files.append(fn)
    # ID pass: flat unique colours -> visible-pixel anchors for anatomical labels
    pl = pv.Plotter(off_screen=True, window_size=(W, H))
    pl.set_background('black')
    ids = {}
    for i, (k, m) in enumerate(meshes.items(), start=1):
        rgb = ((i * 13) % 256, (i * 29) % 256, (i * 47) % 256)
        ids[k] = rgb
        pl.add_mesh(m, color=np.array(rgb) / 255, lighting=False)
    sub_camera(pl)
    idimg = pl.screenshot(None, return_img=True)
    pl.close()
    box = crop_common(files)
    x0, y0, x1, y1 = box
    anchors = {}
    for k, rgb in ids.items():
        mask = np.all(np.abs(idimg[..., :3].astype(int) - np.array(rgb)) <= 1, axis=-1)
        if mask.sum() < 30:
            continue
        lab, n = cc_label(mask)
        big = lab == (np.bincount(lab.ravel())[1:].argmax() + 1)
        dt = distance_transform_edt(big)
        yy, xx = np.unravel_index(dt.argmax(), dt.shape)
        anchors[f'{k[0]}|{k[1]}'] = dict(x=(xx - x0) / (x1 - x0), y=(yy - y0) / (y1 - y0), px=int(mask.sum()))
    json.dump(anchors, open(OUT / 'subcortex_anchors.json', 'w'), indent=1)
    pd.DataFrame(rows).to_csv(OUT / 'subcortex_render_values.csv', index=False)
    return files


def crop_common(files, pad=8):
    ims = [np.asarray(Image.open(f)) for f in files]
    alpha = np.maximum.reduce([im[..., 3] for im in ims])
    ys, xs = np.where(alpha > 8)
    x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad, alpha.shape[1])
    y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad, alpha.shape[0])
    for f, im in zip(files, ims):
        Image.fromarray(im[y0:y1, x0:x1]).save(f)
    return x0, y0, x1, y1


if __name__ == '__main__':
    import sys
    res = load_results()
    what = sys.argv[1] if len(sys.argv) > 1 else 'all'
    if what in ('all', 'cortex'):
        render_cortex(res)
        print('cortex done')
    if what in ('all', 'sub'):
        render_subcortex(res)
        print('subcortex done')
