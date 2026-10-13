"""Shared Figure 3 palette (imported by render3d.py and figure3_editorial.py)."""
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# Thresholded beta display for 3-D maps: only q < 0.05 regions are coloured, so each arm starts
# from a visible tint at |beta| -> 0 and is linear in |beta| up to VMAX (hue family of Figure 5c).
BLUE_ARM = LinearSegmentedColormap.from_list('lower', ['#BBD5EA', '#7FAED0', '#3F7DB3', '#245E8C', '#163E62'])
RED_ARM = LinearSegmentedColormap.from_list('higher', ['#F4CBBE', '#E39A86', '#C9624D', '#AE4636', '#7A2A20'])
VMAX = 0.5
NS_SUB = '#D3D8DC'          # non-significant subcortical structure
NS_CTX = '#DADDE0'          # representative non-significant cortex grey (sulcal shading in renders)
LOWER, HIGHER = '#3F7DB3', '#C9624D'   # discrete direction colours (panel b)
SC = ['#377EB8', '#D8A429', '#C44E52']  # Stage 1-3, identical to Figures 1-6
SHAPES = ['s', 'D', 'o']
INK, MUTED, AXIS, BAND = '#24394A', '#647B88', '#9FB0BA', '#F3F6F8'


def beta_rgb(beta):
    t = min(abs(float(beta)), VMAX) / VMAX
    return np.array((BLUE_ARM if beta < 0 else RED_ARM)(t)[:3])
