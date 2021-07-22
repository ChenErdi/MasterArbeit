from __future__ import print_function
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets

import os
#import re
#import glob
#import datetime as dt

import numpy as np
import scipy
import sklearn
import pandas as pd
import cv2
import PIL
from PIL import Image

import matplotlib.pyplot as plt

from .image import(
    mm2px,
    px2mm,
)

from .util import (
    jupyter_wide_screen,
)

from .roi import (
    find_syringe_xmax,
    create_1stripe_pattern,
    create_3stripes_pattern,
    find_roi,
    create_bbox,
    bbox_to_plot_coor,
    slice_roi
)

def create_1stripe_pattern_with_config(
    y_dpi:int,
    x_dpi:int,
    true_pattern_height_mm=None, 
    true_pattern_width_mm=None, 
    offset_height_mm=0, 
    offset_width_mm=0,
    white_val=150,
    black_val=0
):
    assert true_pattern_height_mm is not None
    assert true_pattern_width_mm is not None
    
    return create_1stripe_pattern(
        true_pattern_height_mm, true_pattern_width_mm, 
        offset_height_mm, offset_width_mm, 
        y_dpi, x_dpi,
        white_val=white_val,
        black_val=black_val
    )

def create_3stripes_pattern_with_config(
    y_dpi:int,
    x_dpi:int,
    pattern_height_mm=None, pattern_width_mm=None,
    stripe1_height_mm=None, gap1_height_mm=None,
    stripe2_height_mm=None, gap2_height_mm=None,
    offset_height_mm=0, offset_width_mm=0,
    white_val=150,
    black_val=0
):
    assert pattern_height_mm is not None
    assert pattern_width_mm is not None
    assert stripe1_height_mm is not None
    assert gap1_height_mm is not None
    assert stripe2_height_mm is not None
    assert gap2_height_mm is not None
    
    return create_3stripes_pattern(
        pattern_height_mm, pattern_width_mm,
        stripe1_height_mm, gap1_height_mm,
        stripe2_height_mm, gap2_height_mm,
        y_dpi, x_dpi,
        offset_height_mm=offset_height_mm, offset_width_mm=offset_width_mm,
        white_val=white_val,
        black_val=black_val
    )

def slice_roi(
    # From database
    im_pil:PIL.Image,
    druckpos_mm=None,
    # Using config
    druckpos_threshold=0.6,
    druckbild_search_width=None,
    pattern_name=None,
    pattern_config=None,
    roi_height=720,
    roi_width=30,
    slice_ymin='auto',
    slice_ymax='auto',
    pm_method=cv2.TM_CCOEFF,
    # For debug purpose only
    debug_pattern=False,
    debug_image=False,
    debug_roi=False
):
    
    x_dpi, y_dpi = im_pil.info['dpi']
    im = np.array(im_pil)

    if druckpos_mm is not None:
        druckpos = mm2px(druckpos_mm, x_dpi)
        syringe_xmax = find_syringe_xmax(im, druckpos_threshold)

        slice_xmin = syringe_xmax-druckpos
        
        if druckbild_search_width is None:
            slice_xmax=None
        else:
            slice_xmax = slice_xmin+druckbild_search_width
    else:
        syringe_xmax = None
        slice_xmin = None
        slice_xmax = None
    
    if pattern_name.lower() == '1stripe':
        p = create_1stripe_pattern_with_config(y_dpi, x_dpi, **pattern_config)
    elif pattern_name.lower() == '3stripes':
        p = create_3stripes_pattern_with_config(y_dpi, x_dpi, **pattern_config)
        
    #print(p.shape)
    if slice_ymin == 'auto':
        slice_ymin = (roi_height-p.shape[0]) // 2 + 1
        #print(slice_ymin)
    if slice_ymax == 'auto':
        slice_ymax = im.shape[0]-slice_ymin
        #print(slice_ymax)
    
    if debug_pattern:
        plt.imshow(p.T, cmap='gray')
        plt.title('Transposed Pattern')
        plt.show()
        
    if debug_image:
        plt.imshow(im, cmap='gray')
        plt.title('Raw Image')
        plt.show()
    
    roi_x, roi_y = find_roi(im, p, 
                            pm_method, 
                            slice_xmin=slice_xmin, slice_xmax=slice_xmax, 
                            slice_ymin=slice_ymin, slice_ymax=slice_ymax,
                           )
    
    bbox = create_bbox(roi_x, roi_y, roi_width, roi_height)
    
    if debug_roi:
        plt.imshow(im, cmap='gray')
        if syringe_xmax is not None:
            plt.plot([syringe_xmax, syringe_xmax], [0, im.shape[0]], 'r', label='syringe_xmax')
            
        if slice_xmin is not None:
            plt.plot([slice_xmin, slice_xmin], [0, im.shape[0]], 'b', label='Druckpos')
            
        if slice_xmax is not None:
            plt.plot([slice_xmax, slice_xmax], [0, im.shape[0]], 'g', label='Max Druckpos')
            
        plt.plot(roi_x, roi_y, '*y', label='ROI midpoint')
        plt.plot(*bbox_to_plot_coor(bbox), 'orange', label='ROI')
        plt.legend()
        plt.show()
           
    xmin, ymin, xmax, ymax = bbox
    
    roi_im = im[bbox[1]:bbox[3], bbox[0]:bbox[2]]
    
    top_padding = np.max([0, -(ymin)])
    bottom_padding = np.max([0, ymax-im.shape[0]])
    roi_im = cv2.copyMakeBorder(roi_im, top_padding, bottom_padding,0,0,cv2.BORDER_REFLECT)
    
    return roi_im
