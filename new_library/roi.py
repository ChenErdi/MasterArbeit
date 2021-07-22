import numpy as np
import scipy
import sklearn
import pandas as pd
import cv2
from PIL import Image


from .image import(
    mm2px,
    px2mm,
)

def calc_bbox_shape(bbox):
    return [bbox[3] - bbox[1], bbox[2] - bbox[0]]

def calc_bbox_midpoint(bbox):
    return [(bbox[3] + bbox[1])//2, (bbox[2] + bbox[0])//2]

def bbox_to_plot_coor(bbox):
    x1, y1, x2, y2 = bbox
    
    return [x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1]

def find_syringe_xmax(im, syringe_rel_thres):

    im_sum = im.sum(axis=0)

    try:
        px_sum_above_thres = (im_sum/im_sum.max()) < syringe_rel_thres
        syringe_xmax = im.shape[1] - np.argwhere(np.flip(px_sum_above_thres))[0]
        syringe_xmax = syringe_xmax[0]
    except ValueError as e:
        print(e)

    return syringe_xmax

def create_1stripe_pattern(
    true_pattern_height_mm, true_pattern_width_mm, 
    offset_height_mm, offset_width_mm, 
    y_dpi, x_dpi,
    white_val=150,
    black_val=0
):
    
    true_pattern_height = mm2px(true_pattern_height_mm, y_dpi)
    true_pattern_width = mm2px(true_pattern_width_mm, x_dpi)
    
    offset_height = mm2px(offset_height_mm, y_dpi)
    offset_width = mm2px(offset_width_mm, x_dpi)
    
    pattern_height = true_pattern_height + 2*offset_height
    pattern_width = true_pattern_width + 2*offset_width

    pattern = np.full((pattern_height, pattern_width), white_val, dtype=np.uint8)
    
    pattern[offset_height:(offset_height+true_pattern_height), offset_width:(offset_width+true_pattern_width)] = black_val
    
    
    return pattern

def create_3stripes_pattern(
    true_pattern_height_mm, true_pattern_width_mm,
    stripe1_height_mm, gap1_height_mm,
    stripe2_height_mm, gap2_height_mm,
    y_dpi, x_dpi,
    offset_height_mm=None, offset_width_mm=None, 
    white_val=150,
    black_val=0
):
    
    true_pattern_height = mm2px(true_pattern_height_mm, y_dpi)
    true_pattern_width = mm2px(true_pattern_width_mm, x_dpi)
    
    #print([true_pattern_height, true_pattern_width])
    
    pattern = np.full((true_pattern_height, true_pattern_width), black_val, dtype=np.uint8)
    
    # Fill gap1 with "white"
    gap1_ymin_mm = stripe1_height_mm
    gap1_ymax_mm = stripe1_height_mm + gap1_height_mm
    gap1_xmin_mm = 0
    gap1_xmax_mm = true_pattern_width_mm
    
    gap1_xmin = mm2px(gap1_xmin_mm, x_dpi)
    gap1_xmax = mm2px(gap1_xmax_mm, x_dpi)
    gap1_ymin = mm2px(gap1_ymin_mm, y_dpi)
    gap1_ymax = mm2px(gap1_ymax_mm, y_dpi)
    
    #print([gap1_xmin, gap1_xmax, gap1_ymin, gap1_ymax])

    pattern[gap1_ymin:gap1_ymax, gap1_xmin:gap1_xmax] = white_val
    
    # Fill gap2 with "white"
    gap2_ymin_mm = stripe1_height_mm + gap1_height_mm + stripe2_height_mm
    gap2_ymax_mm = stripe1_height_mm + gap1_height_mm + stripe2_height_mm + gap2_height_mm
    gap2_xmin_mm = 0
    gap2_xmax_mm = true_pattern_width_mm
    
    gap2_xmin = mm2px(gap2_xmin_mm, x_dpi)
    gap2_xmax = mm2px(gap2_xmax_mm, x_dpi)
    gap2_ymin = mm2px(gap2_ymin_mm, y_dpi)
    gap2_ymax = mm2px(gap2_ymax_mm, y_dpi)
    
    #print([gap2_xmin, gap2_xmax, gap2_ymin, gap2_ymax])

    pattern[gap2_ymin:gap2_ymax, gap2_xmin:gap2_xmax] = white_val
    
    if offset_height_mm is not None:
        offset_height = mm2px(offset_height_mm, y_dpi)
        offset_pattern = np.full((offset_height, true_pattern_width), white_val, dtype=np.uint8)
    
        pattern = np.concatenate([offset_pattern, pattern, offset_pattern], axis=0)
        
    if offset_width_mm is not None:
        offset_width = mm2px(offset_width_mm, x_dpi)
        pattern_height = pattern.shape[0]
        offset_pattern = np.full((pattern_height, offset_width), white_val, dtype=np.uint8)
        pattern = np.concatenate([offset_pattern, pattern, offset_pattern], axis=1)
    
    return pattern

def find_roi(im, pattern, method, slice_xmin=None, slice_xmax=None, slice_ymin=None, slice_ymax=None):
    if slice_xmin is None:
        slice_xmin = 0
        
    if slice_xmax is None:
        slice_xmax = im.shape[1]
        
    if slice_ymin is None:
        slice_ymin = 0
        
    if slice_ymax is None:
        slice_ymax = im.shape[0]
        
    sliced_im = im[slice_ymin:, slice_xmin:slice_xmax]
    
    res = cv2.matchTemplate(sliced_im, pattern, method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        roi_midpoint = min_loc
    else:
        roi_midpoint = max_loc
        
    roi_offset_y, roi_offset_x = np.array(pattern.shape) // 2
    
    # Correct roi position relative to slice_xmin and pattern shape
    roi_x = roi_midpoint[0] + roi_offset_x + slice_xmin
    roi_y = roi_midpoint[1] + roi_offset_y + slice_ymin
    
    return roi_x, roi_y

def create_bbox(roi_x, roi_y, roi_width, roi_height):
    
    bbox_offset_x = roi_width//2
    bbox_offset_y = roi_height//2

    bbox_xstart = roi_x - bbox_offset_x
    bbox_xend = bbox_xstart + roi_width
    bbox_ystart = roi_y - bbox_offset_y
    bbox_yend = bbox_ystart + roi_height

    bbox = [bbox_xstart, bbox_ystart, bbox_xend, bbox_yend]
    
    return bbox

def slice_roi(im, bbox):
    roi_im = im[bbox[1]:bbox[3], bbox[0]:bbox[2]]
    return roi_im