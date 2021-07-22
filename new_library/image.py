import os
import datetime as dt

import numpy as np
import pandas as pd

MM_TO_INCH  = 0.0393701

def mm2px(val_px, dpi, toint=True):
    val_mm = val_px * MM_TO_INCH * dpi
    
    if toint:
        return int(val_mm)
    else:
        return val_mm

def px2mm(val_mm, dpi):
    mm_to_inch = 0.0393701
    val_px = val_mm / (MM_TO_INCH * dpi)
    
    return val_px

def random_translation(
    im:np.ndarray,
):
    x_pos = np.random.randint(im.shape[0])
    y_pos = np.random.randint(im.shape[1])

    con_x = np.concatenate((im, im), axis=0)
    con_xy = np.concatenate((con_x, con_x), axis=1)

    im = con_xy[x_pos:x_pos+im.shape[0], y_pos:y_pos+im.shape[1]]
    
    return im
    

def check_image(
    im_fpath:str, 
    last_update:str, 
    offset:pd.Timedelta=None,
    threshold:float=10.0,
    check_based_on_timestamp=True
):
    """
    Check correctness of an image by computing image metadata to LastUpdateTimeStamp
    
    Parameter:
    ----------
    im_fpath : str
        Absolute path to image
    last_update : str, Pandas.Datetime
        Value from LastUpdateTimeStamp
    offset : None, Pandas.Timedelta
        Offset that happen due to different timezone (happens when using docker jupyternotebook)
    threshold : float
        Maximum absolute time delta in seconds between image last update and LastUpdateTimeStamp allowed
        
    Returns:
    --------
    ret : bool
        Correctness of image based on metadata
    """
    if offset is None:
        offset = pd.Timedelta(0)

    if os.path.exists(im_fpath):
        if check_based_on_timestamp:
            im_fstat = os.stat(im_fpath)
            im_last_updated = dt.datetime.fromtimestamp(im_fstat.st_mtime)
            abs_timedelta = np.absolute(
                (pd.to_datetime(last_update) - im_last_updated - offset).total_seconds()
            )
            if abs_timedelta < threshold:
                return True
        else:
            return True
    return False

def get_timecreated(im_fpath):
    try:
        im_fstat = os.stat(im_fpath)
        im_last_updated = dt.datetime.fromtimestamp(im_fstat.st_mtime)
        return im_last_updated

    except:
        return None