"""Summary
"""
import rasterio
import numpy as np
from random import randint, seed
from rasterio.windows import Window


def read_full_properties(raster):
    """Create dictionary of raster properties.

    Args:
        raster (str): path to geotiff raster

    Returns:
        d: dict of raster properties
    """
    src = rasterio.open(raster)
    d = dict()
    d['width'] = src.meta['width']
    d['height'] = src.meta['height']
    d['y_bound'] = src.meta['transform'][5]
    d['x_bound'] = src.meta['transform'][2]
    d['pixel_size'] = src.transform[0]
    d['crs'] = src.crs
    return d


def get_window_max_dims(d):
    """Generate upper bounds for subsample window sizes.

        The bounds are based on the width to height ratio of the raster.
        The assumption is that users would like the subsample windows to
        reflect the shape of the full raster. For example, if the raster
        swath is "tall and skinny" the subsample windows will not be short
        and wide. 
    Args:
        d (dict): dict of raster properties

    Returns:
        d: updated dict of raster and window properties
    """
    if d['width'] >= d['height']:
        ratio = d['width'] / d['height']
        pad_factor = 1 / ratio
        d['h_limit'] = int(d['height'] * (1 - pad_factor))
        d['w_limit'] = int(d['width'] * (pad_factor))
    else:
        ratio = d['height'] / d['width']
        pad_factor = 1 / ratio
        d['h_limit'] = int(d['height'] * (pad_factor))
        d['w_limit'] = int(d['width'] * (1 - pad_factor))
    return d


def edge_buffer(dim_pct=0.10):
    """Buffer to prevent spawning subsample windows at raster edges.

    Args:
        dim_pct (float, optional): buffer amount

    Returns:
        d: updated dict of raster and window properties
    """
    d['y_buffer'] = d['height'] * dim_pct
    d['x_buffer'] = d['width'] * dim_pct
    return d


def seed_windows(d, n_windows, min_width=100, min_height=100):
    """Generate random subsample windows.
    Forgive the abuse of list comprehensions.


    Args:
        d (dict): dict of raster properties
        n_windows (int): number of windows to generate
        min_width (int, optional): minimum width of window in pixels
        min_height (int, optional): minimum height of window in pixels

    Returns:
        aois: list of subsample window tuples
    """
    x_lower = d['x_buffer']
    x_upper = d['width'] - d['x_buffer']
    y_lower = d['y_buffer']
    y_upper = d['height'] - d['y_buffer']
    x_seeds = [randint(x_lower, x_upper) for i in range(0, n_windows)]
    y_seeds = [randint(y_lower, y_upper) for i in range(0, n_windows)]
    widths = [randint(min_width, d['w_limit']) for i in range(0, n_windows)]
    heights = [randint(min_height, d['h_limit']) for i in range(0, n_windows)]
    x_ends = [sum(x) for x in zip(x_seeds, widths)]
    y_ends = [sum(y) for y in zip(y_seeds, heights)]
    # Windows are tuples: ((row_start, row_stop),(col_start, col_stop)))
    aois = list(zip(zip(y_seeds, y_ends), zip(x_seeds, x_ends)))
    print(len(aois), ' subsample windows were generated.')
    return aois


def cull_impossible_windows(aois, d):
    """Filter subsample windows beyond the data extent.

    Args:
        aois (list): list of subsample window tuples
        d (dict): dict of raster properties

    Returns:
        aois (list): filtered list of subsample window tuples
    """
    aois = [x for x in aois if x[0][1] <=
            d['height'] and x[1][1] <= d['width']]
    print(len(aois), ' subsample windows remain.')
    return aois


def cull_tall_skinny_windows(aois, d):
    """Filter out tall and skinny windows.

     Args:
        aois (list): list of subsample window tuples
        d (dict): dict of raster properties

    Returns:
        aois (list): filtered list of subsample window tuples
    """
    aois = [x for x in aois if (
        (x[0][1] - x[0][0]) / (x[1][1] - x[1][0])) <= 2]
    print(len(aois), ' random areas-of-interest remain.')
    return aois


def cull_short_fat_windows(aois, d):
    """Filter out short and fat windows.

    Args:
        aois (TYPE): Description
        d (TYPE): Description

    Returns:
        TYPE: Description
    """
    aois = [x for x in aois if (
        (x[1][1] - x[1][0]) / (x[0][1] - x[0][0])) <= 2]
    print(len(aois), ' random areas-of-interest remain.')
    return aois


def get_spatial_window_corner_coords(aois, d):
    """Summary

    Args:
        aois (TYPE): Description
        d (TYPE): Description

    Returns:
        TYPE: Description
    """
    # format is row start stop
    tops = [x[0][0] for x in aois]
    bottoms = [x[0][1] for x in aois]
    lefts = [x[1][0] for x in aois]
    rights = [x[1][1] for x in aois]

    # in arr coords
    # top is row start
    # bot is row stop
    # l is col start
    # r is coll stop

    tls = [(t, l) for t, l in zip(tops, lefts)]
    trs = [(t, r) for t, r in zip(tops, rights)]
    brs = [(b, r) for b, r in zip(bottoms, rights)]
    bls = [(b, l) for b, l in zip(bottoms, lefts)]

    # ok so format for geo corners is intended for src sample
    # src sample takes pairs of xy coordinates
    c_tls = [(d['x_bound'] + tl[1], d['y_bound'] - tl[0]) for tl in tls]
    c_trs = [(d['x_bound'] + tr[1], d['y_bound'] - tr[0]) for tr in trs]
    c_brs = [(d['x_bound'] + br[1], d['y_bound'] - br[0]) for br in brs]
    c_bls = [(d['x_bound'] + bl[1], d['y_bound'] - bl[0]) for bl in bls]

    geo_corners = [[i, j, k, l] for i, j, k, l in zip(c_tls, c_trs,
                                                      c_brs, c_bls)]
    return geo_corners


def sample_for_no_data(aois, geo_corners, raster):
    """Summary

    Args:
        aois (TYPE): Description
        geo_corners (TYPE): Description
        raster (TYPE): Description

    Returns:
        TYPE: Description
    """
    src = rasterio.open(raster)
    nodata = src.meta['nodata']
    aois_valid_corners = []
    # print(nodata)
    for i, j in zip(aois, geo_corners):
        samples = [x[0] for x in src.sample(j)]
        samples = [np.nan if x == nodata else x for x in samples]
        # print(samples)
        nodata_count = sum(np.isnan(samples))
        if nodata_count == 0:
            aois_valid_corners.append(i)
        else:
            pass
    print(len(aois_valid_corners), 'random areas-of-interest remain.')
    return aois_valid_corners

# def perform_windowed_read(raster, windows):


if __name__ == '__main__':
    seed(42)
    raster = './example_data/hv_depth_098_2015_corrected_0.18_m.tif'
    d = read_full_properties(raster)
    d = get_window_max_dims(d)
    d = edge_buffer()
    aois = seed_windows(d, 500)
    print("Filtering windows...")
    aois = cull_impossible_windows(aois, d)
    aois = cull_tall_skinny_windows(aois, d)
    aois = cull_short_fat_windows(aois, d)
    corners = get_spatial_window_corner_coords(aois, d)
    valid = sample_for_no_data(aois, corners, raster)
    print(valid)
