# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import numpy as np

NAME_TO_VAR = {
    "latitude": "lat2d",
    "longitude": "lon2d",
    "geopotential_500": "z",
    "temperature_850": "t",
}

VAR_TO_NAME = {v: k for k, v in NAME_TO_VAR.items()}


PRESSURE_LEVEL_VARS = [
    "geopotential_500",
    "temperature_850",
]

NAME_LEVEL_TO_VAR_LEVEL = {}

for var in PRESSURE_LEVEL_VARS:
    NAME_LEVEL_TO_VAR_LEVEL[var] = NAME_TO_VAR[var]

VAR_LEVEL_TO_NAME_LEVEL = {v: k for k, v in NAME_LEVEL_TO_VAR_LEVEL.items()}

BOUNDARIES = {
    'NorthAmerica': { # 8x14
        'lat_range': (15, 65),
        'lon_range': (220, 300)
    },
    'SouthAmerica': { # 14x10
        'lat_range': (-55, 20),
        'lon_range': (270, 330)
    },
    'Europe': { # 6x8
        'lat_range': (30, 65),
        'lon_range': (0, 40)
    },
    'SouthAsia': { # 10, 14
        'lat_range': (-15, 45),
        'lon_range': (25, 110)
    },
    'EastAsia': { # 10, 12
        'lat_range': (5, 65),
        'lon_range': (70, 150)
    },
    'Australia': { # 10x14
        'lat_range': (-50, 10),
        'lon_range': (100, 180)
    },
    'Global': { # 32, 64
        'lat_range': (-90, 90),
        'lon_range': (0, 360)
    }
}

def get_region_info(region, lat, lon, patch_size):
    region = BOUNDARIES[region]
    lat_range = region['lat_range']
    lon_range = region['lon_range']
    lat = lat[::-1] # -90 to 90 from south (bottom) to north (top)
    h, w = len(lat), len(lon)
    lat_matrix = np.expand_dims(lat, axis=1).repeat(w, axis=1)
    lon_matrix = np.expand_dims(lon, axis=0).repeat(h, axis=0)
    valid_cells = (lat_matrix >= lat_range[0]) & (lat_matrix <= lat_range[1]) & (lon_matrix >= lon_range[0]) & (lon_matrix <= lon_range[1])
    h_ids, w_ids = np.nonzero(valid_cells)
    h_from, h_to = h_ids[0], h_ids[-1]
    w_from, w_to = w_ids[0], w_ids[-1]
    patch_idx = -1
    p = patch_size
    valid_patch_ids = []
    min_h, max_h = 1e5, -1e5
    min_w, max_w = 1e5, -1e5
    for i in range(0, h, p):
        for j in range(0, w, p):
            patch_idx += 1
            if (i >= h_from) & (i + p - 1 <= h_to) & (j >= w_from) & (j + p - 1 <= w_to):
                valid_patch_ids.append(patch_idx)
                min_h = min(min_h, i)
                max_h = max(max_h, i + p - 1)
                min_w = min(min_w, j)
                max_w = max(max_w, j + p - 1)
    return {
        'patch_ids': valid_patch_ids,
        'min_h': min_h,
        'max_h': max_h,
        'min_w': min_w,
        'max_w': max_w
    }