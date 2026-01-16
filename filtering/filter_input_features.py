#!/usr/bin/env python3

'''
    Description: Filters input features generated from DINO EXP16

    Method: Uses gcm-filters GridType.IRREGULAR_WITH_LAND with spatially varying
        length scale with max scale the max grid scale from DINO EXP4 (1/4 deg).
'''

import gcm_filters
import xarray as xr
from xnemogcm import open_nemo_and_domain_cfg, get_metrics
import xgcm
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta
import glob

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(filename='filter_input_features.log', level=logging.DEBUG)

logger.info('Begin...')

# ------------ misc parameters ---------------- #

directory = '/gws/nopw/j04/ai4pex/twilder/NEMO_data/DINO/EXP16/features/'
mask16_path = ['~/Python/AI4PEX/DINO/mesh_mask_exp16.nc']
mask025_path = '/gws/nopw/j04/ai4pex/twilder/NEMO_data/DINO/EXP4/mesh_mask_025.nc'

variable = 'vor'

# open spatial filter
kappa = xr.open_dataset('/gws/nopw/j04/ai4pex/twilder/NEMO_data/DINO/EXP16/kappa16.nc')

# rename dimensions for gcm-filters
kappa = kappa.rename({'y': 'y_c', 'x': 'x_c'})

# get max grid scale from EXP4 (1/4 deg)
mask025 = xr.open_dataset(mask025_path)
grid_scale = np.sqrt(mask025.e1t*mask025.e2t)
max_grid_scale = grid_scale.max()

# -------------------------------------------- #

# Initial date string
start_date_init_str = "00610201"

# End date string
end_date_init_str = "00730101"


# Convert date strings to datetime objects
start_date_init = datetime.strptime(start_date_init_str, "%Y%m%d")
end_date_init = datetime.strptime(end_date_init_str, "%Y%m%d")

# Loop to increment date by one day
current_date_init = start_date_init
while current_date_init < end_date_init:

    # Increment for next date
    next_date_init = current_date_init + relativedelta(months=+1)
    print(next_date_init)

    # Convert dates to string for nemo files
    # date = current_date.strftime("%Y%m%d")
    date_init = (
        f"{current_date_init.year:04d}"\
        f"{current_date_init.month:02d}"\
        f"{current_date_init.day:02d}" # fix for adding leading zeros back in
    )

    print(date_init)

    # Assign current date to next date for next loop
    current_date_init = next_date_init

    # set nemo filename using dates
    nemo_files = [f'MINT_1d_{date_init}_*_grid_{variable}.nc']
    print(nemo_files)

    # open dataset using xnemogcm
    # nemo_paths = [directory + f for f in nemo_files]
    nemo_paths =  [glob.glob(directory + f) for f in nemo_files]
    ds = open_nemo_and_domain_cfg(nemo_files=nemo_paths[0],
                                  domcfg_files=mask16_path)

    # Rechunk the dataset
    ds=ds.chunk(dict(y_c=-1))
    ds=ds.chunk(dict(x_f=-1))
    ds=ds.chunk(dict(y_f=-1))
    ds=ds.chunk(dict(x_c=-1))

    # Need west and south scale factor convention for irregular filter on tripolar grid
    dxw = ds.e1u.roll(x_f=-1, roll_coords=False)  # x-spacing centered at western T-cell edge in m
    dyw = ds.e2u.roll(x_f=-1, roll_coords=False)  # y-spacing centered at western T-cell edge in m
    dxs = ds.e1v.roll(y_f=-1, roll_coords=False)  # x-spacing centered at southern T-cell edge in m
    dys = ds.e2v.roll(y_f=-1, roll_coords=False)  # y-spacing centered at southern T-cell edge in m

    # Ensure that coordinates are the same
    wet_mask = ds.tmask.isel(z_c=0)
    dxw = dxw.swap_dims({"x_f": "x_c"})
    dyw = dyw.swap_dims({"x_f": "x_c"})
    dxs = dxs.swap_dims({"y_f": "y_c"})
    dys = dys.swap_dims({"y_f": "y_c"})
    area = ds.e1t*ds.e2t

    # find minimum grid spacing
    dx_min = min(ds.e1t.where(ds.tmask.isel(z_c=0)).min(), ds.e2t.where(ds.tmask.isel(z_c=0)).min())
    dx_min = dx_min.values

    # set up the filter
    specs = {
            'filter_scale': max_grid_scale.values,
            'dx_min': dx_min,
            'filter_shape': gcm_filters.FilterShape.GAUSSIAN
        }

    filter_irregular_with_land = gcm_filters.Filter(
        **specs,
        grid_type=gcm_filters.GridType.IRREGULAR_WITH_LAND,
        grid_vars={
            'wet_mask': wet_mask, 
            'dxw': dxw, 'dyw': dyw, 'dxs': dxs, 'dys': dys, 'area': area, 
            'kappa_w': kappa.kappa, 'kappa_s': kappa.kappa,
        }
    ) # need to vary kappa using the deformation radius.
    filter_irregular_with_land

    # set up xgcm grid
    grid = xgcm.Grid(
                ds,
                metrics=get_metrics(ds),
                periodic={'X': True, 'Y': False},
                boundary={'Y': 'extend'},
                )
    
    # interpolate if not gridT and filter
    ds_tmp = xr.Dataset()
    if variable in ['vor']:
        ds['vor_c'] = grid.interp(ds.vor, ['X', 'Y'])
        ds_tmp['vor_f'] = filter_irregular_with_land.apply(ds['vor_c'], 
                                                           dims=['y_c', 'x_c'])