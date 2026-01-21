#!/usr/bin/env python3

'''
    Description: Filters input features generated from DINO EXP16

    Method: Uses gcm-filters GridType.IRREGULAR_WITH_LAND with spatially varying
        length scale with max scale the max grid scale from DINO EXP4 (1/4 deg).
'''

import gcm_filters
import xarray as xr
from xnemogcm import open_domain_cfg, get_metrics
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

directory = '/gws/nopw/j04/ai4pex/twilder/NEMO_data/DINO/EXP16/features/SO_JET/'
mask16_path = [directory + 'mesh_mask_exp16_SO_JET.nc']
mask025_path = directory + 'mesh_mask_exp4_SO_JET.nc'

region = 'SO_JET'

variable = 'vg'
variable_to_filter = 'fine_ke'
variable_name = 'vg'

#TODO add xnemo option to load in deformation radius

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
start_date_init_str = "00610101"

# End date string
end_date_init_str = "00610201"


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
    nemo_files = [f'MINT_1d_{date_init}_*_{variable}_{region}.nc']
    print(nemo_files)

    # open dataset using xnemogcm
    # nemo_paths = [directory + f for f in nemo_files]
    nemo_paths =  [glob.glob(directory + f) for f in nemo_files]
    # ds = open_nemo_and_domain_cfg(nemo_files=nemo_paths[0],
    #                               domcfg_files=mask16_path)

    domcfg = open_domain_cfg(files = mask16_path)
    data = xr.open_dataset(nemo_paths[0][0])

    if variable == 'Ld':
        data = data.rename({'time_counter': 't', 
                            'x': 'x_c',
                            'y': 'y_c'})
    

    # merge data and domcfg
    ds = xr.merge([data, domcfg])

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
                domcfg,
                metrics=get_metrics(domcfg),
                )

    bd = {'boundary': 'extend'}
    
    # interpolate if not gridT and filter
    if variable == 'vor':
        ds['vor_c'] = grid.interp(ds.vor, ['X', 'Y'], **bd)
        ds['vor_f'] = filter_irregular_with_land.apply(ds['vor_c'], 
                                                           dims=['y_c', 'x_c'])
    elif variable == 'ke':
        if variable_to_filter == 'coarse_ke':
            ds['mke_f'] = filter_irregular_with_land.apply(
                ds['coarse_ke'],                                            
                dims=['y_c', 'x_c'])
            
        elif variable_to_filter == 'fine_ke':
            ds['eke_f'] = filter_irregular_with_land.apply(
                ds['fine_ke'],                                            
                dims=['y_c', 'x_c'])
        # variable = 'mke'
    elif variable == 'ug':
        ds['ug_c'] = grid.interp(ds.ug, ['X'], **bd)
        ds['ug_f'] = filter_irregular_with_land.apply(ds['ug_c'], 
                                                           dims=['y_c', 'x_c'])
        ds['ug_fc'] = grid.interp(ds.ug_f, ['X'], **bd)
    elif variable == 'vg':
        ds['vg_c'] = grid.interp(ds.vg, ['Y'], **bd)
        ds['vg_f'] = filter_irregular_with_land.apply(ds['vg_c'], 
                                                           dims=['y_c', 'x_c'])
        ds['vg_fc'] = grid.interp(ds.vg_f, ['Y'], **bd)
    else:
        ds[f'{variable}_f'] = filter_irregular_with_land.apply(ds[variable], 
                                                           dims=['y_c', 'x_c'])
        
    # create dataset
    if variable =='ug':
        ds_tmp = xr.Dataset(
            data_vars={
                f'{variable_name}': (["t", "y_c", "x_f"], 
                            ds[f'{variable_name}_fc'].values),
            },
            coords={
                "t": (["t"], ds[f'{variable_name}_fc'].t.values,
                            ds[f'{variable_name}_fc'].t.attrs),
                "gphiu": (["y_c", "x_f"], ds[f'{variable_name}_fc'].gphiu.values, 
                        {"standard_name": "Latitude", "units": "degrees_north"}),
                "glamu": (["y_c", "x_f"], ds[f'{variable_name}_fc'].glamu.values, 
                        {"standard_name": "Longitude","units": "degrees_east"}),
            },
            attrs={
                "name": "NEMO dataset",
                "description": f"Filtered {variable_name} \
                                -> ocean U grid variables",
            },
    )
    elif variable =='vg':
        ds_tmp = xr.Dataset(
            data_vars={
                f'{variable_name}': (["t", "y_f", "x_c"], 
                            ds[f'{variable_name}_fc'].values),
            },
            coords={
                "t": (["t"], ds[f'{variable_name}_fc'].t.values,
                            ds[f'{variable_name}_fc'].t.attrs),
                "gphiv": (["y_f", "x_c"], ds[f'{variable_name}_fc'].gphiv.values, 
                        {"standard_name": "Latitude", "units": "degrees_north"}),
                "glamv": (["y_f", "x_c"], ds[f'{variable_name}_fc'].glamv.values, 
                        {"standard_name": "Longitude","units": "degrees_east"}),
            },
            attrs={
                "name": "NEMO dataset",
                "description": f"Filtered {variable_name} \
                                -> ocean V grid variables",
            },
    )
    else:
        ds_tmp = xr.Dataset(
            data_vars={
                f'{variable_name}': (["t", "y_c", "x_c"], 
                            ds[f'{variable_name}_f'].values),
            },
            coords={
                "t": (["t"], ds[f'{variable_name}_f'].t.values,
                            ds[f'{variable_name}_f'].t.attrs),
                "gphit": (["y_c", "x_c"], ds[f'{variable_name}_f'].gphit.values, 
                        {"standard_name": "Latitude", "units": "degrees_north"}),
                "glamt": (["y_c", "x_c"], ds[f'{variable_name}_f'].glamt.values, 
                        {"standard_name": "Longitude","units": "degrees_east"}),
            },
            attrs={
                "name": "NEMO dataset",
                "description": f"Filtered {variable_name} at cell centre \
                                -> ocean T grid variables",
            },
        )

    # extract date_end from nemo_paths
    filename = nemo_paths[0][0].split('/')[-1]
    date_end = filename.split('_')[3]

    # extract time counter bounds from original file
    ref = xr.open_dataset(directory +
                          f'MINT_1d_{date_init}_{date_end}_{variable}_{region}.nc')

    ds_tmp["time_counter_bounds"] = ref["time_counter_bounds"]

    # save data to netcdf
    output_file = f'MINT_1d_{date_init}_{date_end}_{variable_name}_f_{region}.nc'

    save_directory = f'/gws/nopw/j04/ai4pex/twilder/NEMO_data/DINO/EXP16/features/{region}/filtered_data/'

    ds_tmp.to_netcdf(save_directory + output_file)