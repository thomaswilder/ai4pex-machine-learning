#!/usr/bin/env python3

'''
    Description: Regrid features and target using xesmf

    Method: Load in input and target grid from 
        get_mask_bounds_for_conservative_regridding.ipynb
        and load in weights precomputed in get_regridding_weights.ipynb
'''

import xarray as xr
import xesmf as xe
import xgcm
import numpy as np
from xnemogcm import open_domain_cfg, get_metrics
import glob
from datetime import datetime
from dateutil.relativedelta import relativedelta

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(filename='regrid_data.log', level=logging.DEBUG)

logger.info('Begin...')

def create_regridded_dataset(ds_regrid, variable):
    """
        Create dataset 
    """
    
    # Create data_vars dictionary dynamically
    data_vars = {}
    data_vars[variable] = (["t", "y_c", "x_c"], 
                        ds_regrid[variable].values)
    
    # Create the dataset
    ds_tmp = xr.Dataset(
        data_vars=data_vars,
        coords={
            "t": (["t"], ds_regrid.t.values),
            "gphit": (["y_c", "x_c"], ds_regrid.gphit.values,
                      {"standard_name": "Latitude", "units": "degrees_north"}),
            "glamt": (["y_c", "x_c"], ds_regrid.glamt.values,
                      {"standard_name": "Longitude","units": "degrees_east"}),
        },
        attrs={
            "description": \
                "DINO EXP16 regridded to EXP4 -> ocean T grid variables",
        }
    )
    
    return ds_tmp

# ------------ misc parameters ---------------- #

region = 'SO_JET'

variable = 'ke'

directory = f'/gws/nopw/j04/ai4pex/twilder/NEMO_data/DINO/EXP16/features_take2/{region}/'
mask_path = [directory + f'mesh_mask_exp16_{region}.nc']

weights_fn = directory + f'weights_{region}.nc'

# -------------------------------------------- #

# load in input and target grid
array = np.load(directory + f'{region}_input_grid_16.npy', allow_pickle=True)
input_grid = array.item()

array = np.load(directory + f'{region}_target_grid_025.npy', allow_pickle=True)
target_grid = array.item()

# load in regridder
regridder = xe.Regridder(input_grid, 
                         target_grid, 
                         "conservative", 
                         weights=weights_fn)

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
    if variable != 'ke':
        nemo_files = [f'MINT_1d_{date_init}_*_{variable}_f_{region}.nc']
    else:
        nemo_files = [f'MINT_1d_{date_init}_*_{variable}_{region}.nc']

    nemo_paths = [glob.glob(directory + f) for f in nemo_files]

    # extract date_end from nemo_paths
    filename = nemo_paths[0][0].split('/')[-1]
    date_end = filename.split('_')[3]

    nemo_files = [nemo_paths[0][0].split('/')[-1]]

    # open dataset using xnemogcm
    nemo_paths = [directory + f for f in nemo_files]
    # ds = open_nemo_and_domain_cfg(nemo_files=nemo_paths,
    #                               domcfg_files=mask_path)
    #! issues opening subsetted data with full xnemogcm, just hangs?
    #* use this workaround to open data and domcfg separately
    data = xr.open_dataset(nemo_paths[0])
    domcfg = open_domain_cfg(files = mask_path)

    # get surface domcfg
    domcfg_surface = domcfg.isel(z_c=0, z_f=0)

    # set up xgcm grid
    grid = xgcm.Grid(
            domcfg,
            metrics=get_metrics(domcfg),
            )

    bd = {'boundary': 'extend'}

    if variable == 'uo':
        data['uo'] = grid.interp(data.uo, ['X'], **bd)
        data = data.rename({'gphiu': 'lat', 'glamu': 'lon'})
    elif variable == 'vo':
        data['vo'] = grid.interp(data.vo, ['Y'], **bd)
        data = data.rename({'gphiv': 'lat', 'glamv': 'lon'})
    else:
    # Rename coordinates to lat and lon for xesmf
        data = data.rename({'gphit': 'lat', 'glamt': 'lon'})

    # perform regridding
    ds_tmp = xr.Dataset()
    if variable != 'ke':
        ds_tmp[variable] = regridder(data[variable])
    else:
        ds_tmp['fine_ke'] = regridder(data['fine_ke'])
        ds_tmp['coarse_ke'] = regridder(data['coarse_ke'])

    # Rename coordinates back to gphit and glamt
    if variable == 'uo':
        ds_tmp = ds_tmp.rename({"lat": "gphiu", "lon": "glamu"})
    elif variable == 'vo':
        ds_tmp = ds_tmp.rename({"lat": "gphiv", "lon": "glamv"})
    else:
        ds_tmp = ds_tmp.rename({"lat": "gphit", "lon": "glamt"})

    if variable == 'uo':
        ds = ds_tmp
        ds = ds.rename({'y': 'y_c', 'x': 'x_f'})
    elif variable == 'vo':
        ds = ds_tmp
        ds = ds.rename({'y': 'y_f', 'x': 'x_c'})
    elif variable == 'vobn2':
        ds = ds_tmp
        ds = ds.rename({'y': 'y_c', 'x': 'x_c'})
    elif variable == 'ke':
        ds = ds_tmp
        ds = ds.rename({'y': 'y_c', 'x': 'x_c'})
    else:
        ds = create_regridded_dataset(ds_tmp, variable)

    # save data to netcdf
    output_file = f'MINT_1d_{date_init}_{date_end}_{variable}_cg_{region}.nc'

    if variable != 'ke':
        save_directory = directory + '../coarsened_data/'
    else:
        save_directory = directory + 'coarsened_data/'

    ds.to_netcdf(save_directory + output_file)

logger.info('End!')