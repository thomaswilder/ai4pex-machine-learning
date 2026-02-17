#!/usr/bin/env python3

'''

    Description: Calculate divergence of velocity field

'''

import xarray as xr
import xgcm
import numpy as np
from xnemogcm import open_domain_cfg, get_metrics
import glob
from datetime import datetime
from dateutil.relativedelta import relativedelta

# ------------ set parameters ---------------- #

region = 'SO_JET'

directory = f'/gws/nopw/j04/ai4pex/twilder/NEMO_data/DINO/EXP16/features_take2/{region}/coarsened_data/'
mask_path = [directory + f'../mesh_mask_exp16_{region}.nc']

# Initial date string
start_date_init_str = "00610101"

# End date string
end_date_init_str = "00610201"
# -------------------------------------------- #

# Convert date strings to datetime objects
start_date_init = datetime.strptime(start_date_init_str, "%Y%m%d")
end_date_init = datetime.strptime(end_date_init_str, "%Y%m%d")

# Loop to increment date by one day
current_date_init = start_date_init
while current_date_init < end_date_init:

    # logger.info('Processing date: %s', current_date_init)

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
    nemo_files = [f'MINT_1d_{date_init}_*_uo_cg_{region}.nc',
                  f'MINT_1d_{date_init}_*_vo_cg_{region}.nc']
    print(nemo_files)

    nemo_paths = [glob.glob(directory + f) for f in nemo_files]

    nemo_files = [nemo_paths[0][0].split('/')[-1], nemo_paths[1][0].split('/')[-1]]

    # open dataset using xnemogcm
    nemo_paths = [directory + f for f in nemo_files]

    dataU = xr.open_dataset(nemo_paths[0])
    dataV = xr.open_dataset(nemo_paths[1])
    domcfg = open_domain_cfg(files = mask_path)

    # get surface domcfg
    domcfg_surface = domcfg.isel(z_c=0, z_f=0)

    # set up xgcm grid
    grid = xgcm.Grid(
            domcfg,
            metrics=get_metrics(domcfg),
            )

    bd = {'boundary': 'extend'}

    # calculate divergence
    div_uv = grid.diff(dataU.uo * domcfg.e2u * domcfg.e3u_0, 'X', **bd) / \
          (domcfg.e1t_0 * domcfg.e2t_0 * domcfg.e3t_0) \
            + grid.diff(dataV.vo * domcfg.e1v * domcfg.e3v_0, 'Y', **bd) / \
                  (domcfg.e1t * domcfg.e2t * domcfg.e3t_0)
    
    ds_tmp = xr.Dataset(
        data_vars={
            'div': (["y_c", "x_c", "t"], 
                        div_uv.values),
        },
        coords={
            "t": (["t"], div_uv.t.values,
                        div_uv.t.attrs),
            "gphit": (["y_c", "x_c"], domcfg_surface.gphit.values, 
                      {"standard_name": "Latitude", "units": "degrees_north"}),
            "glamt": (["y_c", "x_c"], domcfg_surface.glamt.values, 
                      {"standard_name": "Longitude","units": "degrees_east"}),
        },
        attrs={
            "name": "NEMO dataset",
            "description": "Divergence from coarse_grained \
                 velocities -> ocean T grid variables",
        },
    )

    # logger.info('Dataset to be saved is: %s', ds_tmp)

    # extract date_end from nemo_paths
    filename = nemo_paths[0].split('/')[-1]
    date_end = filename.split('_')[3]

    # save data to netcdf
    output_file = f'MINT_1d_{date_init}_{date_end}_div_cg_{region}.nc'

    save_directory = f'/gws/nopw/j04/ai4pex/twilder/NEMO_data/DINO/EXP16/features_take2/{region}/coarsened_data/'

    ds_tmp.to_netcdf(save_directory + output_file)