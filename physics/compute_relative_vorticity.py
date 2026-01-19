#!/usr/bin/env python3

'''
    Description: Calculates relative vorticity
    
    Method: Uses geostrophic velocity
'''

import glob
import xarray as xr
from xnemogcm import open_domain_cfg, get_metrics
import xgcm
from datetime import datetime
from dateutil.relativedelta import relativedelta
import numpy as np

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(filename='compute_vorticity.log', level=logging.INFO, filemode='w')

logger.info('Begin...')

region = 'SO_JET'


directory = f'/gws/nopw/j04/ai4pex/twilder/NEMO_data/DINO/EXP16/features/{region}/'
mask_path = [directory + f'mesh_mask_exp16_{region}.nc']

# Initial date string
start_date_init_str = "00610201"

# End date string
end_date_init_str = "00690101"


# Convert date strings to datetime objects
start_date_init = datetime.strptime(start_date_init_str, "%Y%m%d")
end_date_init = datetime.strptime(end_date_init_str, "%Y%m%d")

# Loop to increment date by one day
current_date_init = start_date_init
while current_date_init < end_date_init:

    logger.info('Processing date: %s', current_date_init)

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
    nemo_files = [f'MINT_1d_{date_init}_*_ug_{region}.nc',
                  f'MINT_1d_{date_init}_*_vg_{region}.nc']
    print(nemo_files)

    nemo_paths = [glob.glob(directory + f) for f in nemo_files]

    nemo_files = [nemo_paths[0][0].split('/')[-1], nemo_paths[1][0].split('/')[-1]]

    # open dataset using xnemogcm
    nemo_paths = [directory + f for f in nemo_files]
    # ds = open_nemo_and_domain_cfg(nemo_files=nemo_paths,
    #                               domcfg_files=mask_path)
    #! issues opening subsetted data with full xnemogcm, just hangs?
    #* use this workaround to open data and domcfg separately
    dataU = xr.open_dataset(nemo_paths[0])
    dataV = xr.open_dataset(nemo_paths[1])
    domcfg = open_domain_cfg(files = mask_path)

    # get surface domcfg
    domcfg_surface = domcfg.isel(z_c=0, z_f=0)

    # logger.info('Dataset loaded is: %s', ds)

    # set up xgcm grid
    # grid = xgcm.Grid(
    #             ds,
    #             metrics=get_metrics(ds),
    #             periodic={'X': True, 'Y': False},
    #             boundary={'Y': 'extend'},
    #             )

    grid = xgcm.Grid(domcfg, 
                     metrics=get_metrics(domcfg))
    
    bd = {'boundary': 'extend'}

    # subset data for surface
    # ds_ss = ds.isel(z_c=0, z_f=0)

    # logger.info('Dataset subsetted to surface is: %s', ds_ss)

    # remove ds from memory
    # del ds

    # compute relative vorticity
    zeta = 1/(domcfg_surface.e1f*domcfg_surface.e2f) * \
            (grid.diff(dataV.vg*domcfg_surface.e2v, 'X', **bd) \
             - grid.diff(dataU.ug*domcfg_surface.e1u, 'Y', **bd)) \
                * domcfg_surface.fmask
    
    logger.info('Relative vorticity calculated is: %s', zeta)

    # zero over equator region
    # if region == 'full':
    #     zeta = xr.where(((ds_ss.gphif>2) | (ds_ss.gphif<-2)), zeta, 0.0)

    # create dataset
    ds_tmp = xr.Dataset(
        data_vars={
            'vor': (["y_f", "x_f", "t"], 
                        zeta.values),
        },
        coords={
            "t": (["t"], zeta.t.values,
                        zeta.t.attrs),
            "gphif": (["y_f", "x_f"], domcfg_surface.gphif.values, 
                      {"standard_name": "Latitude", "units": "degrees_north"}),
            "glamf": (["y_f", "x_f"], domcfg_surface.glamf.values, 
                      {"standard_name": "Longitude","units": "degrees_east"}),
        },
        attrs={
            "name": "NEMO dataset",
            "description": "Relative vorticity from geostrophic velocities \
                              -> ocean F grid variables",
        },
    )

    logger.info('Dataset to be saved is: %s', ds_tmp)

    # extract date_end from nemo_paths
    filename = nemo_paths[0].split('/')[-1]
    date_end = filename.split('_')[3]
    date_end

    # extract time counter bounds from original file
    ref = xr.open_dataset(directory +
                          f'MINT_1d_{date_init}_{date_end}_ug_{region}.nc')

    ds_tmp["time_counter_bounds"] = ref["time_counter_bounds"]
    # ds_tmp["time_counter_bounds"] = ref["time_counter_bounds"]

    # save data to netcdf
    output_file = f'MINT_1d_{date_init}_{date_end}_vor_{region}.nc'

    save_directory = f'/gws/nopw/j04/ai4pex/twilder/NEMO_data/DINO/EXP16/features/{region}/'

    ds_tmp.to_netcdf(save_directory + output_file)