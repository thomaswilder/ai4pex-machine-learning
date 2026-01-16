#!/usr/bin/env python3

'''
    Description: Calculates relative vorticity
    
    Method: Uses geostrophic velocity
'''

import glob
import xarray as xr
from xnemogcm import open_nemo_and_domain_cfg, get_metrics
import xgcm
from datetime import datetime
from dateutil.relativedelta import relativedelta
import numpy as np


directory = '/gws/nopw/j04/ai4pex/twilder/NEMO_data/DINO/EXP16/features/'
mask_path = ['~/Python/AI4PEX/DINO/mesh_mask_exp16.nc']

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
    nemo_files = [f'MINT_1d_{date_init}_*_ug.nc',
                  f'MINT_1d_{date_init}_*_vg.nc']
    print(nemo_files)

    nemo_paths = [glob.glob(directory + f) for f in nemo_files]

    nemo_files = [nemo_paths[0][0].split('/')[-1], nemo_paths[1][0].split('/')[-1]]

    # open dataset using xnemogcm
    nemo_paths = [directory + f for f in nemo_files]
    ds = open_nemo_and_domain_cfg(nemo_files=nemo_paths,
                                  domcfg_files=mask_path)

    # set up xgcm grid
    grid = xgcm.Grid(
                ds,
                metrics=get_metrics(ds),
                periodic={'X': True, 'Y': False},
                boundary={'Y': 'extend'},
                )

    # subset data for surface
    ds_ss = ds.isel(z_c=0, z_f=0)

    # remove ds from memory
    del ds

    # compute relative vorticity
    zeta = 1/(ds_ss.e1f*ds_ss.e2f) * \
           ( grid.diff(ds_ss.vg*ds_ss.e2v, 'X') \
           - grid.diff(ds_ss.ug*ds_ss.e1u, 'Y') ) * ds_ss.fmask
    
    # zero over equator region
    zeta = xr.where(((ds_ss.gphif>2) | (ds_ss.gphif<-2)), zeta, 0.0)

    # create dataset
    ds_tmp = xr.Dataset(
        data_vars={
            'vor': (["y", "x", "time_counter"], zeta.values),
        },
        coords={
            "time_counter": (["time_counter"], ds_ss.t.values,
                             ds_ss.t.attrs),
            "gphif": (["y", "x"], ds_ss.gphif.values,
                      {"standard_name": "Latitude", "units": "degrees_north"}),
            "glamf": (["y", "x"], ds_ss.glamf.values,
                      {"standard_name": "Longitude", "units": "degrees_east"}),
        },
        attrs={
            "name": "NEMO dataset",
            "description": "Relative vorticity from geostrophic velocities \
                              -> ocean F grid variables",
        },
    )

    # extract date_end from nemo_paths
    filename = nemo_paths[0].split('/')[-1]
    date_end = filename.split('_')[3]
    date_end

    # extract time counter bounds from original file
    ref = xr.open_dataset(directory +
                          f'MINT_1d_{date_init}_{date_end}_ug.nc')

    ds_tmp["time_counter_bounds"] = ref["time_counter_bounds"]
    ds_tmp["time_counter_bounds"] = ref["time_counter_bounds"]

    # save data to netcdf
    output_file = f'MINT_1d_{date_init}_{date_end}_vor.nc'

    save_directory = '/gws/nopw/j04/ai4pex/twilder/NEMO_data/DINO/EXP16/features/'

    ds_tmp.to_netcdf(save_directory + output_file)