#!/usr/bin/env python3

'''
    Description: Calculates geostrophic velocities using sea surface height and
                    thermal wind.
                 The velocities are used to calculate:
                    1) Eddy kinetic energy - target
                    2) Nondivergent streamfunction - input
                    3) Relative vorticity - input
                    4) Strain rate - input
'''

import xarray as xr
from xnemogcm import open_nemo_and_domain_cfg, get_metrics
import xgcm
from datetime import datetime
from dateutil.relativedelta import relativedelta
import numpy as np
import glob


directory = '/gws/nopw/j04/ai4pex/twilder/NEMO_data/DINO/EXP16/production/OUTPUTS/'
mask_path = ['~/Python/AI4PEX/DINO/mesh_mask_exp16.nc']

# Initial date string
start_date_init_str = "00721201"

# End date string
# end_date_init_str = "00610301"
end_date_init_str = "00730101"

# Convert date strings to datetime objects
start_date_init = datetime.strptime(start_date_init_str, "%Y%m%d")
end_date_init = datetime.strptime(end_date_init_str, "%Y%m%d")

# Loop to increment date by one day
current_date_init = start_date_init
# current_date_end = start_date_end
while current_date_init < end_date_init:

    # Increment for next date
    next_date_init = current_date_init + relativedelta(months=+1)
    print(next_date_init)

    # Convert dates to string for nemo files
    date_init = (
        f"{current_date_init.year:04d}"\
        f"{current_date_init.month:02d}"\
        f"{current_date_init.day:02d}" # fix for adding leading zeros back in
    )

    print(date_init)
    # print(date_end)

    # Assign current date to next date for next loop
    current_date_init = next_date_init
    # current_date_end = next_date_end

    # set nemo filename using dates
    nemo_files = [f'MINT_1d_{date_init}_*_grid_T.nc']
    print(nemo_files)

    # open dataset using xnemogcm
    # nemo_paths = [directory + f for f in nemo_files]
    nemo_paths =  [glob.glob(directory + f) for f in nemo_files]
    ds = open_nemo_and_domain_cfg(nemo_files=nemo_paths[0],
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

    # interpolate ssh to u and v points
    zos_u = grid.interp(ds_ss.zos, 'X') * ds_ss.umask
    zos_v = grid.interp(ds_ss.zos, 'Y') * ds_ss.vmask

    # set coriolis parameter to 1e-12 at equator
    ff = xr.where(((ds_ss.gphit>0.1) | (ds_ss.gphit<-0.1)), ds_ss.ff_t, 1e-12)

    # compute geostrophic velocities on t points
    ug = - (9.81/(ff)) * \
            ( grid.diff(zos_v, 'Y') / ds_ss.e2t ) * ds_ss.tmask
    vg = (9.81/(ff)) * \
            ( grid.diff(zos_u, 'X') / ds_ss.e1t ) * ds_ss.tmask

    # put velocities back on u and v points
    vg_v = grid.interp(vg, 'Y') * ds_ss.vmask
    ug_u = grid.interp(ug, 'X') * ds_ss.umask

    # create datasets for each velocity
    ds_u = xr.Dataset(
        data_vars={
            'ug': (["y", "x", "time_counter"], 
                        ug_u.values),
        },
        coords={
            "time_counter": (["time_counter"], ds_ss.t.values, 
                             ds_ss.t.attrs),
            "gphiu": (["y", "x"], ds_ss.gphiu.values, 
                      {"standard_name": "Latitude", "units": "degrees_north"}),
            "glamu": (["y", "x"], ds_ss.glamu.values, 
                      {"standard_name": "Longitude","units": "degrees_east"}),
        },
        attrs={
            "name": "NEMO dataset",
            "description": "Geostrophic ocean current in i direction \
                              -> ocean U grid variables",
        },
    )

    ds_v = xr.Dataset(
        data_vars={
            'vg': (["y", "x", "time_counter"], 
                        vg_v.values),
        },
        coords={
            "time_counter": (["time_counter"], ds_ss.t.values, 
                             ds_ss.t.attrs),
            "gphiv": (["y", "x"], ds_ss.gphiv.values, 
                      {"standard_name": "Latitude", "units": "degrees_north"}),
            "glamv": (["y", "x"], ds_ss.glamv.values, 
                      {"standard_name": "Longitude","units": "degrees_east"}),
        },
        attrs={
            "name": "NEMO dataset",
            "description": "Geostrophic ocean current in j direction \
                              -> ocean V grid variables",
        },
    )

    # extract date_end from nemo_paths
    filename = nemo_paths[0][0].split('/')[-1]
    date_end = filename.split('_')[3]
    date_end

    # extract time counter bounds from original file
    ref = xr.open_dataset(directory + 
                          f'MINT_1d_{date_init}_{date_end}_grid_T.nc')
    
    ds_u["time_counter_bounds"] = ref["time_counter_bounds"]
    # ds_u["time_centered_bounds"] = ref["time_centered_bounds"]
    ds_v["time_counter_bounds"] = ref["time_counter_bounds"]
    # ds_v["time_centered_bounds"] = ref["time_centered_bounds"]

    # save data to netcdf
    output_u_file = f'MINT_1d_{date_init}_{date_end}_ug.nc'
    output_v_file = f'MINT_1d_{date_init}_{date_end}_vg.nc'

    save_directory = '/gws/nopw/j04/ai4pex/twilder/NEMO_data/DINO/EXP16/features/'

    ds_u.to_netcdf(save_directory + output_u_file)
    ds_v.to_netcdf(save_directory + output_v_file)

    
    