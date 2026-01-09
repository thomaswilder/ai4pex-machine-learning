#!/usr/bin/env python3

'''
    Description: Calculates Nondivergent streamfunction

    Method: Uses geostrophic u velocity component for Southern Ocean
        n.b. use v component and integrate eastwards for Northern Ocean
'''

import xarray as xr
from xnemogcm import open_nemo_and_domain_cfg, get_metrics
import xgcm
from datetime import datetime
from dateutil.relativedelta import relativedelta
import numpy as np


directory = '/gws/nopw/j04/ai4pex/twilder/NEMO_data/DINO/EXP16/features/'
mask_path = ['~/Python/AI4PEX/DINO/mesh_mask_exp16.nc']

# Initial date string
start_date_init_str = "00700601"
start_date_end_str = "00700630"
# End date string
end_date_init_str = "00700701"
end_date_end_str = "00700730"

# Convert date strings to datetime objects
start_date_init = datetime.strptime(start_date_init_str, "%Y%m%d")
end_date_init = datetime.strptime(end_date_init_str, "%Y%m%d")
start_date_end = datetime.strptime(start_date_end_str, "%Y%m%d")
end_date_end = datetime.strptime(end_date_end_str, "%Y%m%d")

# Loop to increment date by one day
current_date_init = start_date_init
current_date_end = start_date_end
while current_date_init < end_date_init:

    # Increment for next date
    next_date_init = current_date_init + relativedelta(months=+1)
    next_date_end = current_date_end + relativedelta(months=+1)
    print(next_date_init)

    # Convert dates to string for nemo files
    # date = current_date.strftime("%Y%m%d")
    date_init = (
        f"{current_date_init.year:04d}"\
        f"{current_date_init.month:02d}"\
        f"{current_date_init.day:02d}" # fix for adding leading zeros back in
    )

    date_end = (
        f"{current_date_end.year:04d}"\
        f"{current_date_end.month:02d}"\
        f"{current_date_end.day:02d}" # fix for adding leading zeros back in
    )

    print(date_init)
    print(date_end)

    # Assign current date to next date for next loop
    current_date_init = next_date_init
    current_date_end = next_date_end

    # set nemo filename using dates
    nemo_files = [f'MINT_1d_{date_init}_{date_end}_ug.nc']
    print(nemo_files)

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

    # compute streamfunction
    psi = -grid.cumint(ds_ss.ug, 'Y')

    # mask northern domain since vg is used for sf there
    psi_masked = xr.where(ds_ss.gphif > 0, 0.0, psi)

    # interpolate to grid t
    psi_t = grid.interp(psi_masked, ['X', 'Y']) * ds_ss.tmask

    del psi, psi_masked

    # create dataset
    ds_tmp = xr.Dataset(
        data_vars={
            'psi': (["y", "x", "time_counter"], psi_t.values),
        },
        coords={
            "time_counter": (["time_counter"], ds_ss.t.values,
                             ds_ss.t.attrs),
            "gphit": (["y", "x"], ds_ss.gphit.values,
                      {"standard_name": "Latitude", "units": "degrees_north"}),
            "glamt": (["y", "x"], ds_ss.glamt.values,
                      {"standard_name": "Longitude", "units": "degrees_east"}),
        },
        attrs={
            "name": "NEMO dataset",
            "description": "Streamfunction using cumint of u in j direction \
                              -> ocean T grid variables",
        },
    )

    # extract time counter bounds from original file
    ref = xr.open_dataset(directory +
                          f'MINT_1d_{date_init}_{date_end}_grid_T.nc')

    ds_tmp["time_counter_bounds"] = ref["time_counter_bounds"]
    ds_tmp["time_counter_bounds"] = ref["time_counter_bounds"]

    # save data to netcdf
    output_file = f'MINT_1d_{date_init}_{date_end}_psi_u.nc'

    save_directory = '/gws/nopw/j04/ai4pex/twilder/NEMO_data/DINO/EXP16/features/'

    ds_tmp.to_netcdf(save_directory + output_file)