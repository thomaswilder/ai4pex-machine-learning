#!/usr/bin/env python3

'''
    Description: Calculates baroclinic deformation radius from NEMO DINO data.

    Note: This script is modified slightly from the script in ORCA36 directory.
'''

import os
import xarray as xr
from datetime import datetime
from dateutil.relativedelta import relativedelta
import numpy as np
import glob

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(filename='compute_deformation_radius.log', level=logging.DEBUG)

logger.info('Begin...')


directory = '/gws/nopw/j04/ai4pex/twilder/NEMO_data/DINO/EXP16/features/'
mask_path = '~/Python/AI4PEX/DINO/mesh_mask_exp16.nc'

# open mask dataset
mask = xr.open_dataset(mask_path)
logger.info('Mask dataset is: %s', mask)
# print(mask.e3t_0)

# Initial date string
start_date_init_str = "00710701"

# End date string
end_date_init_str = "00730101"
# end_date_init_str = "00730101"


# Convert date strings to datetime objects
start_date_init = datetime.strptime(start_date_init_str, "%Y%m%d")
end_date_init = datetime.strptime(end_date_init_str, "%Y%m%d")

# Loop to increment date by one day
current_date_init = start_date_init
# current_date_end = start_date_end
while current_date_init < end_date_init:

    logger.info('Processing date: %s', current_date_init)

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

    # set nemo filename using dates
    nemo_files = [f'MINT_1d_{date_init}_*_bn2.nc']

    # get path
    nemo_paths =  [glob.glob(directory + f) for f in nemo_files]
    print(nemo_paths)

    # open dataset
    ds = xr.open_dataset(nemo_paths[0][0])

    # rename coordinate
    ds = ds.rename({'deptht': 'nav_lev'})

    e3t = mask.e3t_0.squeeze('time_counter').\
            expand_dims({'time_counter': ds.vobn2.time_counter})
    # logger.info('e3t is: %s', e3t)
    tmask = mask.tmask.squeeze('time_counter').\
            expand_dims({'time_counter': ds.vobn2.time_counter})
    # logger.info('tmask is: %s', tmask)
    ff_t = mask.ff_t.squeeze('time_counter').\
            expand_dims({'time_counter': ds.vobn2.time_counter})
    # logger.info('ff_t is: %s', ff_t)

    # deal with negative values and save to file
    ds_tmp = xr.Dataset()
    ds_tmp['N2'] = xr.where( ds.vobn2<0, 1e-12, ds.vobn2 )
    ds_tmp.to_netcdf(directory + "N2_temp.nc")

    # close dataset to save memory
    del ds, ds_tmp
    
    # Find buoyancy frequency and save to file
    ds_tmp = xr.open_dataset(directory + "N2_temp.nc")
    ds_tmp1 = xr.Dataset()
    ds_tmp1['N'] = np.sqrt(ds_tmp.N2)
    ds_tmp1.to_netcdf(directory + "N_temp.nc")

    # open buoyancy frequency file
    ds_tmp = xr.open_dataset(directory + "N_temp.nc")
    
    # compute deformation radius in km
    ds_tmp1 = xr.Dataset()
    ds_tmp1['Ld'] = ((ds_tmp.N*e3t)
                     .sum(dim='nav_lev')/(np.abs(ff_t)*1000.0))

    # logger.info('Ld Dataset is: %s', ds_tmp1)
    # print(ds_tmp1)
    
    # set land values to nan
    ds_tmp = xr.Dataset()
    ds_tmp["Ld"] = ds_tmp1.Ld*tmask.isel(nav_lev=0)
    # ds_tmp["Ld"] = xr.where(ds_tmp1.Ld==0, np.nan, ds_tmp1.Ld)

    logger.info('masked Dataset is: %s', ds_tmp)
    # print(ds_tmp)

    # create dataset
    ds = xr.Dataset(
        data_vars={
            'Ld': (["time_counter", "y", "x"], ds_tmp.Ld.values),
        },
        coords={
            "time_counter": (["time_counter"], ds_tmp.time_counter.values,
                             ds_tmp.time_counter.attrs),
            "gphit": (["y", "x"], mask.gphit.isel(time_counter=0).values,
                      {"standard_name": "Latitude", "units": "degrees_north"}),
            "glamt": (["y", "x"], mask.glamt.isel(time_counter=0).values,
                      {"standard_name": "Longitude", "units": "degrees_east"}),
        },
        attrs={
            "name": "NEMO dataset",
            "description": "Deformation radius \
                              -> ocean T grid variables",
        },
    )

    # extract date_end from nemo_paths
    filename = nemo_paths[0][0].split('/')[-1]
    date_end = filename.split('_')[3]
    date_end

    # extract time counter bounds from original file
    ref_dir = '/gws/nopw/j04/ai4pex/twilder/NEMO_data/DINO/EXP16/production/OUTPUTS/'
    ref = xr.open_dataset(ref_dir + 
                          f'MINT_1d_{date_init}_{date_end}_grid_T.nc')
    
    ds["time_counter_bounds"] = ref["time_counter_bounds"]

    # save to netcdf
    output_file = f'MINT_1d_{date_init}_{date_end}_Ld.nc'

    save_directory = '/gws/nopw/j04/ai4pex/twilder/NEMO_data/DINO/EXP16/features/'

    logger.info('Saving deformation radius to: %s', save_directory + output_file)

    ds.to_netcdf(save_directory + output_file)

    # remove temporary files
    files = [directory + 'N2_temp.nc', directory + 'N_temp.nc']
    for filepath in files:
        if os.path.exists(filepath):
            os.remove(filepath)