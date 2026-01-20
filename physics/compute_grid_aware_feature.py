#!/usr/bin/env python3

'''
    Description: Compute the feature to add scale-awareness to the parameterisation
    Ld / \Delta s

    Method: Load in Ld and EXP4 mask and compute.
'''

import xarray as xr
from xnemogcm import open_domain_cfg
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta
import glob

# ------------ misc parameters ---------------- #

region = 'SO_JET'

base_dir = f'/gws/nopw/j04/ai4pex/twilder/NEMO_data/DINO/EXP16/features/{region}/'
data_dir = 'coarsened_data/'
mask_path = [base_dir + 'mesh_mask_exp4_SO_JET.nc']

# -------------------------------------------- #
# Calculate grid scale, Delta s #
# -------------------------------------------- #
domcfg = open_domain_cfg(files = mask_path)

delta_s = np.sqrt(domcfg.e1t*domcfg.e2t)

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
    nemo_files = [f'MINT_1d_{date_init}_*_Ld_c_{region}.nc']

    nemo_paths =  [glob.glob(base_dir + data_dir + f) for f in nemo_files]

    # extract date_end from nemo_paths
    filename = nemo_paths[0][0].split('/')[-1]
    date_end = filename.split('_')[3]

    data = xr.open_dataset(nemo_paths[0][0])

    # compute scale aware feature
    scale_aware =  (data.Ld*1000) / delta_s

    # convert to dataset
    ds = scale_aware.to_dataset(name='sa')

    # assign attributes
    ds['sa'] = ds.sa.assign_attrs({'standard_name': 'scale_aware',
                    'long_name': 'deformation_radius_over_grid_scale'})

    # save data to filename
    output_filename = f'MINT_1d_{date_init}_{date_end}_sa_c_{region}.nc'

    ds.to_netcdf(base_dir + data_dir + output_filename)