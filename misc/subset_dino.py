#!/usr/bin/env python3

'''
Description: This script will subset DINO EXP16 velocity and deformation radius data.
    Doing so will speed up the remaining computations for vorticity, MKE and EKE.

Method: 
    Use EXP4 mask file to confirm if regridded data set will be of an adequate size.

'''

import glob
import xarray as xr
from datetime import datetime
from dateutil.relativedelta import relativedelta

#------------ Parameters to set ------------#
variable = 'ug'

region = 'SO_JET'

open_directory = '/gws/nopw/j04/ai4pex/twilder/NEMO_data/DINO/EXP16/features/'
save_directory = f'/gws/nopw/j04/ai4pex/twilder/NEMO_data/DINO/EXP16/features/{region}/'

lat_min = -47.5
lat_max = -37.5
lon_min = 2.5
lon_max = 17.5
#------------------------------------------#

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

    nemo_files = [f'MINT_1d_{date_init}_*{variable}.nc']

    nemo_paths = [glob.glob(open_directory + f) for f in nemo_files]

    # now open dataset to subset
    ds = xr.open_dataset(nemo_paths[0][0])

    # mask and subset data 
    ds_masked = ds.where( (lat_min < ds.gphiu) &
            (ds.gphiu < lat_max) &
            (lon_min < ds.glamu) &
            (ds.glamu < lon_max)
            )

    ds_subset = ds_masked.dropna(dim='y', how='all').dropna(dim='x', how='all')

    # extract date_end from filename
    filename = nemo_paths[0][0].split('/')[-1]
    date_end = filename.split('_')[3]

    # save subsetted data
    ds_subset.to_netcdf(save_directory + 
                        f'MINT_1d_{date_init}_{date_end}_{variable}_{region}.nc')