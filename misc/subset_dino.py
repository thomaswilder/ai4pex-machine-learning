#!/usr/bin/env python3

'''
Description: This script will subset DINO EXP16 velocity and bn2 data.
    Doing so will speed up the remaining computations for vorticity, 
    Ld, MKE and EKE.

Method: 
    Use EXP4 mask file to confirm if regridded data set will be of an adequate size.
    Uses indices to subset, which are found in subset_dino.ipynb



'''

import glob
import xarray as xr
from datetime import datetime
from dateutil.relativedelta import relativedelta

#------------ Parameters to set ------------#
variable = 'Ld'

grid = 'T'

region = 'SO_JET'

open_directory = '/gws/nopw/j04/ai4pex/twilder/NEMO_data/DINO/EXP16/features/'
save_directory = f'/gws/nopw/j04/ai4pex/twilder/NEMO_data/DINO/EXP16/features/{region}/'

#* dictionary listing regions and indices
regions = {
    'SO_JET': {'jmin': 726,
               'jmax': 942,
               'imin': 41,
               'imax': 279},
}

region_params = regions[region]
jmin = region_params['jmin']
jmax = region_params['jmax']
imin = region_params['imin']
imax = region_params['imax']

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
    if grid == 'U':
        ds_masked = ds.where( (ds.gphiu.isel(x_f=imin-1,y_c=jmin-1).values < ds.gphiu) &
                (ds.gphiu < ds.gphiu.isel(x_f=imax,y_c=jmax).values) &
                (ds.glamu.isel(x_f=imin-1,y_c=jmin-1).values < ds.glamu) &
                (ds.glamu < ds.glamu.isel(x_f=imax,y_c=jmax).values)
                )

        ds_subset = ds_masked.dropna(dim='y_c', how='all').dropna(dim='x_f', how='all')
    elif grid == 'V':
        ds_masked = ds.where( (ds.gphiv.isel(x_c=imin-1,y_f=jmin-1).values < ds.gphiv) &
                (ds.gphiv < ds.gphiv.isel(x_c=imax,y_f=jmax).values) &
                (ds.glamv.isel(x_c=imin-1,y_f=jmin-1).values < ds.glamv) &
                (ds.glamv < ds.glamv.isel(x_c=imax,y_f=jmax).values)
                )

        ds_subset = ds_masked.dropna(dim='y_f', how='all').dropna(dim='x_c', how='all')
    #* note indice change as Ld was computed and set with xnemo readable
    elif grid == 'T':
        ds_masked = ds.where( (ds.gphit.isel(x=imin-1,y=jmin-1).values < ds.gphit) &
                (ds.gphit < ds.gphit.isel(x=imax,y=jmax).values) &
                (ds.glamt.isel(x=imin-1,y=jmin-1).values < ds.glamt) &
                (ds.glamt < ds.glamt.isel(x=imax,y=jmax).values)
                )

        ds_subset = ds_masked.dropna(dim='y', how='all').dropna(dim='x', how='all')
    else:
        print('Grid type not recognised. Please use U, V or T.')

    # extract date_end from filename
    filename = nemo_paths[0][0].split('/')[-1]
    date_end = filename.split('_')[3]

    # save subsetted data
    ds_subset.to_netcdf(save_directory + 
                        f'MINT_1d_{date_init}_{date_end}_{variable}_{region}.nc')