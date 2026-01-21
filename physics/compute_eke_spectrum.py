#!/usr/bin/python3

'''
    Description:
        Computes the energy spectrum using a similar coarse graining approach
        to Sadek and Aluie (2018), but employs GCM-Filters. 

    Method:
        1) EKE is computed using Eq 9 in Buzzicotti et al., (2023).
        2) Length scales are defined and coarse EKE is computed at each.
        3) Cumulative EKE is then the average of EKE over a set domain.
        4) Cumulative EKE is differenced wrt lengthscale/wavenumber.
            See Eq 33 in Sadek and Aluie.

    Notes:
        Originally developed on LUMI for ORCA025 data.
        Adapted for JASMIN to diagnose spectrum of DINO configuration.
'''

#TODO adapt for DINO. Rebuild mask file.

import gcm_filters
import numpy as np
import xarray as xr
from xnemogcm import open_domain_cfg, get_metrics
import xgcm
from datetime import datetime
from dateutil.relativedelta import relativedelta

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(filename='compute_eke_spectrum.log', level=logging.INFO, filemode='w')

# logger.info('Modules loaded...')

logger.info('Begin...')

directory="/gws/nopw/j04/ai4pex/twilder/NEMO_data/DINO/EXP16/features/SO_JET/coarsened_data/"
mask_file="../mesh_mask_exp4_SO_JET.nc"


# # Initial date string
# start_date_str = "20181001"
# # End date string
# end_date_str = "20190101"

# # Convert date strings to datetime objects
# start_date = datetime.strptime(start_date_str, "%Y%m%d")
# end_date = datetime.strptime(end_date_str, "%Y%m%d")

# # Loop to increment date by one day
# current_date = start_date
# while current_date < end_date:

#     # Increment for next date
#     next_date = current_date + relativedelta(days=+1)
#     # print(next_date)

#     # Convert dates to string for nemo files
#     date = current_date.strftime("%Y%m%d")
#     # print(date)

# set nemo filename using dates
nemo_files = ["MINT_1d_00610101_00610130_ug_c_SO_JET.nc",
              "MINT_1d_00610101_00610130_vg_c_SO_JET.nc"]
#     # print(nemo_files)

#     # Assign current date to next date for next loop
#     current_date = next_date

#     # open dataset using xnemogcm
nemo_paths = [directory + f for f in nemo_files]
mask_path = [directory + mask_file]
# ds = open_nemo_and_domain_cfg(nemo_files=nemo_paths,
                            #   domcfg_files=mask_path,
                            #  )
mask = open_domain_cfg(files=mask_path)
dsU = xr.open_dataset(nemo_paths[0])
dsV = xr.open_dataset(nemo_paths[1])

logger.info('Dataset is: %s', dsU)

# Need west and south scale factor convention for irregular filter on tripolar grid
# in this case, all scale factor dimensions have been renamed to x_c, despite still being on x_f,...
dxw = mask.e1u.roll(x_f=-1, roll_coords=False)  # x-spacing centered at western T-cell edge in m
dyw = mask.e2u.roll(x_f=-1, roll_coords=False)  # y-spacing centered at western T-cell edge in m
dxs = mask.e1v.roll(y_f=-1, roll_coords=False)  # x-spacing centered at southern T-cell edge in m
dys = mask.e2v.roll(y_f=-1, roll_coords=False)  # y-spacing centered at southern T-cell edge in m

wet_mask = mask.tmask.isel(z_c=0)
dxw = dxw.swap_dims({"x_f": "x_c"})
dyw = dyw.swap_dims({"x_f": "x_c"})
dxs = dxs.swap_dims({"y_f": "y_c"})
dys = dys.swap_dims({"y_f": "y_c"})
area = mask.e1t*mask.e2t

# find minimum grid spacing
dx_min = min(mask.e1t.where(mask.tmask.isel(z_c=0)).min(), mask.e2t.where(mask.tmask.isel(z_c=0)).min())
dx_min = dx_min.values

# now interpolate velocities to cell centre
grid = xgcm.Grid(mask, metrics=get_metrics(mask))

bd = {'boundary': 'extend'}

# interpolate velocity to t-grid point
uo_c = grid.interp(dsU.ug, 'X', **bd)
vo_c = grid.interp(dsV.vg, 'Y', **bd)

# set lengthscales
# Define length scales in kilometers for coarse graining.
# Multiple ranges are used to provide finer resolution at small scales and coarser at large scales.
length_ranges_km = [
    np.arange(np.ceil(dx_min*1e-3), 100, 5),      # 5 to 99 km, step 5 km
    np.arange(100, 250, 10),    # 100 to 249 km, step 10 km
    np.arange(250, 600, 25),   # 250 to 599 km, step 25 km
    np.arange(600, 1000, 50)   # 600 to 950 km, step 50 km
]
L = np.concatenate(length_ranges_km) * 1000  # Convert to meters

# set empty cumulative EKE (domain average)
ceke = np.zeros((30, np.size(L)))

# set a loop through lengthscales (in m)
for i in range(np.size(L)):
# for i in range(1):

    logger.info(f'Calculating ceke at lengthscale {L[i]}')

    # set up the filter
    specs = {
            'filter_scale': L[i],
            'dx_min': dx_min,
            'filter_shape': gcm_filters.FilterShape.GAUSSIAN
        }

    filter_irregular_with_land = gcm_filters.Filter(
        **specs,
        grid_type=gcm_filters.GridType.IRREGULAR_WITH_LAND,
        grid_vars={
            'wet_mask': wet_mask, 
            'dxw': dxw, 'dyw': dyw, 'dxs': dxs, 'dys': dys, 'area': area, 
            'kappa_w': xr.ones_like(dxw), 'kappa_s': xr.ones_like(dxs),
        }
    ) # need to vary kappa using the deformation radius.
    filter_irregular_with_land

    # Ensure that velocity has same coordinates
    ds_tmp = xr.Dataset()  # temporary dataset with swapped dimensions

    uoce_mean = filter_irregular_with_land.apply(uo_c, dims=['y_c', 'x_c'])
    voce_mean = filter_irregular_with_land.apply(vo_c, dims=['y_c', 'x_c'])


    # velocities already on centre grid from initial regridding

    # now compute coarse eddy energy at lengthscale L
    ds_tmp['coarse_ke'] = ( 0.5 * ( abs( uoce_mean**2 + 
                            voce_mean**2 ) ) )
    
    logger.info('Dataset is: %s', ds_tmp)

    # cumulative eke
    ceke[:, i] = ds_tmp['coarse_ke'].mean(dim=['x_c', 'y_c'])

    # print(L[i], ceke[i, 1])
    logger.info(f'Cumulative ke is: %s', ceke[:, i])

# difference ceke wrt to lengthscale
E = np.zeros((30, np.size(L)-1))
for j in range(30):
    for i in range(np.size(L)-1):
        E[j, i] = -(L[i]**2 / 10**6) * (ceke[j, i+1]-ceke[j, i]) / (L[i+1]-L[i])
        logger.info(f'E at lengthscale {L[i]} is {E[j, i]}')

ds_tmp1 = xr.Dataset(
    data_vars={
        'E': (["t", "l"], E,
                {"standard_name": "Kinetic energy spectrum", "units": "m^3/s^2"}),
    },
    coords={
        'l': (['l'], L[:-1], 
                {"standard_name": "Lengthscale", "units": "m"}),
    },
    attrs={
        "name": "Energy spectrum",
        "description": f"Kinetic energy spectrum in EXP4",
    },
)

# expand dimension to include time
# ds_tmp1 = ds_tmp1.expand_dims(dim="t", axis=0)

logger.info('dataset is: %s', ds_tmp1)

output_filename = f"ke_spectrum_exp4.nc"
logger.info(f'save data to filename {output_filename}')
ds_tmp1.to_netcdf(directory + output_filename)

#     ds.close() 