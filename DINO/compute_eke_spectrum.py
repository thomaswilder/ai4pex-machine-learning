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

mode = 'exp16'
region = 'SO_JET'

directory=f"/gws/nopw/j04/ai4pex/twilder/NEMO_data/DINO/EXP16/features_take2/{region}/coarsened_data/"
mask_file=f"../mesh_mask_{mode}_{region}.nc"

# set nemo filename using dates
if mode == "exp4":
    nemo_files = [f"MINT_1d_00610101_00610130_uo_cg_{region}.nc",
                  f"MINT_1d_00610101_00610130_vo_cg_{region}.nc"]
elif mode == "exp16":
    nemo_files = [f"../../production_take2/{region}/MINT_1d_00610101_00610130_grid_U_{region}.nc",
                  f"../../production_take2/{region}/MINT_1d_00610101_00610130_grid_V_{region}.nc"]

#     # open dataset using xnemogcm
nemo_paths = [directory + f for f in nemo_files]
mask_path = [directory + mask_file]
if mode == 'exp16':
    from xnemogcm import open_nemo_and_domain_cfg
    ds = open_nemo_and_domain_cfg(nemo_files=nemo_paths,
                                  domcfg_files=mask_path,
    )
elif mode == 'exp4':
    mask = open_domain_cfg(files=mask_path)
    dsU = xr.open_dataset(nemo_paths[0])
    dsV = xr.open_dataset(nemo_paths[1])

    ds = xr.merge([dsU, dsV])
    ds = ds.assign_coords(mask.coords)
    for var in ['e1u', 'e2u', 'e1v', 'e2v', 'e1t', 'e2t', 'tmask']:
        if var in mask:
            ds[var] = mask[var]


logger.info('Dataset is: %s', ds)

# Need west and south scale factor convention for irregular filter on tripolar grid
# in this case, all scale factor dimensions have been renamed to x_c, despite still being on x_f,...
dxw = ds.e1u.roll(x_f=-1, roll_coords=False)  # x-spacing centered at western T-cell edge in m
dyw = ds.e2u.roll(x_f=-1, roll_coords=False)  # y-spacing centered at western T-cell edge in m
dxs = ds.e1v.roll(y_f=-1, roll_coords=False)  # x-spacing centered at southern T-cell edge in m
dys = ds.e2v.roll(y_f=-1, roll_coords=False)  # y-spacing centered at southern T-cell edge in m

wet_mask = ds.tmask.isel(z_c=0)
dxw = dxw.swap_dims({"x_f": "x_c"})
dyw = dyw.swap_dims({"x_f": "x_c"})
dxs = dxs.swap_dims({"y_f": "y_c"})
dys = dys.swap_dims({"y_f": "y_c"})
area = ds.e1t*ds.e2t

# find minimum grid spacing
dx_min = min(ds.e1t.where(ds.tmask.isel(z_c=0)).min(), 
             ds.e2t.where(ds.tmask.isel(z_c=0)).min())
dx_min = dx_min.values

# now interpolate velocities to cell centre
grid = xgcm.Grid(ds, metrics=get_metrics(ds))

bd = {'boundary': 'extend'}

# interpolate velocity to t-grid point
uo_c = grid.interp(ds.uo, 'X', **bd)
vo_c = grid.interp(ds.vo, 'Y', **bd)

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

    # subset to avoid boundary artefacts
    if mode == 'exp4':
        ds_tmp = ds_tmp.isel(x_c=slice(3, 43), y_c=slice(7, 47)) 
        area_subset = area.isel(x_c=slice(3, 43), y_c=slice(7, 47))

        # cumulative eke
        ceke[:, i] = (ds_tmp['coarse_ke']*area_subset).sum(dim=['x_c', 'y_c']) \
                        / area_subset.sum()
        
    elif mode == 'exp16':
        # cumulative eke
        ceke[:, i] = (ds_tmp['coarse_ke']*area).sum(dim=['x_c', 'y_c']) \
                        / area.sum()

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
                {"standard_name": "Kinetic energy spectrum", "units": "[m$^3$/s$^2$]"}),
    },
    coords={
        'l': (['l'], L[:-1], 
                {"standard_name": "Lengthscale", "units": "[m]"}),
    },
    attrs={
        "name": "Energy spectrum",
        "description": f"Kinetic energy spectrum in {mode}",
    },
)

logger.info('dataset is: %s', ds_tmp1)

if mode == 'exp4':
    save_dir = directory
elif mode == 'exp16':
    save_dir = directory + '../'

output_filename = f"ke_spectrum_{mode}.nc"
logger.info(f'save data to filename {output_filename}')
ds_tmp1.to_netcdf(save_dir + output_filename)

#     ds.close() 