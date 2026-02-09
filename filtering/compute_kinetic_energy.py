#!/usr/bin/python3

'''
    Description: Computes eddy kinetic energy using GCM-Filters.
        Currently in CPU mode.
'''

import glob
import gcm_filters
import xarray as xr
from xnemogcm import open_domain_cfg, get_metrics, open_nemo_and_domain_cfg
import xgcm
from datetime import datetime
from dateutil.relativedelta import relativedelta
import numpy as np

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(filename='compute_ke_dino.log', level=logging.INFO, filemode='w')

logger.info('Begin...')

# ------------ set parameters ---------------- #

region = 'SO_JET'

directory = f'/gws/nopw/j04/ai4pex/twilder/NEMO_data/DINO/EXP16/production_take2/{region}/'
mask_path = [directory + f'../../features_take2/{region}/mesh_mask_exp16_surface_{region}.nc']

# Initial date string
start_date_init_str = "00610201"

# End date string
end_date_init_str = "00730101"

# scale deformation radius
ld_scaling = 3
# -------------------------------------------- #

# -------------------------------------------- #
# get max deformation radius in metres
#TODO need to time average Ld and load this in
directory_ld = f'/gws/nopw/j04/ai4pex/twilder/NEMO_data/DINO/EXP16/features_take2/{region}/'
ld = xr.open_dataset(directory_ld + f'MINT_1d_00610101_00721230_Ld_{region}.nc')

# #! temporary until Ld is complete
# ld_mean = ld.Ld.mean(dim='time_counter', skipna=True)

# del ld

# # zero out equator region
# ld = xr.where(((ld_mean.gphit>5) | (ld_mean.gphit<-5)), ld_mean, 0.0)

# scale Ld
ld = ld_scaling * ld.Ld

# get max Ld (scaling)
Lmax = 1000*ld.max(dim=['x_c', 'y_c'], skipna=True).values

# set kappa as a scale of deformation radius
kappa = (1000*ld)**2 / Lmax**2
# kappa = kappa.swap_dims({"x": "x_c"})
# kappa = kappa.swap_dims({"y": "y_c"})
# -------------------------------------------- #

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
    nemo_files = [f'MINT_1d_{date_init}_*_grid_U_{region}.nc',
                  f'MINT_1d_{date_init}_*_grid_V_{region}.nc']
    print(nemo_files)

    nemo_paths = [glob.glob(directory + f) for f in nemo_files]

    logger.info('nemo_paths are: %s', nemo_paths)

    # extract date_end from nemo_paths
    filename = nemo_paths[0][0].split('/')[-1]
    date_end = filename.split('_')[3]

    nemo_files = [nemo_paths[0][0].split('/')[-1], nemo_paths[1][0].split('/')[-1]]

    # open dataset using xnemogcm
    nemo_paths = [directory + f for f in nemo_files]
    ds = open_nemo_and_domain_cfg(nemo_files=nemo_paths,
                                  domcfg_files=mask_path)
    
    logger.info('Dataset is: %s', ds)

    # nemo_paths = [directory + f for f in nemo_files]
    # domcfg = open_domain_cfg(files=mask_path)
    # dataU = xr.open_dataset(nemo_paths[0])
    # dataV = xr.open_dataset(nemo_paths[1])
    
    #! just for testing
    # subset ds for first 2 time steps
    # ds = ds.isel(t=slice(0,2))
    # dataU = dataU.isel(t=slice(0,2))
    # dataV = dataV.isel(t=slice(0,2))
    
    # Rechunk the dataset
    ds = ds.chunk(dict(y_c=-1))
    ds = ds.chunk(dict(x_f=-1))
    ds = ds.chunk(dict(y_f=-1))
    ds = ds.chunk(dict(x_c=-1))

    # Need west and south scale factor convention for irregular filter on tripolar grid
    dxw = ds.e1u.roll(x_f=-1, roll_coords=False)  # x-spacing centered at western T-cell edge in m
    dyw = ds.e2u.roll(x_f=-1, roll_coords=False)  # y-spacing centered at western T-cell edge in m
    dxs = ds.e1v.roll(y_f=-1, roll_coords=False)  # x-spacing centered at southern T-cell edge in m
    dys = ds.e2v.roll(y_f=-1, roll_coords=False)  # y-spacing centered at southern T-cell edge in m

    # Ensure that coordinates are the same
    wet_mask = ds.tmask.isel(z_c=0)
    dxw = dxw.swap_dims({"x_f": "x_c"})
    dyw = dyw.swap_dims({"x_f": "x_c"})
    dxs = dxs.swap_dims({"y_f": "y_c"})
    dys = dys.swap_dims({"y_f": "y_c"})
    area = ds.e1t * ds.e2t

    # find minimum grid spacing
    dx_min = min(ds.e1t.where(ds.tmask.isel(z_c=0)).min(), \
                  ds.e2t.where(ds.tmask.isel(z_c=0)).min())
    dx_min = dx_min.values

    # set up the filter
    specs = {
            'filter_scale': Lmax,
            'dx_min': dx_min,
            'filter_shape': gcm_filters.FilterShape.GAUSSIAN
        }

    filter_irregular_with_land = gcm_filters.Filter(
        **specs,
        grid_type=gcm_filters.GridType.IRREGULAR_WITH_LAND,
        grid_vars={
            'wet_mask': wet_mask, 
            'dxw': dxw, 'dyw': dyw, 'dxs': dxs, 'dys': dys, 'area': area, 
            'kappa_w': kappa, 'kappa_s': kappa,
        }
    ) # need to vary kappa using the deformation radius.
    filter_irregular_with_land

    # set up xgcm grid
    # grid = xgcm.Grid(
    #             ds,
    #             metrics=get_metrics(ds),
                # periodic={'X': False, 'Y': False},
                # boundary={'Y': 'fill', 'X': 'extend'},
                # )
    
    grid = xgcm.Grid(
                ds,
                metrics=get_metrics(ds),
    )
    
    # interpolate velocity to t-grid point
    # ds_tmp = xr.Dataset()
    logger.info('Interpolating velocities to t-grid')
    uo_c = grid.interp(ds.uo.isel(z_c=0), 'X', boundary='extend').persist()
    vo_c = grid.interp(ds.vo.isel(z_c=0), 'Y', boundary='extend').persist()

    # logger.info('Writing interpolated velocity to temp file')
    # ds_tmp.to_netcdf(directory + "temp_velocity_on_tgrid.nc")

    # del ds_tmp

    # logger.info('Loading interpolated velocity from temp file')
    # vels_c = xr.open_dataset(directory + "temp_velocity_on_tgrid.nc",
    #                          chunks={'y_c': 400, 'x_c': 400})


    # compute mean components of velocity
    # ds_tmp = xr.Dataset()  # temporary dataset with swapped dimensions
    logger.info('Computing mean velocities using GCM-Filters')
    uoce_mean = filter_irregular_with_land.apply(uo_c, 
                                                dims=['y_c', 'x_c']).persist()
    voce_mean = filter_irregular_with_land.apply(vo_c, 
                                                dims=['y_c', 'x_c']).persist()

    # logger.info('Writing mean velocities to temp file')
    # ds_tmp.to_netcdf(directory + "temp_mean_velocity.nc")

    # del ds_tmp

    # following definition of EKE by Buzzicotti et al., (2023)
    u_sq_c = abs(uo_c**2+vo_c**2)
    # filter ds_tmp['u_sq_c']
    # ds_tmp = xr.Dataset()  # temporary dataset
    logger.info('Computing filtered velocity squared using GCM-Filters')
    u_sq_c_filt = filter_irregular_with_land.apply(u_sq_c, 
                                                dims=['y_c', 'x_c']).persist()
    
    # logger.info('Writing filtered velocity squared to temp file')
    # ds_tmp.to_netcdf(directory + "temp_filtered_velocity_squared.nc")

    # del ds_tmp, u_sq_c

    # logger.info('Loading filtered velocity squared from temp file')
    # u_sq_c_filt = xr.open_dataset(directory + "temp_filtered_velocity_squared.nc",
    #                               chunks={'y_c': 400, 'x_c': 400})
    # logger.info('Loading mean velocities from temp file')
    # vel_mean = xr.open_dataset(directory + "temp_mean_velocity.nc",
    #                            chunks={'y_c': 400, 'x_c': 400})
    
    # now compute eddy energy
    bare_ke = 0.5 * ( uo_c**2 + vo_c**2 )
    coarse_ke = 0.5 * ( abs( uoce_mean**2 
                            + voce_mean**2 ) )
    fine_ke = 0.5 * ( u_sq_c_filt 
                        - abs( uoce_mean**2 + voce_mean**2 ) )

    logger.info('bare_ke is: %s', bare_ke)
    logger.info('coarse_ke is: %s', coarse_ke)
    logger.info('fine_ke is: %s', fine_ke)

    # create dataset for xnemo readable
    ds_eke = xr.Dataset(
        data_vars={
            'bare_ke': (["t", "y_c", "x_c"], 
                        bare_ke.values),
            'coarse_ke': (["t", "y_c", "x_c"], 
                        coarse_ke.values),
            'fine_ke': (["t", "y_c", "x_c"], 
                        fine_ke.values),
        },
        coords={
            "t": (["t"], ds.t.values,
                        ds.t.attrs),
            "gphit": (["y_c", "x_c"], ds.gphit.values, 
                      {"standard_name": "Latitude", "units": "degrees_north"}),
            "glamt": (["y_c", "x_c"], ds.glamt.values, 
                      {"standard_name": "Longitude","units": "degrees_east"}),
        },
        attrs={
            "name": "NEMO dataset",
            "description": "Contains eddy kinetic energy -> ocean T grid variables",
        },
    )

    logger.info('ds_eke is: %s', ds_eke)


    # extract time counter bounds from original file
    # ref_dir = f'/gws/nopw/j04/ai4pex/twilder/NEMO_data/DINO/EXP16/production_take2/{region}/'
    # ref = xr.open_dataset(ref_dir + 
    #                       f'MINT_1d_{date_init}_{date_end}_grid_T_{region}.nc')
    
    # ds_eke["time_counter_bounds"] = ref["time_counter_bounds"]

    # save to netcdf
    output_file = f'MINT_1d_{date_init}_{date_end}_ke_{region}.nc'

    save_directory = f'/gws/nopw/j04/ai4pex/twilder/NEMO_data/DINO/EXP16/features_take2/{region}//'
    logger.info('Saving kinetic energy to: %s', save_directory + output_file)

    ds_eke.to_netcdf(save_directory + output_file)

logger.info('End!')