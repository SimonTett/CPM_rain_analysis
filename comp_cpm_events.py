# COmpute simulated events
# fit the CPM rainfall data.
import typing

import pandas as pd

import commonLib
import numpy as np
import xarray
import CPMlib
import CPM_rainlib
import dask

my_logger = CPM_rainlib.logger
commonLib.init_log(my_logger, level='INFO')
test= False
filtered = True # if True use filtered data!
# first get in CET -- our covariate

obs_cet = commonLib.read_cet()  # read in the obs CET
obs_cet_jja = obs_cet.sel(time=(obs_cet.time.dt.season == 'JJA'))

cpm_cet = xarray.load_dataset(CPM_rainlib.dataDir / "CPM_ts/cet_tas.nc")
cpm_cet = cpm_cet.tas.resample(time='QS-DEC').mean()  # resample to seasonal means
cpm_cet_jja = cpm_cet.where(cpm_cet.time.dt.season == 'JJA', drop=True)  # and pull out summer.
# Load up CPM extreme rainfall data, select to region of interest and mask
# test
if test:
    rgn = {'grid_longitude': slice(359.75, 360.25),'grid_latitude': slice(4.25, 4.75)}
    my_logger.info(f'In test mode. Using small rgn')
else:
    rgn = CPMlib.carmont_rgn

if filtered:
    paths = sorted(CPMlib.CPM_filt_dir.glob('CPM*/*11_30_23.nc'))
    outpath = CPMlib.CPM_filt_dir / "CPM_filter_all_events.nc"
else:
    paths = sorted(CPMlib.CPM_dir.glob('CPM*/*.nc'))
    outpath = CPMlib.CPM_dir / "CPM_all_events.nc"
my_logger.info(f"Opening data for {len(paths)} files. ")
with dask.config.set(**{'array.slicing.split_large_chunks': True}):
    ds = xarray.open_mfdataset(paths, parallel=True, drop_variables=['month_number', 'year', 'yyyymmddhh'], chunks={})
my_logger.info("Opened files")
ds = ds.where(ds.time.dt.season == 'JJA', drop=True)  # get summer only
CPM_rainlib.fix_coords(ds)
ds = ds.sel(**rgn)  # reduce to rgn of interest

topog = xarray.load_dataarray(CPM_rainlib.dataDir / 'cpm_topog_fix_c2.nc')
CPM_rainlib.fix_coords(topog)
topog= topog.sel(**rgn)
## iterate over ensemble_member and then stack them at the end...
grped_list = []

if test:
    ensemble_list=[1,5]
else:
    ensemble_list = ds.ensemble_member
for ensemble in ensemble_list:
    my_logger.info(f"Processing ensemble: {int(ensemble)}")
    dataset = ds.sel(ensemble_member=ensemble).load()  # extract the ensemble
    my_logger.info(f"Loaded ensemble: {int(ensemble)}")
    mxTime = dataset.seasonalMaxTime.squeeze(drop=True)
    grp = CPMlib.discretise(mxTime.where(topog > 0)).rename('EventTime')
    # remove unwanted co-ords.
    coords_to_drop = [c for c in mxTime.coords if c not in mxTime.dims]
    # add on t & surface.
    coords_to_drop += ['t', 'surface']
    grp = grp.drop_vars(coords_to_drop, errors='ignore')
    my_logger.info(f"Computed grouping for ensemble: {int(ensemble)}")
    dd = CPM_rainlib.comp_events(dataset.seasonalMax,dataset.seasonalMaxTime, grp, topog, cpm_cet_jja.sel(ensemble_member=ensemble),source='CPM')

    # remove unneded dimensions
    dims = list(dd.dims) + ['ensemble_member', 'ensemble_member_id']
    coords_to_drop = [c for c in dd.coords if c not in dims]
    dd = dd.drop_vars(coords_to_drop)
    grped_list.append(dd)

dataset = xarray.concat(grped_list, dim='ensemble_member')
# need to convert times
for var in ['t']:
    dataset[var] = CPMlib.time_convert(dataset[var])

my_logger.info(f"Writing events to {outpath}")
if not test:
    dataset.to_netcdf(outpath)
