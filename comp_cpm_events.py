# COmpute simulated events
# fit the CPM rainfall data.
import commonLib
import numpy as np
import xarray
import CPMlib
import CPM_rainlib

my_logger = CPM_rainlib.logger
commonLib.init_log(my_logger, level='INFO')
# first get in CET -- our covariate

obs_cet = commonLib.read_cet()  # read in the obs CET
obs_cet_jja = obs_cet.sel(time=(obs_cet.time.dt.season == 'JJA'))

cpm_cet = xarray.load_dataset(CPM_rainlib.dataDir / "CPM_ts/cet_tas.nc")
cpm_cet = cpm_cet.tas.resample(time='QS-DEC').mean()  # resample to seasonal means
cpm_cet_jja = cpm_cet.where(cpm_cet.time.dt.season == 'JJA', drop=True)  # and pull out summer.

# Load up CPM extreme rainfall data, select to region of interest and mask

rgn = CPMlib.carmont_rgn
filtered = False
if filtered:
    paths = sorted(CPMlib.CPM_filt_dir.glob('CPM*/*11_30_23.nc'))
    outpath = CPMlib.CPM_filt_dir / "CPM_filter_all_events.nc"
else:
    paths = sorted(CPMlib.CPM_dir.glob('CPM*/*.nc'))
    outpath = CPMlib.CPM_dir / "CPM_all_events.nc"
my_logger.info(f"Opening data for {len(paths)} files")
ds = xarray.open_mfdataset(paths, parallel=True)
ds = ds.where(ds.time.dt.season == 'JJA', drop=True)  # get summer only
ds = ds.sel(**rgn).load()  # reduce to rgn of interest
CPM_rainlib.fix_coords(ds)
topog = xarray.load_dataarray(CPM_rainlib.dataDir / 'cpm_topog_fix_c2.nc').sel(**rgn)
my_logger.info("Loaded data")
## iterate over ensemble_member and then stack them at the end...
grped_list = []
for ensemble in ds.ensemble_member:
    my_logger.info(f"Processing ensemble: {int(ensemble)}")
    dataset = ds.sel(ensemble_member=ensemble) # extract the ensemble
    mxTime = dataset.seasonalMaxTime.squeeze(drop=True).load()
    grp = CPMlib.discretise(mxTime).where(topog > 0, 0).rename('EventTime')
    # remove unwanted co-ords.
    coords_to_drop = [c for c in mxTime.coords if c not in mxTime.dims]
    # add on t & surface.
    coords_to_drop += ['t', 'surface']
    grp = grp.drop_vars(coords_to_drop, errors='ignore')
    my_logger.info(f"Computed grouping: {int(ensemble)}")
    dd_lst = []
    for roll in dataset['rolling'].values:
        dd = CPM_rainlib.event_stats(dataset.seasonalMax.sel(rolling=roll).load(),
                                     mxTime.sel(rolling=roll),
                                     grp.sel(rolling=roll)
                                     ).sel(EventTime=slice(1, None))
        event_time_values = np.arange(0, len(dd.EventTime))
        dd = dd.assign_coords(rolling=roll, EventTime=event_time_values)
        # get the CPM summer mean CET out and force its time's to be the same.
        tc = np.array([f"{int(y)}-06-01" for y in dd.t.isel(quantv=0).dt.year])
        cet_extreme_times = cpm_cet_jja.sel(ensemble_member=ensemble).interp(time=tc).rename(dict(time='EventTime'))
        # convert EventTime to an index.
        cet_extreme_times = cet_extreme_times.assign_coords(rolling=roll, EventTime=event_time_values)
        dd['CET'] = cet_extreme_times  # add in the CET data.
        dd_lst.append(dd)
        # add in hts
        ht = topog.sel(grid_longitude=dd.x, grid_latitude=dd.y)
        # drop unnneded coords
        coords_to_drop = [c for c in ht.coords if c not in ht.dims]
        ht = ht.drop_vars(coords_to_drop)
        dd['height'] = ht
        my_logger.info(f"Processed rolling: {roll}")
    dd = xarray.concat(dd_lst, dim='rolling')
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
dataset.to_netcdf(outpath)
