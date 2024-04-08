# COmpute simulated events
# fit the CPM rainfall data.
import commonLib
import numpy as np
import xarray
import CPMlib
import CPM_rainlib
my_logger=CPM_rainlib.logger
commonLib.init_log(my_logger,level='INFO')
# first get in CET -- our covariate

obs_cet=commonLib.read_cet() # read in the obs CET
obs_cet_jja = obs_cet.sel(time=(obs_cet.time.dt.season == 'JJA'))

cpm_cet=xarray.load_dataset(CPM_rainlib.dataDir/"CPM_ts/cet_tas.nc")
cpm_cet=cpm_cet.tas.resample(time='QS-DEC').mean() # resample to seasonal means
cpm_cet_jja = cpm_cet.where(cpm_cet.time.dt.season=='JJA',drop=True) # and pull out summer.

# Load up CPM extreme rainfall data, select to region of interest and mask
rgn_all=dict(longitude=slice(357.5,361.0),latitude=slice(1.5,7.5)) # region for which extraction was done.
rgn = CPMlib.carmont_rgn
filtered=True
if filtered:
    paths = sorted(CPMlib.CPM_filt_dir.glob('CPM*/*11_30_23.nc'))
    outpath = CPMlib.CPM_filt_dir/"CPM_filter_all_events.nc"
else:
    paths = sorted(CPMlib.CPM_dir.glob('CPM*/*.nc'))
    outpath =  CPMlib.CPM_dir/"CPM_all_events.nc"
my_logger.info(f"Opening data for {len(paths)}")
ds=xarray.open_mfdataset(paths,parallel=True)
ds=ds.where(ds.time.dt.season=='JJA',drop=True) # get summer only
ds = ds.sel(**rgn) # reduce to rgn of interest
print("Opened data")
topog= xarray.load_dataset(CPM_rainlib.dataDir/'orog_land-cpm_BI_2.2km.nc',decode_times=False)
topog= topog.ht.sel(rgn_all).rename(dict(longitude='grid_longitude',latitude='grid_latitude')).squeeze()
t=topog.coarsen(grid_longitude=2,grid_latitude=2,boundary='trim').min().sel(rgn)
tmn = topog.coarsen(grid_longitude=2,grid_latitude=2,boundary='trim').mean().sel(rgn)
# check rel errors < 1.e-6
max_rel_lon = float((np.abs(t.grid_longitude.values/ds.grid_longitude.values-1)).max())
max_rel_lat = float((np.abs(t.grid_latitude.values/ds.grid_latitude.values-1)).max())

if max_rel_lon > 1e-6:
    raise ValueError(f"Lon Grids differ by more than 1e-6. Max = {max_rel_lon}")

if max_rel_lat > 1e-6:
    raise ValueError(f"Lat Grids differ by more than 1e-6. Max = {max_rel_lat}")
t=t.interp(grid_longitude=ds.grid_longitude,grid_latitude=ds.grid_latitude) # grids differ by tiny amount.
tmn=tmn.interp(grid_longitude=ds.grid_longitude,grid_latitude=ds.grid_latitude)

# iterate over ensemble_member and then stack them at the end...
grped_list=[]
for ensemble in ds.ensemble_member:
    my_logger.info(f"Processing ensemble: {int(ensemble)}")
    dataset=ds.sel(ensemble_member=ensemble) # extract the ensemble
    mxTime = dataset.seasonalMaxTime.squeeze(drop=True).load()
    grp = CPMlib.discretise(mxTime).where(t>1,0).rename('EventTime')
    # remove unwanted co-ords.
    coords_to_drop = [c for c in mxTime.coords if c not in mxTime.dims]
    # add on t & surface.
    coords_to_drop += ['t','surface']
    grp = grp.drop_vars(coords_to_drop,errors='ignore')
    my_logger.info(f"Computed grouping: {int(ensemble)}")
    dd_lst=[]
    for roll in dataset['rolling'].values:

        dd=CPM_rainlib.event_stats(dataset.seasonalMax.sel(rolling=roll).load(),
                                   mxTime.sel(rolling=roll),
                                   grp.sel(rolling=roll)).sel(EventTime=slice(1,None))
        event_time_values = np.arange(0, len(dd.EventTime))
        dd = dd.assign_coords(rolling=roll,EventTime=event_time_values)
        # get the CPM summer mean CET out and force its time's to be the same.
        tc = np.array([f"{int(y)}-06-01" for y in dd.t.isel(quantv=0).dt.year])
        cet_extreme_times = cpm_cet_jja.sel(ensemble_member=ensemble).interp(time=tc).rename(dict(time='EventTime'))
        # convert EventTime to an index.
        cet_extreme_times = cet_extreme_times.assign_coords(rolling=roll,EventTime=event_time_values)
        dd['CET'] = cet_extreme_times  # add in the CET data.
        dd_lst.append(dd)
        # add in hts
        ht = tmn.sel(grid_longitude=dd.x, grid_latitude=dd.y)
        # drop unnneded coords
        coords_to_drop = [c for c in ht.coords if c not in ht.dims]
        ht = ht.drop_vars(coords_to_drop)
        dd['height'] = ht
        my_logger.info(f"Processed rolling: {roll}")
    dd = xarray.concat(dd_lst,dim='rolling')
    # remove unneded dimensions
    dims = list(dd.dims) + ['ensemble_member','ensemble_member_id']
    coords_to_drop = [c for c in dd.coords if c not in dims]
    dd = dd.drop_vars(coords_to_drop)
    grped_list.append(dd)


dataset=xarray.concat(grped_list,dim='ensemble_member')
# need to convert times
for var in ['t']:
    dataset[var]=CPMlib.time_convert(dataset[var])

my_logger.info(f"Writing events to {outpath}")
dataset.to_netcdf(outpath)