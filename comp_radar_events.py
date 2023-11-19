# analyse the 5km radar data near StoneHaven.
# need to have co-ords on OSGB grid (coz that is what the radar is).

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import CPMlib
import xarray
import edinburghRainLib
import numpy as np
import commonLib
import datetime

proj=ccrs.PlateCarree()
projGB=ccrs.OSGB()
v=tuple(CPMlib.stonehaven.values())
stonehaven = projGB.transform_point(*v,CPMlib.projRot)
var_names = ["projection_x_coordinate","projection_y_coordinate"]
stonehaven = dict(zip(var_names,stonehaven))
stonehaven_rgn={k:slice(v-75e3,v+75e3) for k,v in stonehaven.items()}
path=CPMlib.datadir/'summary_5km_1hr_scotland.nc'
rseasMskmax, mxTime=edinburghRainLib.get_radar_data(path,region=stonehaven_rgn,height_range=slice(1,300))
# get in hts
topog990m = edinburghRainLib.read_90m_topog(region=stonehaven_rgn, resample=55)
top_fit_grid = topog990m.interp_like(rseasMskmax.isel(time=0).squeeze())
grp = ((mxTime.dt.dayofyear-1)+mxTime.dt.year*1000).rename('EventTime')
# mask!
grp=xarray.where(~rseasMskmax.isnull(),grp,0).rename('EventTime')
radar_dataset=CPMlib.event_stats(rseasMskmax,mxTime,grp,'EventTime',source='radar').sel(EventTime=slice(1,None))

# get the  summer mean CET out and force its time's to be the same.
obs_cet=commonLib.read_cet() # read in the obs CET
obs_cet_jja = obs_cet.sel(time=(obs_cet.time.dt.season == 'JJA'))
tc = np.array([f"{int(y)}-06-01" for y in radar_dataset.t.isel(quantv=0).dt.year])
cet_extreme_times = obs_cet_jja.interp(time=tc).rename(dict(time='EventTime'))
# convert EventTime to an index.
etime=np.arange(0, len(radar_dataset.EventTime))
cet_extreme_times = cet_extreme_times.assign_coords(dict(EventTime=etime))
radar_dataset = radar_dataset.assign_coords(dict(EventTime=etime))
radar_dataset['CET'] = cet_extreme_times  # add in the CET data.

# add in hts
ht = top_fit_grid.sel(projection_x_coordinate=radar_dataset.x, projection_y_coordinate=radar_dataset.y)
# drop unnneded coords
coords_to_drop = [c for c in ht.coords if c not in ht.dims]
ht = ht.drop_vars(coords_to_drop)
radar_dataset['height'] = ht
# remove unneded dimensions
dims = list(radar_dataset.dims)
coords_to_drop = [c for c in radar_dataset.coords if c not in dims]
radar_dataset = radar_dataset.drop_vars(coords_to_drop)
# add some attributes
source_str=f'Processed NIMORD RADAR data to events using comp_radar_events.py on{datetime.datetime.now()}'
radar_dataset.attrs.update(source=source_str)

# need to convert times
#for var in ['t']:
#    radar_dataset[var]=CPMlib.time_convert(radar_dataset[var])

path = CPMlib.datadir/"radar_events.nc"
path.parent.mkdir(exist_ok=True)
radar_dataset.to_netcdf(path)
#fit_radar = gev_r.xarray_gev(dd_radar_large.quant_precip,dim='EventTime',verbose=True,file=save_file)
