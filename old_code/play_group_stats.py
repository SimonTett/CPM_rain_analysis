# play with the group stats.
import cftime
import matplotlib.pyplot as plt
import numpy as np
import xarray
import CPMlib

rgn_all=dict(longitude=slice(357.5,361.0),latitude=slice(1.5,7.5))
rgn = CPMlib.stonehaven_rgn
topog= xarray.load_dataset(CPMlib.CPM_dir/'orog_land-cpm_BI_2.2km.nc',decode_times=False)

topog= topog.ht.sel(rgn_all).rename(dict(longitude='grid_longitude',latitude='grid_latitude')).squeeze()
t=topog.coarsen(grid_longitude=2,grid_latitude=2,boundary='trim').min().sel(rgn)
tmn = topog.coarsen(grid_longitude=2,grid_latitude=2,boundary='trim').mean().sel(rgn)
paths = sorted((CPMlib.CPM_dir/'CPM01/').glob('*19*.nc'))
ds=xarray.open_mfdataset(paths)
ds=ds.sel(time=(ds.time.dt.season=='JJA')) # get summer only
ds = ds.sel(rgn).load() # reduce to rgn of interest
# check rel errors < 1.06
max_rel_lon = float((np.abs(t.grid_longitude.values/ds.grid_longitude.values-1)).max())
max_rel_lat = float((np.abs(t.grid_latitude.values/ds.grid_latitude.values-1)).max())

if max_rel_lon > 1e-6:
    raise ValueError(f"Lon Grids differ by more than 1e-6. Max = {max_rel_lon}")

if max_rel_lat > 1e-6:
    raise ValueError(f"Lat Grids differ by more than 1e-6. Max = {max_rel_lat}")
t=t.interp(grid_longitude=ds.grid_longitude,grid_latitude=ds.grid_latitude) # grids differ by tiny amount.
tmn=tmn.interp(grid_longitude=ds.grid_longitude,grid_latitude=ds.grid_latitude)
#ds = xarray.where(t > 1,ds,np.NAN)
grp = CPMlib.discretise(ds.seasonalMaxTime)
grp = xarray.where((t>1) & (tmn<300.), grp,0).rename("EventTime") # land < 300m
dd=CPMlib.event_stats(ds.seasonalMax,ds.seasonalMaxTime,grp).sel(EventTime=slice(1,None))

# restrict to "large" events --13 cells or more ~ 250 km^2
large= dd.count_cells > 13
dd_large = dd.sel(EventTime=large)

## Now plot where the events occur.
fig,ax = plt.subplots(nrows=1,ncols=1,figsize=[6,8],clear=True,
                      num='event_medians',subplot_kw=dict(projection=CPMlib.projRot))
ax.set_extent([rgn['grid_longitude'].start,rgn['grid_longitude'].stop,
              rgn['grid_latitude'].start,rgn['grid_latitude'].stop],crs=CPMlib.projRot)
ax.scatter(dd_large.x,dd_large.y,transform=CPMlib.projRot,
           s=60,c=dd_large.quant_precip.sel(quant=0.9),norm='log',marker='o',alpha=0.5)
c=tuple(CPMlib.stonehaven.values())
ax.plot(c[0],c[1],marker='*',ms=12,transform=CPMlib.projRot)

# getting some co-ords over sea when do all of scotland analysis. Regional analysis is not prone to thsi.
L=(t.interp(grid_latitude=dd_large.x,grid_longitude=dd_large.y) < 1)
dd_sea=dd_large.sel(EventTime=L)
# plot rain for each group.
# plot rains

# for indx,g in enumerate(dd_sea.group):
#     time = cftime.num2date(g,CPMlib.time_unit,calendar='360_day')
#     xarray.where(grp == g,indx,np.nan).max('time').plot(transform=CPMlib.projRot,add_colorbar=False)
#ax.scatter(dd_sea.lon,dd_sea.lat,transform=CPMlib.projRot,s=60,c=np.arange(0,len(dd_sea.lon)),norm='log',marker='*')

ax.coastlines()
fig.show()
