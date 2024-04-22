# compute topography for carmont region different radar datasets and CPM dataset.
import CPM_rainlib
import CPMlib
import xarray
import commonLib
import matplotlib.pyplot as plt
import numpy as np
import pathlib

def dx_dy(radar:xarray.Dataset) -> np.ndarray:
    """
    Compute the mean dx, dy for a radar dataset
    :param radar: radar dataset -- only use the coords
    :return: 2 element numpy array . element 0 = mean dx,element 1 = mean dy
    """
    dx = radar.projection_x_coordinate.diff('projection_x_coordinate')
    dy = radar.projection_y_coordinate.diff('projection_y_coordinate')
    return np.array([float(dx.mean()),float(dy.mean())])

my_logger = CPM_rainlib.logger
commonLib.init_log(my_logger,level='INFO')
out_dir = CPMlib.radar_dir / 'topography'
out_dir.mkdir(exist_ok=True,parents=True)
# get in an example field from the radar for each radar type
rgn = {k:slice(v-75e3,v+75e3) for k,v in CPMlib.carmont_drain_OSGB.items()}

rgn_extent = []
for v in rgn.values():
    rgn_extent.extend([v.start,v.stop])
files = (CPMlib.radar_dir/'summary').glob('summary_2008*.nc')
radar_ds=dict()

for path in files:
    name = path.stem.replace('summary_2008_','')
    radar = xarray.open_dataset(path).isel(time=0).sel(**rgn)
    radar_ds[name] = radar
my_logger.info(f"Loaded radar data")
region = CPMlib.carmont_rgn_OSGB
topog_cpm = CPM_rainlib.read_90m_topog(region=rgn).drop_vars(['band','spatial_ref'])
my_logger.info(f"Loaded topography")
radar_top=dict()
for key,radar in radar_ds.items():
    delta = dx_dy(radar)
    npt = np.floor(delta/90.).astype(int)
    minp=int(0.5*npt[0]*npt[1])
    topog_avg = topog_cpm.rolling(projection_x_coordinate=npt[0],projection_y_coordinate=npt[1],
                              center=True,min_periods=minp).mean().sel(**rgn).load() # average up to rough resoln of radar
    radar_top[key] = topog_avg.interp_like(radar) # extract pts at radar grid.
    my_logger.info(f"Computed topography for {key}")
## save the radar data
for key,r_topog in radar_top.items():
    outpath = out_dir/f'topog_{key}.nc'
    r_topog.to_netcdf(outpath)
    my_logger.info(f"Saved topography for {key} to {outpath}")

## compute the CPM topography.
topog_cpm = xarray.load_dataset(CPM_rainlib.dataDir / 'orog_land-cpm_BI_2.2km.nc', decode_times=False).ht.squeeze()
topog_cpm = topog_cpm.drop_vars(['t', 'surface'], errors='ignore')
topog_cpm = topog_cpm.rename(dict(longitude='grid_longitude', latitude='grid_latitude'))[1:, 1:]
topog_cpm = topog_cpm.coarsen(grid_longitude=2, grid_latitude=2, boundary='pad').mean()
CPM_rainlib.fix_coords(topog_cpm)
topog_cpm = topog_cpm.sel(**CPMlib.carmont_rgn)
outpath= CPM_rainlib.dataDir / 'cpm_topog_fix_c2.nc'
topog_cpm.to_netcdf(outpath)
my_logger.info(f"Saved topography for CPM to {outpath}") # save it
## plot the topographies for inspection
# CPM first
levels=[0, 50, 100, 200, 300, 400, 500, 600, 700, 800,1000,1200]
fig_cpm_topog,ax = plt.subplots(nrows=1,ncols=1,num='CPM Topog',
                                   figsize=[5,5],clear=True,subplot_kw=dict(projection=CPMlib.projRot))
ax.set_extent(rgn_extent,crs=CPMlib.projOSGB)
topog_cpm.plot(ax=ax,add_colorbar=True,cmap='YlOrBr',levels=levels,label='height (m)',cbar_kwargs=CPMlib.kw_colorbar,transform=CPMlib.projRot)
ax.set_title('CPM')
CPM_rainlib.std_decorators(ax)
CPMlib.plot_carmont(ax)
fig_cpm_topog.show()

nplots = len(radar_top)
nrows = int(np.floor(np.sqrt(nplots)))
ncols = int(np.ceil(nplots/nrows))
fig_topog,axes = plt.subplots(nrows=nrows,ncols=ncols,figsize=[11,7],clear=True,num='Topog',layout='constrained',
                              subplot_kw=dict(projection=CPMlib.projOSGB))

for (key,topog),ax in zip(radar_top.items(),axes.flat):
    ax.set_extent(rgn_extent, crs=CPMlib.projOSGB)
    cm=topog.plot(ax=ax, add_colorbar=False, cmap='YlOrBr', levels=levels)
    ax.set_title(key)
    CPM_rainlib.std_decorators(ax)
    CPMlib.plot_carmont(ax)


fig_topog.colorbar(cm,ax=axes,label='height (m)',**CPMlib.kw_colorbar)
fig_topog.show()

