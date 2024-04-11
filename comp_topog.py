# compute topography for different radar datasets
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
rgn = {k:slice(v-150e3,v+150e3) for k,v in CPMlib.carmont_drain_OSGB.items()}
# fix the rgn a bit!
rgn['projection_x_coordinate'] = slice(rgn['projection_x_coordinate'].start,
                                       rgn['projection_x_coordinate'].start+225e3)
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
topog = CPM_rainlib.read_90m_topog(region=rgn).drop_vars(['band','spatial_ref'])
my_logger.info(f"Loaded topography")
radar_top=dict()
for key,radar in radar_ds.items():
    delta = dx_dy(radar)
    npt = np.floor(delta/90.).astype(int)
    minp=int(0.5*npt[0]*npt[1])
    topog_avg = topog.rolling(projection_x_coordinate=npt[0],projection_y_coordinate=npt[1],
                              center=True,min_periods=minp).mean() # average up to rough resoln of radar
    radar_top[key] = topog_avg.interp_like(radar) # extract pts at radar grid.
    my_logger.info(f"Computed topography for {key}")
## save the data
for key,r_topog in radar_top.items():
    outpath = out_dir/f'topog_{key}.nc'
    r_topog.to_netcdf(outpath)
    my_logger.info(f"Saved topography for {key} to {outpath}")

## plot the topographies for inspection
nplots = len(radar_top)
nrows = int(np.floor(np.sqrt(nplots)))
ncols = int(np.ceil(nplots/nrows))
fig_topog,axes = plt.subplots(nrows=nrows,ncols=ncols,figsize=[11,7],clear=True,num='radar_topog',layout='constrained',
                              subplot_kw=dict(projection=CPMlib.projOSGB))
levels=[0, 50, 100, 200, 300, 400, 500, 600, 700, 800,1000,1200]
for (key,topog),ax in zip(radar_top.items(),axes.flat):
    ax.set_extent(rgn_extent,crs=CPMlib.projOSGB)
    cm=topog.plot(ax=ax,add_colorbar=False,cmap='YlOrBr',levels=levels)
    ax.set_title(key)
    CPM_rainlib.std_decorators(ax)
fig_topog.colorbar(cm,ax=axes,label='height (m)',**CPMlib.kw_colorbar)
fig_topog.show()

