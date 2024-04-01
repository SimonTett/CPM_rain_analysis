# compute the distribution parameters for GEV fit to radar data,
# This is done by sampling randomly from the quantiles of each distribution
# Then fit to that with weighting given by event size. Rise and repeat to get uncert.
import typing

import matplotlib.ticker
import xarray

import CPM_rainlib
import commonLib
from R_python import gev_r
import matplotlib.pyplot as plt
import numpy as np
import numpy.random
import CPMlib
import scipy.stats
import cartopy.crs as ccrs

def comp_fits(rng,radar_events,nsamps:int = 100) -> xarray.DataArray:
    n_events = radar_events.EventTime.shape[0]
    rand_index = rng.integers(len(radar_events.quantv), size=(n_events, nsamps))
    wt = radar_events.count_cells
    fits = []
    for indx in range(0, nsamps):  # iterate over random samples
        ind = xarray.DataArray(rand_index[:, indx],
                               coords=dict(EventTime=radar_events.coords['EventTime']))
        sample = radar_events.max_precip.isel(quantv=ind).drop_vars('quantv').assign_coords(sample=indx)
        fits.append(gev_r.xarray_gev(sample, dim='EventTime', weights=wt))

    fits = xarray.concat(fits, dim='sample')
    return fits

def print_rp(radar_fit_params:xarray.DataArray,
             radar_rain:float):
    rp_event=1.0/gev_r.xarray_gev_sf(radar_fit_params,radar_rain)
    mn_rp=rp_event.mean('sample')
    qv=rp_event.quantile([0.05,0.95],dim='sample')
    print(f'{name} Carmont rain {float(radar_rain):3.0f} mm/h '
      f'Return period: {float(mn_rp.values):3.0f}'
      f'({float(qv.isel(quantile=0).values):3.0f} '
      f'-{float(qv.isel(quantile=1).values):3.0f}) years')

def plot_rp(rp:np.ndarray,
            radar_fit_params:xarray.DataArray,
            ax,
            color:typing.Optional[str] = None,
            label:typing.Optional[str]=None):
    rv = gev_r.xarray_gev_isf(radar_fit_params, 1.0 / rp)
    rv_uncert = gev_r.xarray_gev_isf(radar_fit_params,1.0/rp)
    rv_q = rv_uncert.quantile([0.05,0.95],dim='sample') # compute 5-95% uncertainty
    rv.mean('sample').plot(x='return_period',ax=ax,color=color,label=label)
    ax.fill_between(y1=rv_q.isel(quantile=-1).values,
                y2=rv_q.isel(quantile=0).values,
                x=rv_q.return_period,alpha=0.5,color=color)
rain_aug2020=dict()
radar_fit=dict()
radar_rain=dict()
radar_fit_uncert=dict()
rng = numpy.random.default_rng(123456)
nsamps=100
projGB = ccrs.OSGB()
carmont = CPMlib.carmont_rgn_OSGB.copy()
carmont.update(time='2020-06-01')

for name,radar_ncfile,radar_event_ncfile in zip(
    ['5km','1km','1km_c4','1km_c5'][0],
    ["5km_summary_2004_2023.nc","summary_1km/1km_summary.nc","summary_1km/1km_c4_summary.nc","summary_1km/1km_c5_summary.nc"],
    ["5km_events_2008_2023.nc","radar_events_1km.nc","radar_events_1km_c4.nc","radar_events_1km_c5.nc"]

):

    path = CPMlib.radar_dir / radar_ncfile
    rain, mxTime, top_fit_grid = CPM_rainlib.get_radar_data(path, region= carmont,
                                                                   height_range=slice(50, None))
    rain = rain.where(mxTime.dt.strftime('%Y-%m-%d') == '2020-08-12').sel(time='2020-06')

    rain_aug2020[name]=rain

    radar_rain[name]=float(rain.sel(**CPMlib.carmont_drain_OSGB,method='nearest').
                            load().values)
    radar_events = xarray.load_dataset(CPMlib.radar_dir/'radar_events'/radar_event_ncfile)
    radar_fit[name]=comp_fits(rng,radar_events,nsamps= nsamps)
    print(f"Computed fits for {name}")
    ps = scipy.stats.multivariate_normal(mean=radar_fit[name].Parameters.mean('sample'), cov=radar_fit[name].Cov.mean('sample'))
    coords = dict(sample=np.arange(0, nsamps), parameter=['location', 'scale', 'shape'])
    radar_fit_uncert[name]=xarray.DataArray(ps.rvs(size=nsamps),coords=coords)





#mn_fit = fits.mean('sample')
## now to plot return periods and their uncertainties.
import matplotlib.ticker
fig,axis = plt.subplots(nrows=2,ncols=1,num='radar_gev_fit',
                 figsize=(8,5),clear=True,layout='constrained',sharex='all')
rp = np.geomspace(10, 200)
# plot c4 and c5 cases.
for name,color in zip( ['1km_c4','1km_c5'],['red','purple']):
    plot_rp(rp,radar_fit_uncert[name],axis[0],color=color,label=name)
    axis[0].axhline(radar_rain[name],linestyle='dashed',linewidth=2,color=color)
# plot 5km and 1km data on separate axis
for name,color,ax in zip(['5km','1km'],['blue','k'],axis.flatten()):
    plot_rp(rp, radar_fit_uncert[name],ax, color=color,label=name)
    ax.set_xlabel('Return Period (Summers)')
    ax.set_ylabel('JJA Rx1h (mm/h)')
    ax.set_title(f"Radar {name} JJA Rx1h return values")
    ax.set_xscale('log')
    ax.set_xticks([10,20,50,100,200])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    # add on the actual radar value
    ax.axhline(radar_rain[name],linestyle='dashed',linewidth=2,color=color)

axis[0].legend()
## and print out the mean values + 5-95%
for name in radar_fit.keys():
    print_rp(radar_fit_uncert[name],radar_rain[name])


fig.show()
commonLib.saveFig(fig)

## plot RP's for all field!
import cartopy.crs as ccrs
import CPM_rainlib
fig,axis = plt.subplots(nrows=2,ncols=2,num='map_rtn_prds',clear=True,
                        subplot_kw=dict(projection=ccrs.OSGB()),
                        layout='constrained',figsize=(8,8))
levels=[2,5,10,20,50,100,200,500]
label = commonLib.plotLabel()
for ax,(key,fld) in zip(axis.flatten(),rain_aug2020.items()):

    mnp=radar_fit_uncert[key].mean('sample')
    params = [mnp.sel(parameter=p) for p in ['shape', 'location', 'scale']]
    dist = scipy.stats.genextreme(*params)
    rp=xarray.DataArray(1.0/dist.sf(fld),coords=fld.coords) # compute the return prd
    ax.set_extent(CPMlib.stonehaven_rgn_extent,crs=CPMlib.projRot)
    cm=rp.plot(levels=levels,cmap='RdYlBu',add_colorbar=False,ax=ax)
    CPM_rainlib.std_decorators(ax,radar_col='green')
    ax.set_title(key)
    ax.plot(*CPMlib.carmont_drain_OSGB.values(),marker='*',color='black',
            ms=10,transform=ccrs.OSGB())
    label.plot(ax)
fig.colorbar(cm,ax=axis,**CPMlib.kw_colorbar)
fig.show()
commonLib.saveFig(fig)
