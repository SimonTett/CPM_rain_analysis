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


def comp_fits(rng, radar_events, nsamps: int = 100) -> xarray.DataArray:
    n_events = radar_events.EventTime.shape[0]
    rand_index = rng.integers(len(radar_events.quantv), size=(n_events, nsamps))
    wt = radar_events.count_cells
    fits = []
    for indx in range(0, nsamps):  # iterate over random samples
        ind = xarray.DataArray(rand_index[:, indx],
                               coords=dict(EventTime=radar_events.coords['EventTime'])
                               )
        sample = radar_events.max_precip.isel(quantv=ind).drop_vars('quantv').assign_coords(sample=indx)
        #sample = sample[sample.notnull()] # drop missing values.
        fits.append(gev_r.xarray_gev(sample, dim='EventTime', weights=wt))

    fits = xarray.concat(fits, dim='sample')
    return fits


def print_rp(
        radar_fit_params: xarray.DataArray,
        radar_rain: float
        ):
    rp_event = 1.0 / gev_r.xarray_gev_sf(radar_fit_params, radar_rain)
    mn_rp = rp_event.mean('sample')
    qv = rp_event.quantile([0.05, 0.95], dim='sample')
    print(f'{name} Carmont rain {float(radar_rain):3.0f} mm/h '
          f'Return period: {float(mn_rp):3.0f}'
          f'({float(qv.isel(quantile=0)):3.0f} '
          f'-{float(qv.isel(quantile=1)):3.0f}) years',
          f' for rolling {int(radar_fit_params["rolling"]):d}'
          )


def plot_rp(
        rp: np.ndarray,
        radar_fit_params: xarray.DataArray,
        ax,
        color: typing.Optional[str] = None,
        label: typing.Optional[str] = None
        ):
    rv = gev_r.xarray_gev_isf(radar_fit_params, 1.0 / rp)
    rv_uncert = gev_r.xarray_gev_isf(radar_fit_params, 1.0 / rp)
    rv_q = rv_uncert.quantile([0.05, 0.95], dim='sample')  # compute 5-95% uncertainty
    rv.mean('sample').plot(x='return_period', ax=ax, color=color, label=label)
    ax.fill_between(y1=rv_q.isel(quantile=-1).values,
                    y2=rv_q.isel(quantile=0).values,
                    x=rv_q.return_period, alpha=0.5, color=color
                    )


rain_aug2020 = dict()
radar_fit = dict()
radar_rain = dict()
radar_fit_uncert = dict()
radar_rain_time = dict()
rng = numpy.random.default_rng(123456)
nsamps = 100
projGB = ccrs.OSGB()
carmont = CPMlib.carmont_rgn_OSGB.copy()
carmont.update(time=slice('2020-06-01','2020-08-31')) # include summer 2020
radar_events = list((CPMlib.radar_dir/'radar_events').glob('*.nc'))
for  path in radar_events:
    summary_path = path.parent.parent / 'summary' / path.name.replace('events', 'summary')
    name = "_".join(path.stem.split("_")[2:])
    radar_events = xarray.load_dataset(path)
    rain, rainMx, topog = CPM_rainlib.get_radar_data(summary_path, region=carmont,height_range=slice(50., None),
                                                     topog_grid=radar_events.attrs['topog_grid'])
    rain = rain.where(rainMx.dt.strftime('%Y-%m-%d') == '2020-08-12')
    rain_aug2020[name] = rain
    radar_rain[name] = rain.sel(**CPMlib.carmont_drain_OSGB, method='nearest')
    radar_rain_time[name] = rain.sel(**CPMlib.carmont_drain_OSGB, method='nearest')

    radar_fit[name] = comp_fits(rng, radar_events, nsamps=nsamps)
    print(f"Computed fits for {name}")
    rolling_uncert=[]
    for r in radar_fit[name]['rolling']:
        mean = radar_fit[name].Parameters.sel(rolling=r).mean('sample')
        cov = radar_fit[name].sel(rolling=r).Cov.mean('sample')
        ps = scipy.stats.multivariate_normal(mean=mean,cov=cov)
        coords = dict(sample=np.arange(0, nsamps), parameter=['location', 'scale', 'shape'])
        uncert = xarray.DataArray(ps.rvs(size=nsamps), coords=coords).assign_coords(rolling=r)
        rolling_uncert.append(uncert)
    radar_fit_uncert[name] = xarray.concat(rolling_uncert, dim='rolling')

# mn_fit = fits.mean('sample')Radar_rain_Mean
## now to plot return periods and their uncertainties.
import matplotlib.ticker

fig, all_axes = plt.subplots(nrows=2, ncols=2, num='radar_gev_fit',
                         figsize=(8, 5), clear=True, layout='constrained', sharex='all'
                         )
rp = np.geomspace(10, 200)
for rolling,axis in zip([1,4],all_axes):

# plot c4 and c5 cases.
    for name, color in zip(['1km_c4', '1km_c5'], ['red', 'purple']):
        plot_rp(rp, radar_fit_uncert[name].sel(rolling=rolling), axis[0], color=color, label=name)
        axis[0].axhline(radar_rain[name].sel(rolling=rolling), linestyle='dashed', linewidth=2, color=color)
    # plot 5km and 1km data on separate axis
    for name, color, ax in zip(['5km', '1km'], ['blue', 'k'], axis.flatten()):
        plot_rp(rp, radar_fit_uncert[name].sel(rolling=rolling), ax, color=color, label=name)
        ax.set_xlabel('Return Period (Summers)')
        ax.set_ylabel('JJA Rx1h (mm/h)')
        ax.set_title(f"Radar {name} JJA Rx{rolling:d}h return values")
        ax.set_xscale('log')
        ax.set_xticks([10, 20, 50, 100, 200])
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        # add on the actual radar value
        ax.axhline(radar_rain[name].sel(rolling=rolling), linestyle='dashed', linewidth=2, color=color)

    axis[0].legend()
    # and print out the mean values + 5-95%
    for name in radar_fit.keys():
        print_rp(radar_fit_uncert[name].sel(rolling=rolling), radar_rain[name].sel(rolling=rolling))

fig.show()
commonLib.saveFig(fig)

## plot RP's for all field!
import cartopy.crs as ccrs
import CPM_rainlib

fig, axis = plt.subplots(nrows=2, ncols=2, num='map_rtn_prds', clear=True,
                         subplot_kw=dict(projection=ccrs.OSGB()),
                         layout='constrained', figsize=(8, 8)
                         )
levels = [2, 5, 10, 20, 50, 100, 200]
label = commonLib.plotLabel()
rolling=4 # 4 hour rolling extremes
for ax, key in zip(axis.flatten(), ['1km','5km','1km_c4','1km_c5']):
    mnp = radar_fit_uncert[key].sel(rolling=rolling).mean('sample')
    params = [mnp.sel(parameter=p) for p in ['shape', 'location', 'scale']]
    dist = scipy.stats.genextreme(*params)
    f = rain_aug2020[key].sel(rolling=rolling).squeeze(drop=True)
    rp = xarray.DataArray(1.0 / dist.sf(f), coords=f.coords)  # compute the return prd
    ax.set_extent(CPMlib.stonehaven_rgn_extent, crs=CPMlib.projRot)
    cm = rp.plot(levels=levels, cmap='RdYlBu', add_colorbar=False, ax=ax)
    CPM_rainlib.std_decorators(ax, radar_col='green',radarNames=True,show_railways=True)
    ax.set_title(key)
    ax.plot(*CPMlib.carmont_drain_OSGB.values(), marker='*', color='black',
            ms=10, transform=ccrs.OSGB()
            )
    label.plot(ax)
fig.colorbar(cm, ax=axis, **CPMlib.kw_colorbar)
fig.show()
commonLib.saveFig(fig)
