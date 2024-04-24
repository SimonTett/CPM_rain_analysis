# plot the radar return periods and maps of them.
# uses data produced by comp_radar_dist_params.py

import xarray

import matplotlib.ticker
import cartopy.crs as ccrs
import CPM_rainlib
import commonLib
import matplotlib.pyplot as plt
import numpy as np
import CPMlib
import typing
from R_python import gev_r
import scipy.stats
import itertools

my_logger = CPM_rainlib.logger
commonLib.init_log(my_logger)


def print_rp(
        radar_fit_params: xarray.DataArray,
        radar_rain: float
):
    rp_event = 1.0 / gev_r.xarray_gev_sf(radar_fit_params, radar_rain)

    qv = rp_event.quantile([0.05, 0.5, 0.95], dim='sample')
    rolling = int(radar_fit_params['rolling'])
    print(f'{name} Rx{rolling:d}h Carmont rain {float(radar_rain) * rolling:3.0f} mm '
          f'Return period: {float(qv.sel(quantile=0.05)):3.0f}'
          f'({float(qv.sel(quantile=0.05)):3.0f} '
          f'-{float(qv.sel(quantile=0.95)):3.0f}) years'
          )


def plot_rp(
        rp: np.ndarray,
        radar_fit_params: xarray.DataArray,
        ax,
        color: typing.Optional[str] = None,
        label: typing.Optional[str] = None
):
    rv_uncert = gev_r.xarray_gev_isf(radar_fit_params, 1.0 / rp) * float(radar_fit_params['rolling'])
    rv_q = rv_uncert.quantile([0.05, 0.5, 0.95], dim='sample')  # compute 5-95% uncertainty
    rv_q.sel(quantile=0.5).plot(x='return_period', ax=ax, color=color, label=label)
    ax.fill_between(y1=rv_q.sel(quantile=0.95).values,
                    y2=rv_q.sel(quantile=0.05).values,
                    x=rv_q.return_period, alpha=0.5, color=color
                    )


radar_fit_dir = CPMlib.radar_dir / 'radar_rgn_fit'
radar_fit = dict()
radar_rain = dict()
radar_fit_uncert = dict()
radar_aug2020 = dict()
carmont = CPMlib.carmont_drain_OSGB.copy()
carmont.update(time='2020-08')

for path in radar_fit_dir.glob('*_fit.nc'):
    name = path.stem.replace('_fit', '')
    fit = xarray.load_dataset(path)
    radar_fit[name] = fit
    # generate samples from distribution
    radar_fit_uncert[name] = CPM_rainlib.xarray_gen_cov_samp(fit.Parameters, fit.Cov, nsamp=1000)

# read in the summary data and get the actual rain values
for name in radar_fit.keys():
    summary_path = CPMlib.radar_dir / 'summary' / f'summary_2008_{name}.nc'
    radar_rain[name] = xarray.open_dataset(summary_path).Radar_rain_Max.sel(**carmont, method='nearest').load()

    top_path = CPMlib.radar_dir / 'topography' / f'topog_{name}.nc'
    topog = xarray.open_dataarray(top_path).sel(**CPMlib.carmont_rgn_OSGB).load() # get in the topog
    ds = xarray.open_dataset(summary_path).sel(time=slice('2020-06', '2020-08')).sel(**CPMlib.carmont_rgn_OSGB).load()
    ok = ds.Radar_rain_Max.notnull()
    idx = ds.Radar_rain_Max.where(ok, 0.0).argmax('time')
    mx_time = ds.Radar_rain_MaxTime.isel(time=idx).drop_vars('time')
    season_mx = ds.Radar_rain_Max.isel(time=idx).drop_vars('time')
    msk = (mx_time.dt.strftime('%Y-%m-%d') == '2020-08-12') & (topog > 0.0)
    radar_aug2020[name] = season_mx.where(msk)
    mx_time = xarray.open_dataset(summary_path).Radar_rain_MaxTime.sel(**carmont, method='nearest').load()
    my_logger.debug(f"Max Time for {name} radar data is {mx_time.dt.strftime('%Y-%m-%d %H:%M').values}")

## plot radar return periods and rain
names = ['5km', '1km-c4', '1km-c5', '1km', ]
fig, all_axes = plt.subplots(nrows=1, ncols=2, num='radar_return_prds',
                             figsize=(8, 3), clear=True, layout='constrained', sharex='col', sharey='row',
                             )
rp = np.geomspace(10, 200)
label = commonLib.plotLabel()
roll_values = [1, 4]
for rolling, ax in zip(roll_values, all_axes):

    for name in names:
        plot_rp(rp, radar_fit_uncert[name].sel(rolling=rolling), ax, color=CPMlib.radar_cols[name],
                label=name
                )
        accum = radar_rain[name].sel(rolling=rolling) * rolling

        # work out median rp for radar_rain.
        rp_intersect = 1.0 / gev_r.xarray_gev_sf(radar_fit_uncert[name].sel(rolling=rolling),
                                                 [float(radar_rain[name].sel(rolling=rolling))]
                                                 )
        q = rp_intersect.quantile([0.05, 0.5, 0.95], dim='sample')
        l, m, u = q[0], q[1], q[2]
        ax.errorbar(m, accum, xerr=(m - l, u - l), marker='o',
                    color=CPMlib.radar_cols[name], markersize=4, capsize=10, capthick=2
                    )
        # add on the actual radar value
        ax.plot(10 + np.random.uniform(2), accum, ms=8, marker='o', mfc='None', mec=CPMlib.radar_cols[name], mew=2)
    label.plot(ax)
    ax.set_xscale('log')
    ax.set_xticks([10, 20, 50, 100, 200])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel('Return Period (Summers)')
    ax.set_ylabel(f'JJA Rx{rolling:d}h (mm)')
    ax.set_title(f"Radar Rain Rx{rolling:d}h  return values")
all_axes[0].legend(ncol=2)

fig.show()
commonLib.saveFig(fig)
## and print out the mean values + 5-95%
for rolling in roll_values:
    for name in names:
        print_rp(radar_fit_uncert[name].sel(rolling=rolling), float(radar_rain[name].sel(rolling=rolling)))
## plot RP's for all field!


fig, axis = plt.subplots(nrows=2, ncols=2, num='map_return_prds', clear=True,
                         subplot_kw=dict(projection=ccrs.OSGB()),
                         layout='constrained', figsize=(7, 6)
                         )
levels = [10,50, 20, 30,40,60,80,120,160]
label = commonLib.plotLabel()

for ax, (key, rolling) in zip(axis.T.flatten(), itertools.product(['1km', '5km'], [1, 4])):
    mnp = radar_fit_uncert[key].sel(rolling=rolling).mean('sample')
    params = [mnp.sel(parameter=p) for p in ['shape', 'location', 'scale']]
    dist = scipy.stats.genextreme(*params)
    f = radar_aug2020[key].sel(rolling=rolling).squeeze(drop=True)
    rp = xarray.DataArray(1.0 / dist.sf(f), coords=f.coords)  # compute the return prd
    ax.set_extent(CPMlib.carmont_rgn_extent, crs=CPMlib.projRot)
    cm = rp.plot(levels=levels, cmap='RdYlBu', add_colorbar=False, ax=ax)
    CPM_rainlib.std_decorators(ax, radar_col='green', radarNames=True, show_railways=True)
    ax.set_title(key + f' Rx{rolling:d}h Return Period')
    ax.plot(*CPMlib.carmont_drain_OSGB.values(), marker='o', mec='cornflowerblue', mfc='None', mew=2,
            ms=10, transform=ccrs.OSGB()
            )
for ax in axis.flatten():
    label.plot(ax)
fig.colorbar(cm, ax=axis,label='Return Period (summers)', **CPMlib.kw_colorbar)
fig.show()
commonLib.saveFig(fig)
