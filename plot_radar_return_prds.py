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
my_logger=CPM_rainlib.logger
commonLib.init_log(my_logger)

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
radar_fit_dir = CPMlib.radar_dir/'radar_rgn_fit'
radar_fit_dir.mkdir(exist_ok=True,parents=True) # make sure the directory exists
radar_fit = dict()
radar_rain = dict()
radar_fit_uncert = dict()
rain_aug2020 = dict()

for path in radar_fit_dir.glob('*.nc'):
    name = path.stem
    if name.endswith('_fit'):
        radar_fit[name.replace('_fit','')] = xarray.load_dataset(path)
    elif name.endswith('_fit_uncert'):
        radar_fit_uncert[name.replace('_fit_uncert','')] = xarray.load_dataarray(path)
    elif name.endswith('_carmont_rain'):
        radar_rain[name.replace('_carmont_rain','')] = xarray.load_dataarray(path)
    elif name.endswith('_aug2020'):
        rain_aug2020[name.replace('_aug2020','')] = xarray.load_dataarray(path)
    else:
        my_logger.warning(f"Unknown file {path}. Ignoring")

fig, all_axes = plt.subplots(nrows=2, ncols=2, num='radar_gev_fit',
                         figsize=(8, 5), clear=True, layout='constrained', sharex='all',sharey='row',
                         )
rp = np.geomspace(10, 200)
label = commonLib.plotLabel()
for rolling,axis in zip([1,4],all_axes):

# plot c4 and c5 cases.
    for name, color in zip(['1km_c4', '1km_c5'], ['red', 'purple']):
        plot_rp(rp, radar_fit[name].Parameters.sel(rolling=rolling), axis[0], color=color, label=name)
        axis[0].axhline(radar_rain[name].sel(rolling=rolling), linestyle='dashed', linewidth=2, color=color)
    # plot 5km and 1km data on separate axis
    for name, color, ax in zip(['5km', '1km'], ['blue', 'k'], axis.flatten()):
        plot_rp(rp, radar_fit[name].Parameters.sel(rolling=rolling), ax, color=color, label=name)
        ax.set_xlabel('Return Period (Summers)')
        ax.set_ylabel(f'JJA Rx{rolling:d}h (mm/h)')
        ax.set_title(f"Radar {name} JJA Rx{rolling:d}h return values")
        ax.set_xscale('log')
        ax.set_xticks([10, 20, 50, 100, 200])
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        # add on the actual radar value
        ax.axhline(radar_rain[name].sel(rolling=rolling), linestyle='dashed', linewidth=2, color=color)
        label.plot(ax)

    axis[0].legend()
    # and print out the mean values + 5-95%
    for name in radar_fit.keys():
        print_rp(radar_fit_uncert[name].sel(rolling=rolling), float(radar_rain[name].sel(rolling=rolling)))

fig.show()
commonLib.saveFig(fig)

## plot RP's for all field!


fig, axis = plt.subplots(nrows=2, ncols=2, num='map_rtn_prds', clear=True,
                         subplot_kw=dict(projection=ccrs.OSGB()),
                         layout='constrained', figsize=(8, 8)
                         )
levels = [2, 5, 10, 20, 50, 100, 200]
label = commonLib.plotLabel()
rolling=4 # 4 hour rolling extremes
import itertools
for ax, (key,rolling) in zip(axis.flatten(), itertools.product(['1km','5km'],[1,4])):
    mnp = radar_fit_uncert[key].sel(rolling=rolling).mean('sample')
    params = [mnp.sel(parameter=p) for p in ['shape', 'location', 'scale']]
    dist = scipy.stats.genextreme(*params)
    f = rain_aug2020[key].sel(rolling=rolling).squeeze(drop=True)
    rp = xarray.DataArray(1.0 / dist.sf(f), coords=f.coords)  # compute the return prd
    ax.set_extent(CPMlib.carmont_rgn_extent, crs=CPMlib.projRot)
    cm = rp.plot(levels=levels, cmap='RdYlBu', add_colorbar=False, ax=ax)
    CPM_rainlib.std_decorators(ax, radar_col='green',radarNames=True,show_railways=True)
    ax.set_title(key+f' Rx{rolling:d}h Return Period')
    ax.plot(*CPMlib.carmont_drain_OSGB.values(), marker='*', mec='black',mfc='None',mew=2,
            ms=10, transform=ccrs.OSGB()
            )
    label.plot(ax)
fig.colorbar(cm, ax=axis, **CPMlib.kw_colorbar)
fig.show()
commonLib.saveFig(fig)


