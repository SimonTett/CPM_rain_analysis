# Plot CPM intensity and % change for different rollings. Also plot the change in risk for radar rain at carmont.
import matplotlib.pyplot as plt
import numpy as np

import CPM_rainlib
import CPMlib
import xarray

import commonLib
from R_python import gev_r
import typing
import math
import logging
import scipy
import matplotlib.colors as mcolors
import matplotlib.cm as mcm
import cartopy.crs as ccrs
import CPM_rainlib


def comp_params(
        fit: xarray.Dataset,
        temperature: float = 0.0,
        log10_area: typing.Optional[float] = None,
        hour: typing.Optional[float] = None,
        height: typing.Optional[float] = None
        ):
    if log10_area is None:
        log10_area = math.log10(150.)
    if hour is None:
        hour = 13.0
    if height is None:
        height = 100.
    result = [fit.Parameters.sel(parameter='shape')]  # start with shape
    param = dict(log10_area=log10_area, hour=hour, hour_sqr=hour ** 2, height=height,
                 CET=temperature, CET_sqr=temperature ** 2
                 )
    for k in ['location', 'scale']:
        r = fit.Parameters.sel(parameter=k)
        for pname, v in param.items():
            p = f"D{k}_{pname}"
            try:
                r = r + v * fit.Parameters.sel(parameter=p)
            except KeyError:
                logging.warning(f"{k} missing {p} so ignoring")
        r = r.assign_coords(parameter=k)  #  rename
        result.append(r)
    result = xarray.concat(result, 'parameter')
    return result


# do the GEV calculations
recreate_fit = False  # set False to use cache
fit_dir = CPM_rainlib.dataDir / 'CPM_scotland_filter' / "fits"
raw_fit_dir = CPM_rainlib.dataDir / 'CPM_scotland' / "fits"  # no filtering
obs_cet = commonLib.read_cet()  # read in the obs CET
obs_cet_jja = obs_cet.where(obs_cet.time.dt.season == 'JJA', drop=True)
t_today = obs_cet_jja.sel(**CPMlib.today_sel).mean()
temps = dict()

# compute the fits!
rgn = CPMlib.carmont_rgn
ds = xarray.open_mfdataset(CPMlib.CPM_filt_dir.glob("**/CPM*11_30_23.nc"), parallel=True)
L = ds.time.dt.season == 'JJA'
maxRain = ds.seasonalMax.sel(**rgn).where(L, drop=True).load()

# raw data
ds_raw = xarray.open_mfdataset(CPMlib.CPM_dir.glob("**/CPM*11_30_23.nc"), parallel=True)
L = ds_raw.time.dt.season == 'JJA'
maxRain_raw = ds_raw.seasonalMax.sel(**rgn).where(L, drop=True).load()
#rgn_all = dict(longitude=slice(357.5, 361.0), latitude=slice(1.5, 7.5))  # region for which extraction was done.

cpm_cet = xarray.load_dataset(CPM_rainlib.dataDir / "CPM_ts/cet_tas.nc")
cpm_cet = cpm_cet.tas.resample(time='QS-DEC').mean()  # resample to seasonal means
cpm_cet_jja = cpm_cet.where(cpm_cet.time.dt.season == 'JJA', drop=True)  # and pull out summer.
stack_dim = dict(t_e=['time', "ensemble_member"])

fit = gev_r.xarray_gev(maxRain.stack(**stack_dim), cov=[(cpm_cet_jja - t_today).rename('CET').stack(**stack_dim)],
                       dim='t_e',
                       name='Rgn_c', file=fit_dir / 'rgn_fit_cet.nc', recreate_fit=recreate_fit
                       )

raw_fit = gev_r.xarray_gev(maxRain_raw.stack(**stack_dim),
                           cov=[(cpm_cet_jja - t_today).rename('CET').stack(**stack_dim)], dim='t_e',
                           name='Rgn_c', file=raw_fit_dir / 'rgn_fit_cet.nc', recreate_fit=recreate_fit
                           )
## plot the results
rtn_prd=100
pv = 1.0 /rtn_prd
accum_filt_carmont  = gev_r.xarray_gev_isf(comp_params(fit).sel(**CPMlib.carmont_drain, method='Nearest'), [pv])*fit['rolling']
for fit_params, fig_name in zip([fit, raw_fit], ['cpm_intensity_delta', 'cpm_intensity_delta_raw']):
    intensity = gev_r.xarray_gev_isf(comp_params(fit_params), [pv])
    intensity_p1k = gev_r.xarray_gev_isf(comp_params(fit_params, temperature=1.0), [pv])
    i_percent = (100 * intensity_p1k / intensity) - 100.

    if 'raw' in fig_name:
        extra_title='Raw '
    else:
        extra_title =''

    fig_today, axes = plt.subplots(nrows=2, ncols=2, figsize=(7, 7), clear=True,
                                   num=fig_name, layout='constrained',
                                   subplot_kw=dict(projection=CPMlib.projRot)
                                   )

    label = commonLib.plotLabel()
    cmap = 'RdYlBu'
    for (axis_today, axis_delta), rolling in zip(axes, [1, 4]):
        carmont = float(intensity.sel(**CPMlib.carmont_drain, method='Nearest').sel(rolling=rolling))
        carmont_ip = float(i_percent.sel(**CPMlib.carmont_drain, method='Nearest').sel(rolling=rolling))
        print(f"{extra_title}CPM Carmont drain Rx{rolling:d}h rp={math.floor(1.0 / pv):d} "
              f"i {carmont:3.1f} mm/h change {carmont_ip:3.1f} %"
              )
        # want fixed levels so compute them from filtered data.
        carmont_filt= float(accum_filt_carmont.sel(rolling=rolling))
        delta = math.ceil(carmont_filt * 0.3) / 5
        intensity_levels = np.arange(math.ceil( carmont_filt* 0.85 ), math.floor(carmont_filt * 1.15  + delta),
                                     delta
                                     )
        ratio_levels = np.arange(2, 14)
        kw_colorbar = CPMlib.kw_colorbar.copy()
        kw_colorbar.update(label='mm')
        (intensity.sel(rolling=rolling) * rolling).plot(ax=axis_today, cmap=cmap, levels=intensity_levels,
                                                        cbar_kwargs=kw_colorbar, alpha=1.0
                                                        )
        kw_colorbar.update(label='% change')

        i_percent.sel(rolling=rolling).plot(ax=axis_delta, cmap=cmap, levels=ratio_levels, cbar_kwargs=kw_colorbar)
        i_percent.sel(rolling=rolling).squeeze(drop=True).plot.contour(ax=axis_delta, levels=[CPMlib.cc_dist.mean()],
                                                                       colors='black', linewidths=1, linestyles='dashed'
                                                                       )
        axis_today.set_title(f'Rx{rolling:d}h Total (2012-22)')
        axis_delta.set_title(f'Rx{rolling:d}h Total $\Delta$ %/K')
    carmont_rgn = {k: slice(v - 75e3, v + 75e3) for k, v in CPMlib.carmont_drain_OSGB.items()}
    xstart = carmont_rgn['projection_x_coordinate'].start
    xstop = carmont_rgn['projection_x_coordinate'].stop
    ystart = carmont_rgn['projection_y_coordinate'].start
    ystop = carmont_rgn['projection_y_coordinate'].stop
    x, y = [xstart, xstart, xstop, xstop, xstart], [ystart, ystop, ystop, ystart, ystart]  # coords for box.

    for ax in axes.flat:
        ax.set_extent(CPMlib.carmont_rgn_extent)
        CPM_rainlib.std_decorators(ax, radar_col='green', show_railways=True)
        g = ax.gridlines(draw_labels=True)
        g.top_labels = False
        g.left_labels = False

        label.plot(ax)
        # add on carmont
        ax.plot(*CPMlib.carmont_drain_long_lat, transform=ccrs.PlateCarree(), marker='o', ms=6, color='cornflowerblue')
        ax.plot(x, y, color='black', linewidth=2, transform=CPMlib.projOSGB)
    fig_today.suptitle(f"{extra_title}CPM intensity and % change for return period = {rtn_prd:d}", size='small')
    fig_today.show()
    commonLib.saveFig(fig_today)
