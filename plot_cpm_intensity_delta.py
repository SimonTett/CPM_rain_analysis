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

def comp_params(fit: xarray.Dataset,
                temperature: float = 0.0,
               log10_area: typing.Optional[float]=None,
               hour:typing.Optional[float] = None,
               height:typing.Optional[float] = None):
    if log10_area is None:
        log10_area = math.log10(150.)
    if hour is None:
        hour = 13.0
    if height is None:
        height = 100.
    result = [fit.Parameters.sel(parameter='shape')] # start with shape
    param=dict(log10_area=log10_area,hour=hour,hour_sqr=hour**2,height=height,
                   CET=temperature,CET_sqr=temperature**2)
    for k in ['location', 'scale']:
        r = fit.Parameters.sel(parameter=k)
        for pname,v in param.items():
            p = f"D{k}_{pname}"
            try:
                r = r + v * fit.Parameters.sel(parameter=p)
            except KeyError:
                logging.warning(f"{k} missing {p} so ignoring")
        r = r.assign_coords(parameter=k)  #  rename
        result.append(r)
    result=xarray.concat(result,'parameter')
    return result
# do the GEV calculations
recreate_fit = True # set False to use cache
fit_dir = CPM_rainlib.dataDir / 'CPM_scotland_filter' / "fits"
obs_cet = commonLib.read_cet()  # read in the obs CET
obs_cet_jja = obs_cet.where(obs_cet.time.dt.season == 'JJA', drop=True)
t_today = obs_cet_jja.sel(time=slice('2012', '2021')).mean()
temps = dict()

# compute the fits!
ds = xarray.open_mfdataset(CPMlib.CPM_filt_dir.glob("**/CPM*11_30_23.nc"), parallel=True)
L = ds.time.dt.season == 'JJA'
rgn = CPMlib.stonehaven_rgn
maxRain = ds.seasonalMax.sel(**rgn).where(L, drop=True).load()

#rgn_all = dict(longitude=slice(357.5, 361.0), latitude=slice(1.5, 7.5))  # region for which extraction was done.

cpm_cet = xarray.load_dataset(CPM_rainlib.dataDir / "CPM_ts/cet_tas.nc")
cpm_cet = cpm_cet.tas.resample(time='QS-DEC').mean()  # resample to seasonal means
cpm_cet_jja = cpm_cet.where(cpm_cet.time.dt.season == 'JJA', drop=True)  # and pull out summer.
stack_dim = dict(t_e=['time', "ensemble_member"])

fit = gev_r.xarray_gev(maxRain.stack(**stack_dim), cov=[(cpm_cet_jja-t_today).rename('CET').stack(**stack_dim)], dim='t_e',
                       name='Rgn_c', file=fit_dir / 'rgn_fit_cet.nc',recreate_fit=recreate_fit)
rolling=4
pv=1.0/50.
intensity = gev_r.xarray_gev_isf(comp_params(fit),[pv])
intensity_p1k = gev_r.xarray_gev_isf(comp_params(fit,temperature=1.0),[pv])
i_percent = (100*intensity_p1k/intensity)-100.
## plot the results



fig_today, axes = plt.subplots(nrows=2, ncols=2, figsize=(8, 7), clear=True,
                                                   num='map_intensity',
                                                   subplot_kw=dict(projection=CPMlib.projRot))
label = commonLib.plotLabel()
cmap = 'RdYlBu'
for (axis_today, axis_delta),rolling in zip(axes, [1, 4]):
    carmont = float(intensity.sel(**CPMlib.carmont_drain, method='Nearest').sel(rolling=rolling))
    carmont_ip = float(i_percent.sel(**CPMlib.carmont_drain, method='Nearest').sel(rolling=rolling))


    print(f"CPM Carmont drain Rx{rolling:d}h rp={math.floor(1.0/pv):d} "
          f"i {carmont:3.1f} mm/h change {carmont_ip:3.1f} %")
    delta=math.ceil(carmont*0.2)/5
    intensity_levels = np.arange(math.floor(carmont*0.9), math.ceil(carmont*1.1+delta),delta)
    ratio_levels = np.arange(2,14)
    msk = (intensity < carmont * 1.1) * (intensity > carmont * 0.9)
    msk = True # no masking wanted
    #intensity.plot(ax=axis_today, cmap=cmap, levels=intensity_levels, add_colorbar=False, alpha=0.4)
    kw_colorbar = CPMlib.kw_colorbar.copy()
    kw_colorbar.update(label='mm/h')
    intensity.sel(rolling=rolling).plot(ax=axis_today, cmap=cmap, levels=intensity_levels, cbar_kwargs=kw_colorbar, alpha=1.0)
    kw_colorbar.update(label='% change')
    #i_percent.plot(ax=axis_delta, cmap=cmap, levels=ratio_levels, add_colorbar=False, alpha=0.4)
    i_percent.sel(rolling=rolling).plot(ax=axis_delta, cmap=cmap, levels=ratio_levels, cbar_kwargs=kw_colorbar, alpha=1.0)
    axis_today.set_title(f'Rx{rolling:d}h Intensity (2012-22)')
    axis_delta.set_title(f'Rx{rolling:d}h Intensity $\Delta$ %/K')
carmont_rgn = {k: slice(v - 75e3, v + 75e3) for k, v in CPMlib.carmont_drain_OSGB.items()}
xstart=carmont_rgn['projection_x_coordinate'].start
xstop=carmont_rgn['projection_x_coordinate'].stop
ystart=carmont_rgn['projection_y_coordinate'].start
ystop=carmont_rgn['projection_y_coordinate'].stop
x, y = [xstart, xstart, xstop, xstop, xstart], [ystart, ystop, ystop, ystart, ystart] # coords for box.



for ax in axes.flat:
    ax.set_extent(CPMlib.stonehaven_rgn_extent)
    CPM_rainlib.std_decorators(ax, radar_col='green', show_railways=True)
    g = ax.gridlines(draw_labels=True)
    g.top_labels = False
    g.left_labels = False


    label.plot(ax)
    # add on carmont
    ax.plot(*CPMlib.carmont_drain_long_lat, transform=ccrs.PlateCarree(), marker='*', ms=10, color='black')
    ax.plot(x, y, color='black', linewidth=2, transform=CPMlib.projOSGB)

fig_today.show()
commonLib.saveFig(fig_today)
