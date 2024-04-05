# compute hte probability and intensity ratios for carmont


## now let's compute the intensity and PR's
import logging
import math
import numpy as np
import xarray
import matplotlib.pyplot as plt
import typing
import CPM_rainlib
import commonLib
import CPMlib
from R_python import gev_r
import itertools


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


# read in the radar data
radar_fit_dir = CPMlib.radar_dir/'radar_rgn_fit'
radar_fit_dir.mkdir(exist_ok=True,parents=True) # make sure the directory exists
radar_fit = dict()
radar_rain = dict()
radar_fit_uncert = dict()
rain_aug2020 = dict()
my_logger=CPM_rainlib.logger
commonLib.init_log(my_logger)
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

## get in the model fits and work out ratios for today and PI
fit_dir = CPM_rainlib.dataDir / 'CPM_scotland_filter' / "fits"
fit_file = fit_dir / 'rgn_fit_cet.nc'
cpm_gev_params = xarray.load_dataset(fit_file).coarsen(grid_latitude=5,grid_longitude=5,boundary='pad').mean().sel(**CPMlib.carmont_drain,method='nearest')
obs_cet = commonLib.read_cet()  # read in the obs CET
obs_cet_jja = obs_cet.where(obs_cet.time.dt.season == 'JJA', drop=True)
t_today = obs_cet_jja.sel(**CPMlib.today_sel).mean()
t_PI = obs_cet_jja.sel(**CPMlib.PI_sel).mean()
delta = t_PI-t_today
params_PI = comp_params(cpm_gev_params, temperature=delta)
params_p2k = comp_params(cpm_gev_params, temperature=delta+2)
params_today = comp_params(cpm_gev_params)
ratio = params_PI/params_today
ratio_p2k = params_p2k/params_today
mn_radar = {key:fit.mean('sample') for key,fit in radar_fit_uncert.items()}



## now plot the data
fig,axis = plt.subplots(num='PR',clear=True,nrows=2,ncols=2,layout='constrained',figsize=(8,5),sharex='col',sharey='col')


#for ax,(name,rolling) in zip(axis.flat,itertools.product(['1km','5km'],[1,4])):
rain=np.geomspace(2,100)
rtn_period=np.geomspace(10,200)
for (ax_pr, ax_intensity),name in zip(axis,['1km', '5km']):
    mnp  = mn_radar[name]
    # now compute the return periods for the PI and today
    p_pi = gev_r.xarray_gev_sf(mnp * ratio, rain)
    p_today = gev_r.xarray_gev_sf(mnp, rain)
    p_p2k = gev_r.xarray_gev_sf(mnp * ratio_p2k, rain)
    pr = 100 * (p_today / p_pi)
    pr_p2k = 100 * (p_p2k / p_pi)
    # and the intensity changes as a fun of rp
    i_pi = gev_r.xarray_gev_isf(mnp * ratio, 1.0/rtn_period)
    i_today = gev_r.xarray_gev_isf(mnp, 1.0/rtn_period)
    i_p2k = gev_r.xarray_gev_isf(mnp * ratio_p2k, 1.0/rtn_period)
    ir = 100 * (i_today / i_pi)-100
    ir_p2k = 100 * (i_p2k / i_pi)-100
    for rolling, linestyle in zip([1, 4], ['dashed', 'solid']):
        event_rp =float(1.0/(gev_r.xarray_gev_sf(mnp.sel(rolling=rolling),float(radar_rain[name].sel(rolling=rolling)))))
        pr.sel(rolling=rolling).plot(ax=ax_pr,label=f'Rx{rolling:d}h PR (today)',linestyle=linestyle,color='blue',linewidth=2)
        pr_p2k.sel(rolling=rolling).plot(ax=ax_pr,label=f'Rx{rolling:d}h PR (+2K)',linestyle=linestyle,color='red',linewidth=2)
        ax_pr.axvline(radar_rain[name].sel(rolling=rolling), linestyle=linestyle,color='black',linewidth=2)
        ir.sel(rolling=rolling).plot(ax=ax_intensity,x='return_period',label=f'Rx{rolling:d}h IR (today)',linestyle=linestyle,color='blue',linewidth=2)
        ir_p2k.sel(rolling=rolling).plot(ax=ax_intensity,x='return_period',label=f'Rx{rolling:d}h IR (+2K)',linestyle=linestyle,color='red',linewidth=2)
        ax_intensity.axvline(event_rp, linestyle=linestyle,color='black',linewidth=2)
    ax_pr.set_title(f'Prob Ratio -- {name} Carmont')
    ax_pr.set_xlabel('Rain (mm/h)')
    ax_pr.set_ylabel('PR (%)')

    ax_intensity.set_title(f'Int. Ratio -- {name} Carmont')
    ax_intensity.set_xlabel('Return Period (summers)')
    ax_intensity.set_ylabel('Int Change (%)')
# now do the intensity ratios
axis[0][0].legend()
fig.show()
commonLib.saveFig(fig)
