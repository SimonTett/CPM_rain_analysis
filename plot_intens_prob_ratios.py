# compute the  probability and intensity ratios for carmont.
# uses radar fits computed by comp_radar_dist_params

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
import scipy.stats


def comp_params(param: xarray.DataArray,
                temperature: float = 0.0,
                log10_area: typing.Optional[float] = None,
                hour: typing.Optional[float] = None,
                height: typing.Optional[float] = None):
    if log10_area is None:
        log10_area = math.log10(150.)
    if hour is None:
        hour = 13.0
    if height is None:
        height = 100.
    result = [param.sel(parameter='shape')]  # start with shape
    param_names = dict(log10_area=log10_area, hour=hour, hour_sqr=hour ** 2, height=height,
                 CET=temperature, CET_sqr=temperature ** 2)
    for k in ['location', 'scale']:
        r = param.sel(parameter=k)
        for pname, v in param_names.items():
            p = f"D{k}_{pname}"
            try:
                r = r + v * param.sel(parameter=p)
            except KeyError:
                logging.warning(f"{k} missing {p} so ignoring")
        r = r.assign_coords(parameter=k)  #  rename
        result.append(r)
    result = xarray.concat(result, 'parameter')
    return result

def xarray_samp_cov(da:xarray.DataArray,
                    dim:str,
                    param_dim:str = 'parameter',
                    param_dim2:typing.Optional[str] = None) -> xarray.DataArray:
    """ Compute the sample covariance of the data array"""
    def cov_matrix(arr):
        return np.cov(arr, rowvar=False)
    if param_dim2 is None:
        param_dim2 = param_dim + '2'
    output_core_dims = [[param_dim,   param_dim2]]
    input_core_dims= [[dim,param_dim]]
    cov = xarray.apply_ufunc(cov_matrix, da, input_core_dims=input_core_dims,
                            output_core_dims=output_core_dims, vectorize=True)
    cov = cov.assign_coords({param_dim2: da[param_dim].values})
    return cov

def xarray_gen_cov_samp(mean,cov,rng,nsamp):
    """ Generate  samples from the covariance matrix"""
    def gen_cov(mean,cov,rng=123456):
        return scipy.stats.multivariate_normal(mean,cov).rvs(nsamp,random_state=rng).T
    input_core_dims=[['parameter'],['parameter','parameter2']]
    output_core_dims=[['parameter','sample',]]
    samps = xarray.apply_ufunc(gen_cov,mean,cov,kwargs=dict(rng=rng),
                              input_core_dims=input_core_dims,
                              output_core_dims=output_core_dims,vectorize=True)
    samps  = samps.assign_coords(sample=np.arange(nsamp))
    return samps

def comp_scaling_cet_samps(fits:xarray.Dataset,rng,nsamp,temperature:float=0.0):
    """
       Compute an estimate of the scaling factor for the CET relative to today.
    :param fits: fits array -- want the parameters and covariance.

    :return:cov change per frac
    :rtype:
    """
    samps = xarray_gen_cov_samp(fits.Parameters,fits.Cov,rng,nsamp)
    params=comp_params(samps,temperature=temperature)
    return params




my_logger = CPM_rainlib.logger
commonLib.init_log(my_logger,level='DEBUG')
# read in the radar data
radar_fit_dir = CPMlib.radar_dir / 'radar_rgn_fit'

radar_fit = dict()
radar_rain = dict()
radar_fit_uncert = dict()
rain_aug2020 = dict()

for path in radar_fit_dir.glob('*.nc'):
    name = path.stem
    if name.endswith('_fit'):
        radar_fit[name.replace('_fit', '')] = xarray.load_dataset(path)
    #elif name.endswith('_fit_uncert'):
    #    radar_fit_uncert[name.replace('_fit_uncert', '')] = xarray.load_dataarray(path)
    elif name.endswith('_carmont_rain'):
        radar_rain[name.replace('_carmont_rain', '')] = xarray.load_dataarray(path)
    elif name.endswith('_aug2020'):
        rain_aug2020[name.replace('_aug2020', '')] = xarray.load_dataarray(path)
    else:
        my_logger.warning(f"Unknown file {path}. Ignoring")

my_logger.debug(f"Loaded radar data")

## get in the model fits and work out ratios for today and PI
fit_dir = CPM_rainlib.dataDir / 'CPM_scotland_filter' / "fits"
fit_file = fit_dir / 'rgn_fit_cet.nc'
cpm_gev_params = xarray.load_dataset(fit_file).rolling(grid_latitude=5, grid_longitude=5, center=True).mean().sel(
    **CPMlib.carmont_drain, method='nearest')
cpm_gev_params['Cov']= cpm_gev_params['Cov']/ 25.

obs_cet = commonLib.read_cet()  # read in the obs CET
obs_cet_jja = obs_cet.where(obs_cet.time.dt.season == 'JJA', drop=True)
t_today = float(obs_cet_jja.sel(**CPMlib.today_sel).mean())
t_PI = float(obs_cet_jja.sel(**CPMlib.PI_sel).mean())
delta = t_PI - t_today
cet_scale =0.92 # scaling factor for  CET from Gm temp.
t_p2k = 2*cet_scale + t_PI# check scaling value
params_PI = comp_params(cpm_gev_params.Parameters, temperature=t_PI-t_today)
params_p2k = comp_params(cpm_gev_params.Parameters, temperature=t_p2k-t_today)
params_today = comp_params(cpm_gev_params.Parameters)
ratio = params_PI / params_today
ratio_p2k = params_p2k / params_today


my_logger.debug(f"Loaded  data")
## compute the uncertainties from the various covariances we have...
# CPM uncerts first -- note the same rng for all
params_today_uncerts = comp_scaling_cet_samps(cpm_gev_params, 123456, 100)
params_PI_uncerts = comp_scaling_cet_samps(cpm_gev_params, 123456, 100, temperature=delta)
params_p2k_uncerts = comp_scaling_cet_samps(cpm_gev_params, 123456, 100, temperature=delta + 2*cet_scale)
# then compute ratio uncertainties.
ratio_PI_uncerts = params_PI_uncerts / params_today_uncerts
ratio_p2k_uncerts = params_p2k_uncerts/params_today_uncerts

# then compute radar PI, today  and p2k uncerts.

radar_pi_uncert = {key: fit.Parameters*ratio_PI_uncerts for key, fit in radar_fit.items()}
radar_p2k_uncert = {key: fit.Parameters*ratio_p2k_uncerts for key, fit in radar_fit.items()}
radar_today_uncert = {key: fit.Parameters for key, fit in radar_fit.items()}

# sampling uncert.

# and the mean cov.
cc_rate = 5.6
## now plot the data
fig, axs = plt.subplots(num='intens_prob_ratios', clear=True, nrows=2, ncols=3, layout='constrained', figsize=(8, 6),sharey='col',sharex='col')
fig.get_layout_engine().set(rect=[0.05,0.0,0.95,1.0])
#for ax,(name,rolling) in zip(axis.flat,itertools.product(['1km','5km'],[1,4])):
rain = np.geomspace(2, 60)
rtn_period = np.geomspace(5, 200)
label = commonLib.plotLabel()
i_pi = {key: gev_r.xarray_gev_isf(value, 1.0/rtn_period) for key,value in radar_pi_uncert.items()}
names = ['1km', '5km']
for axis,rolling in zip(axs,[1,4]):
    pos = axis[0].get_position()
    y = (pos.ymax + pos.ymin) / 2
    x = 0.02
    fig.text(x, y, f'Rx{rolling:d}h',
             ha='left', va='center', rotation=90, fontsize=10
             )
    for index,name in enumerate(names): # loop over different radars
        rain_2020 = float(radar_rain[name].sel(rolling=rolling))
        p_pi=gev_r.xarray_gev_sf(radar_pi_uncert[name].sel(rolling=rolling),rain)
        intensity_pi = gev_r.xarray_gev_isf(radar_pi_uncert[name].sel(rolling=rolling), 1.0 / rtn_period)
        ax_pr = axis[index]
        try:
            ax_intensity = axis[index + len(names)]
        except IndexError:
            ax_intensity = None
        for radar_prd_uncert,temp,color,line_label in zip([radar_today_uncert, radar_p2k_uncert],
                                                                [t_today,t_p2k],
                                                             ['blue', 'red'], ['today', '+2K']):

            p_prd = gev_r.xarray_gev_sf(radar_prd_uncert[name].sel(rolling=rolling),rain) # probs of rain
            ratio = p_prd/p_pi*100 # prob ratio as a percentage
            quant = ratio.quantile([0.05, 0.5, 0.95], dim='sample')
            ax_pr.fill_between(rain*rolling, quant.sel(quantile=0.05),quant.sel(quantile=0.95),
                               color=color,alpha=0.5)
            # median estimate
            ax_pr.plot(rain * rolling, quant.sel(quantile=0.5), label=line_label, linewidth=2, color=color)
            # actual rain values
            ax_pr.axvline( rain_2020* rolling, color='black', linewidth=2)

            # now plot the intensity ratios but only iff have room
            if ax_intensity is None:
                continue

            intensity_prd = gev_r.xarray_gev_isf(radar_prd_uncert[name].sel(rolling=rolling), 1.0/rtn_period)  # Intensity at rtn prd
            change = 100 * (intensity_prd / intensity_pi) - 100
            quant = change.quantile([0.05, 0.5, 0.95], dim='sample')
            ax_intensity.fill_between(quant['return_period'], quant.sel(quantile=0.05),quant.sel(quantile=0.95),
                                      alpha=0.5, color=color)
            quant.sel(quantile=0.5).plot(ax=ax_intensity, x='return_period', label=line_label,
                                         color=color, linewidth=2)
            ax_intensity.axhline(cc_rate*(temp-t_PI), linestyle='dashed', color=color)


        # decorate the axis
        ax_pr.set_title(f'{name} Probability Ratio', fontsize='small')
        ax_pr.set_xlabel('Accumulated Rain (mm)')
        ax_pr.set_ylabel('PR (%)')
        # ax_pr.axhline(100,linestyle='dashed',color='black')
        ax_pr.set_xlim(5, 60)
        ax_pr.set_ylim(100, 200)
        if ax_intensity is not None:
            ax_intensity.set_title(f'{name} Intensity Increase',fontsize='small')
            ax_intensity.set_xlabel('Return Period (summers)')
            ax_intensity.set_ylabel('Intensity Change (%)')
            ax_intensity.set_xscale('log')
            # compute the return periods for event today and plot them as median and 5-95% uncertainty.
            rp = 1.0 / (gev_r.xarray_gev_sf(radar_today_uncert[name].sel(rolling=rolling), rain_2020))
            rp_quant = rp.quantile([0.05, 0.5, 0.95], dim='sample')
            ax_intensity.axvspan(float(rp_quant.sel(quantile=0.05)), float(rp_quant.sel(quantile=0.95)), color='grey', alpha=0.5)
            ax_intensity.axvline(rp_quant.sel(quantile=0.5), linestyle='solid', color='black', linewidth=2)

            xticks = [5, 10, 20, 50, 100,200]
            ax_intensity.set_xticks(xticks)
            ax_intensity.set_xticklabels([str(tick) for tick in xticks])
for a in axs.flat:
    label.plot(a)
axs[0][-1].legend()
fig.show()
commonLib.saveFig(fig)

"""
        for rolling, linestyle in zip([1, 4], ['dashed', 'solid']):
            # plot uncert
            ax.fill_between(rain * rolling, quant.sel(quantile=0.05, rolling=rolling),
                            quant.sel(quantile=0.95, rolling=rolling), color='grey', alpha=0.5
                            )
            # median estimate
            ax.plot(rain * rolling, quant.sel(quantile=0.5, rolling=rolling), label=f'Rx{rolling:d}h PR ({name})',
                    linestyle=linestyle, linewidth=2, colorsys='black'
                    )
            # actual rtn values
            p  = gev_r.xarray_gev_sf(params.sel(rolling=rolling), float(radar_rain[name].sel(rolling=rolling)))
        event_rp_q = 1.0/p.quantile([0.05,0.5,0.95],dim='sample')
        ax.fill_between

        ax.axvline(event_rp_q.sel(quantile=0.5), linestyle=linestyle, color='black', linewidth=2 )


        # now compute the return periods for the PI and today

    p_today = gev_r.xarray_gev_sf(mnp, rain)
    p_p2k = gev_r.xarray_gev_sf(mnp * ratio_p2k, rain)
    pr = 100 * (p_today / p_pi)
    pr_p2k = 100 * (p_p2k / p_pi)
    # compute uncerts
    p_pi_uncert = gev_r.xarray_gev_sf(radar_fit[name].Parameters * ratio_uncerts, rain)
    p_today_uncert = gev_r.xarray_gev_sf(radar_fit[name].Parameters, rain)
    p_p2k_uncert = gev_r.xarray_gev_sf(radar_fit[name].Parameters * ratio_p2k_uncerts, rain)
    pr_uncert = 100 * (p_today_uncert / p_pi_uncert).quantile([0.05, 0.95], dim='sample')
    pr_p2k_uncert=100 *(p_p2k_uncert/p_pi_uncert).quantile([0.05,0.95],dim='sample')
    # and the intensity changes as a fun of rp
    pv = 1.0/rtn_period
    i_pi = gev_r.xarray_gev_isf(mnp * ratio, pv)
    i_today = gev_r.xarray_gev_isf(mnp, pv)
    i_p2k = gev_r.xarray_gev_isf(mnp * ratio_p2k, pv)
    ir = 100 * (i_today / i_pi) - 100
    ir_p2k = 100 * (i_p2k / i_pi) - 100
    i_pi_uncert = gev_r.xarray_gev_isf(radar_fit[name].Parameters * ratio_uncerts, pv)
    i_today_uncert = gev_r.xarray_gev_isf(radar_fit[name].Parameters, pv)
    i_p2k_uncert = gev_r.xarray_gev_isf(radar_fit[name].Parameters * ratio_p2k_uncerts, pv)
    ir_uncert = (100 * (i_today_uncert / i_pi_uncert) - 100).quantile([0.05, 0.95], dim='sample')
    ir_p2k_uncert = (100 * (i_p2k_uncert / i_pi_uncert) - 100).quantile([0.05, 0.95], dim='sample')
    # and the uncertainties.
    for rolling, linestyle in zip([1, 4], ['dashed', 'solid']):
        event_rp = float(
            1.0 / (gev_r.xarray_gev_sf(mnp.sel(rolling=rolling), float(radar_rain[name].sel(rolling=rolling)))))
        ax_pr.plot(pr.threshold*rolling,pr.sel(rolling=rolling), label=f'Ax{rolling:d}h PR (today)', linestyle=linestyle, color='blue',
                                     linewidth=2)
        ax_pr.fill_between(pr_uncert.threshold*rolling, pr_uncert.sel(quantile=0.05,rolling=rolling),
                           pr_uncert.sel(quantile=0.95,rolling=rolling),
                           alpha=0.5,color='blue')
        # ax_pr.plot(pr_p2k.threshold * rolling, pr_p2k.sel(rolling=rolling), label=f'Ax{rolling:d}h PR (+2K)',
        #            linestyle=linestyle, color='red',
        #            linewidth=2
        #            )
        #pr_p2k.sel(rolling=rolling).plot(ax=ax_pr, label=f'Rx{rolling:d}h PR (+2K)', linestyle=linestyle, color='red',
        #                                 linewidth=2)

        # ax_pr.fill_between(pr_p2k_uncert.threshold*rolling, pr_p2k_uncert.sel(quantile=0.05,rolling=rolling),
        #                    pr_p2k_uncert.sel(quantile=0.95,rolling=rolling),
        #                    alpha=0.5,color='red')
        ax_pr.axvline(radar_rain[name].sel(rolling=rolling)*rolling, linestyle=linestyle, color='black', linewidth=2)
        ir.sel(rolling=rolling).plot(ax=ax_intensity, x='return_period', label=f'Rx{rolling:d}h IR (today)',
                                     linestyle=linestyle, color='blue', linewidth=2)
        #ir_p2k.sel(rolling=rolling).plot(ax=ax_intensity, x='return_period', label=f'Rx{rolling:d}h IR (+2K)',
        #                                 linestyle=linestyle, color='red', linewidth=2)
        ax_intensity.fill_between(ir_uncert['return_period'], ir_uncert.sel(quantile=0.05,rolling=rolling),
                           ir_uncert.sel(quantile=0.95,rolling=rolling),
                           alpha=0.5,color='blue')
        ax_intensity.axvline(event_rp, linestyle=linestyle, color='black', linewidth=2)
    ax_pr.set_title(f'{name} Probability Ratio')
    ax_pr.set_xlabel('Accumulated Rain (mm)')
    ax_pr.set_ylabel('PR (%)')
    #ax_pr.axhline(100,linestyle='dashed',color='black')
    ax_pr.set_xlim(5,60)
    ax_pr.set_ylim(100,160)
    label.plot(ax_pr)

    ax_intensity.set_title(f'{name} Intensity Increase')
    ax_intensity.set_xlabel('Return Period (summers)')
    ax_intensity.set_ylabel('Intensity Change (%)')
    ax_intensity.set_xscale('log')
    ax_intensity.axhline(5.5,linestyle='dashed',color='black')
    xticks = [5,10,20,50,100]
    ax_intensity.set_xticks(xticks)
    ax_intensity.set_xticklabels([str(tick) for tick in xticks])
    label.plot(ax_intensity)

# now do the intensity ratios
axis[0][0].legend(ncols=2)
fig.show()
commonLib.saveFig(fig)
"""