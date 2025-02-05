# Plot and compute % change in small region for location and scale. Testing if evidence of intensification
import copy
import pathlib
import typing

import CPMlib
import CPM_rainlib
import matplotlib.pyplot as plt
import numpy as np
from R_python import gev_r
import xarray
import commonLib
import matplotlib


def get_mxRain(files: typing.List[pathlib.Path], rgn: dict) -> xarray.DataArray:
    logger.info(f'Opening {len(files)} files')
    ds = xarray.open_mfdataset(files, parallel=True).sel(**rgn)
    ds = ds.where(ds.time.dt.season == 'JJA', drop=True)
    mxRain = ds.seasonalMax.load()
    logger.info(f'Have {len(mxRain.time)} time points from {mxRain.time.min().values} to {mxRain.time.max().values}')
    return mxRain


def comp_ratio_gev(
        mxRain: xarray.DataArray,
        cet: xarray.DataArray,
        t_today: float,
) -> \
        xarray.Dataset:
    
    def comp_ratio(fit: xarray.Dataset) -> xarray.DataArray:
        p_today = CPM_rainlib.comp_params(fit.Parameters, 0.0)
        p_p1k = CPM_rainlib.comp_params(fit.Parameters, 1.0)
        ratios = 100 * (p_p1k / p_today - 1.0)  # % sens per k warming,
        return ratios

    mxRain_quant = mxRain.quantile(np.linspace(0, 1, 11), CPMlib.CPM_coords).load()

    logger.info('Doing GEV fit')
    fit = gev_r.xarray_gev(mxRain_quant.stack(ensemble_time=['ensemble_member', 'time']),
                           cov=[(cet - t_today).stack(ensemble_time=['ensemble_member', 'time']).rename('CET')],
                           dim='ensemble_time'
                           )
    ratios = comp_ratio(fit)
    fit_spatial = gev_r.xarray_gev(mxRain.stack(ensemble_time=['ensemble_member', 'time']),
                                   cov=[(cet - t_today).stack(ensemble_time=['ensemble_member', 'time']).rename('CET')],
                                   dim='ensemble_time'
                                   ).mean(CPM_rainlib.cpm_horizontal_coords)
    ratios_spatial = comp_ratio(fit_spatial)
    logger.info('Done GEV fits')

    return xarray.merge([ratios.rename('Quant_ratio'), ratios_spatial.rename('Mean_ratio')])


use_cache = True # if True load data from Cache
nsamp = 100

logger = CPM_rainlib.logger
commonLib.init_log(logger, level='INFO')

filt_carmont_file = CPMlib.CPM_filt_dir / 'fits/carmont_rgn_fit_CET.nc'
nofilt_carmont_file = CPMlib.CPM_dir / 'fits/carmont_fit_raw_CET.nc'
filt_rand_carmont_file = CPMlib.CPM_filt_dir / 'fits/carmont_rgn_rand_fit.nc'
nofilt_rand_carmont_file = CPMlib.CPM_dir / 'fits/carmont_rgn_rand_fit.nc'
if use_cache:
    ratio_filt = xarray.load_dataset(filt_carmont_file)
    ratio_nofilt = xarray.load_dataset(nofilt_carmont_file)
    sample_ratio_filt = xarray.load_dataset(filt_rand_carmont_file)
    sample_ratio_nofilt = xarray.load_dataset(nofilt_rand_carmont_file)
else:  # read in data and process.
    # read in simulated CET
    cet = xarray.load_dataset(CPM_rainlib.dataDir / 'CPM_ts' / 'cet.nc').tas  # monthly means
    cet = cet.resample(time='QS-DEC').mean('time')
    cet = cet.where(cet.time.dt.season == 'JJA', drop=True)

    obs_cet = commonLib.read_cet()
    obs_cet = obs_cet.where(obs_cet.time.dt.season == 'JJA', drop=True)
    t_today = float(obs_cet.sel(**CPMlib.today_sel).mean())
    rgn = dict(zip(CPMlib.CPM_coords,
                   [slice(v - 0.10, v + 0.10) for v in CPMlib.carmont_drain.values()]
                   )
               )

    files = list(CPMlib.CPM_filt_dir.glob("**/*11_30_23.nc"))
    mxRain_filt = get_mxRain(files, rgn)
    ratio_filt = comp_ratio_gev(mxRain_filt, cet, t_today)
    ratio_filt.to_netcdf(filt_carmont_file)

    # generate random fits
    cet_copy = copy.deepcopy(cet)
    rng = np.random.default_rng(123456)
    samp = []
    for s in range(nsamp):
        rng.shuffle(cet_copy.values, axis=1)  # shuffle the time axis
        fits = comp_ratio_gev(mxRain_filt, cet_copy, t_today).assign_coords(sample=s)
        samp.append(fits)

    sample_ratio_filt = xarray.concat(samp, dim='sample')
    sample_ratio_filt.to_netcdf(filt_rand_carmont_file)

    files = list(CPMlib.CPM_dir.glob("**/*11_30_23.nc"))
    mxRain_nofilt = get_mxRain(files, rgn)
    ratio_nofilt = comp_ratio_gev(mxRain_nofilt, cet, t_today)
    ratio_nofilt.to_netcdf(nofilt_carmont_file)
    samp = []
    for s in range(nsamp):
        rng.shuffle(cet_copy.values, axis=1)  # shuffle the time axis
        fits = comp_ratio_gev(mxRain_nofilt, cet_copy, t_today).assign_coords(sample=s)
        samp.append(fits)
    sample_ratio_nofilt = xarray.concat(samp, dim='sample')
    sample_ratio_nofilt.to_netcdf(nofilt_rand_carmont_file)

cc = CPMlib.cc_dist.mean()  # CC is roughly 5.5%/K CET
## now plot 1, 2 % 4 hr param changes
fig, axs = plt.subplots(1, 3, figsize=(8, 4), clear=True, num='carmont_gev_quant_change', layout='constrained',
                        sharey=True, sharex=True
                        )
label   = commonLib.plotLabel()

def comp_vars(sample_ratio:xarray.DataArray,
              ratio:xarray.DataArray,
              sd_scale:float = 3.0) -> (xarray.Dataset,xarray.Dataset,xarray.Dataset):
    """
    Compute uncertainties for Monte Carlo sample ratio at each quantile - mean.
    Returns 
    1) boolian array which  is True where 
      deteriminstic values at each quantile - mean value are: 
          > sd_scale * sigma + sample mean.  
          OR
          < sample_mean - sd_scale * sigma
      AND
      determinisic values > random sample 2 sigma + sample_mean for each quant
      
    2) max values from MC data
    3) Err from MC value -- sd_scale * sigma 
      
    """
    
    param_sel = dict(parameter=['location','scale'])
    
    #sample_rat = sample_ratio.Quant_ratio - sample_ratio.Quant_ratio.sel(quantile=0.5)
    sample_rat = sample_ratio.Quant_ratio - sample_ratio.Mean_ratio
    sample_rat = sample_rat.sel(param_sel)
    mean_delta = sample_rat.mean('sample')
    std_delta = sample_rat.std('sample')
    err = sd_scale * std_delta
    mx_delta = mean_delta + err
    mn_delta  = mean_delta - err
    #rat = ratio.Quant_ratio-ratio.Quant_ratio.sel(quantile=0.5)
    rat = ratio.Quant_ratio - ratio.Mean_ratio
    rat = rat.sel(param_sel)
    sig_change = (rat   >= mx_delta) | (rat <= mn_delta)
    # Greater than random distribution
    sample_rat = sample_ratio.Quant_ratio.sel(param_sel)
    mx_values =  sample_rat.mean('sample')+sd_scale*sample_rat.std('sample')
    sig_change = (ratio_filt.Quant_ratio.sel(param_sel) >= mx_values) & sig_change
    return sig_change, mx_values,err 

sig_change_filt,mx_values_filt,err_filt = comp_vars(sample_ratio_filt,ratio_filt)
sig_change_nofilt,mx_values_nofilt,err_nofilt = comp_vars(sample_ratio_nofilt,ratio_nofilt)


cmap = matplotlib.colormaps.get_cmap('GnBu')
#q = ratio_filt['quantile'].values
norm = matplotlib.colors.Normalize(vmin=-0.1, vmax=1.1)
names=[]

for roll, ax in zip([1, 2, 4], axs):

    msk = sig_change_filt.sel(rolling=roll).any('parameter')
    size = xarray.where(msk, 12, 8                         )  # large where any param sig different, small where not.
    rat = ratio_filt.Quant_ratio.sel(rolling=roll)
    ax.errorbar(x=rat.sel(parameter='location'),y=rat.sel(parameter='scale'),
                xerr=err_filt.sel(parameter='location',rolling=roll),
                yerr=err_filt.sel(parameter='scale',rolling=roll),color='black',
                linestyle='None')
    bbox=dict(facecolor='white', edgecolor='none',pad=0.0)
    for q,loc,scale,sz in zip(rat['quantile'], rat.sel( parameter='location'), rat.sel( parameter='scale'),size):
       ax.text(loc,scale,f'{int(q*10):1d}',fontsize=sz,ha='center',
               va='center',color='black',weight='bold',bbox=bbox)
    ax.plot(mx_values_filt.sel(rolling=roll,parameter='location'),
            mx_values_filt.sel(rolling=roll,parameter='scale'),color='black',marker='x')
    ax.plot(sample_ratio_filt.Quant_ratio.sel(rolling=roll,parameter='location').mean('sample'),
            sample_ratio_filt.Quant_ratio.sel(rolling=roll,parameter='scale').mean('sample'),
            marker='',linestyle='dashed',color='black',
            )

    ax.plot(ratio_filt.Mean_ratio.sel(parameter='location', rolling=roll),
            ratio_filt.Mean_ratio.sel(rolling=roll, parameter='scale'),
            mec='black', mfc='None',marker='o',ms=8, mew=3
            ) # plot the Mean_ratio



    
    
    msk = sig_change_nofilt.sel(rolling=roll).any('parameter')
    size_nofilt = xarray.where(msk, 12, 8  )  # large where any param sig different, small where not.
    rat = ratio_nofilt.Quant_ratio.sel(rolling=roll)
    # no error bar for non filtered data. Plot too busy with it in. 
# =============================================================================
#     ax.errorbar(x=rat.sel(parameter='location'),y=rat.sel(parameter='scale'),
#                 xerr=err_nofilt.sel(parameter='location',rolling=roll),
#                 yerr=err_nofilt.sel(parameter='scale',rolling=roll),color='red',
#                 linestyle='None')
# =============================================================================
    for q,loc,scale,sz in zip(rat['quantile'], rat.sel( parameter='location'), 
                              rat.sel( parameter='scale'),size_nofilt):
       ax.text(loc,scale,f'{int(q*10):1d}',fontsize=sz,ha='center',va='center',color='purple',weight='bold')
    
# =============================================================================
#     ax.plot(mx_values_nofilt.sel(rolling=1,parameter='location'),
#             mx_values_nofilt.sel(rolling=1,parameter='scale'),color='red')
# =============================================================================


    ax.plot(ratio_nofilt.Mean_ratio.sel(parameter='location', rolling=roll),
            ratio_nofilt.Mean_ratio.sel(rolling=roll, parameter='scale'),
            mec='purple', marker='o',mfc='None',ms=8, mew=3
            )
    ax.set_title(f'Rx{roll}h')
    ax.set_xlabel(r'Location $\Delta$ %/K')
    ax.set_ylabel(r'Scale $\Delta$ %/K')
    ax.axhline(cc, color='k', linestyle='--')
    ax.axhline(cc*2, color='red', linestyle='--') # twice CC
    ax.axvline(cc, color='k', linestyle='--')
    ax.set_ylim(0., 14)
    ax.set_xlim(0,7)
    label.plot(ax)
   # breakpoint()
fig.show()
#fig.colorbar(cm, ax=axs, label='quantile', **CPMlib.kw_colorbar)
commonLib.saveFig(fig,figtype=['pdf','png'])
