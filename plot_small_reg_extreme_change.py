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


use_cache = False # if True load data from Cache
nsamp = 100

logger = CPM_rainlib.logger
commonLib.init_log(logger, level='INFO')

filt_carmont_file = CPMlib.CPM_filt_dir / 'carmont_rgn_fit.nc'
nofilt_carmont_file = CPMlib.CPM_dir / 'carmont_rgn_fit.nc'
filt_rand_carmont_file = CPMlib.CPM_filt_dir / 'carmont_rgn_rand_fit.nc'
nofilt_rand_carmont_file = CPMlib.CPM_dir / 'carmont_rgn_rand_fit.nc'
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

cc = 5.5  # CC is roughly 5.5%/K CET
## now plot 1, 2 % 4 hr param changes
fig, axs = plt.subplots(1, 3, figsize=(8, 5), clear=True, num='carmot_gev_params_sens', layout='constrained',
                        sharey=True, sharex=True
                        )
delta_rand_quand_filt = (sample_ratio_filt.Quant_ratio - sample_ratio_filt.Quant_ratio.isel(quantile=5))
mn_delta = delta_rand_quand_filt.mean('sample')
std_delta = delta_rand_quand_filt.std('sample')
mx_delta = mn_delta + 2 * std_delta
min_delta = mn_delta - 2 * std_delta
delta_quant = ratio_filt.Quant_ratio - ratio_filt.Quant_ratio.isel(quantile=5)
sig_change_filt = (delta_quant > mx_delta) | (delta_quant < min_delta)

delta_rand_quand_nofilt = (sample_ratio_nofilt.Quant_ratio - sample_ratio_nofilt.Quant_ratio.sel(quantile=0.5))
mn_delta = delta_rand_quand_nofilt.mean('sample')
std_delta = delta_rand_quand_nofilt.std('sample')
mx_delta = mn_delta + 2 * std_delta
min_delta = mn_delta - 2 * std_delta
delta_quant = ratio_nofilt.Quant_ratio - ratio_nofilt.Quant_ratio.sel(quantile=0.5)
sig_change_nofilt = (delta_quant > mx_delta) | (delta_quant < min_delta)

cmap = matplotlib.colormaps.get_cmap('GnBu')
q = ratio_filt['quantile'].values
norm = matplotlib.colors.Normalize(vmin=-0.1, vmax=1.1)
for roll, ax in zip([1, 2, 4], axs):
    size = xarray.where(sig_change_filt.sel(rolling=1).any('parameter'), 80, 30
                        )  # large where any param sig different, small where not.
    cm = ax.scatter(ratio_filt.Quant_ratio.sel(rolling=roll, parameter='location'),
                    ratio_filt.Quant_ratio.sel(rolling=roll, parameter='scale'),
                    c=ratio_filt['quantile'], cmap=cmap, norm=norm, ec='black', s=size, marker='o'
                    )
    size = xarray.where(sig_change_nofilt.sel(rolling=1).any('parameter'), 80, 30
                        )  # large where any param sig different, small where not.
    cm2 = ax.scatter(ratio_nofilt.Quant_ratio.sel(rolling=roll, parameter='location'),
                     ratio_nofilt.Quant_ratio.sel(rolling=roll, parameter='scale'),
                     c=ratio_nofilt['quantile'], cmap=cmap, norm=norm, ec='black', s=size, marker='s'
                     )
    ax.plot(ratio_filt.Mean_ratio.sel(parameter='location', rolling=roll),
            ratio_filt.Mean_ratio.sel(rolling=roll, parameter='scale'),
            mec='k', mfc='None',marker='o',ms=10, mew=3
            )
    ax.plot(ratio_nofilt.Mean_ratio.sel(parameter='location', rolling=roll),
            ratio_nofilt.Mean_ratio.sel(rolling=roll, parameter='scale'),
            mec='k', marker='s',mfc='None',ms=10, mew=3,
            )
    ax.set_title(f'Rx{roll}h')
    ax.set_xlabel(r'Location $\Delta$ %/K')
    ax.set_ylabel(r'Scale $\Delta$ %/K')
    ax.axhline(cc, color='k', linestyle='--')
    ax.axvline(cc, color='k', linestyle='--')
    ax.set_ylim(0., 14)
fig.show()
fig.colorbar(cm, ax=axs, label='quantile', **CPMlib.kw_colorbar)
commonLib.saveFig(fig)
