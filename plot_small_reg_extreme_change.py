# Plot and compute % change in small region for location and scale. Testing if evidence of intensification
import CPMlib
import CPM_rainlib
import matplotlib.pyplot as plt
import numpy as np
from R_python import gev_r
import xarray
import commonLib
def do_gev_fit(files,cet,rgn,t_today):
    logger.info(f'Opening {len(files)} files')
    ds = xarray.open_mfdataset(files, parallel=True).sel(**rgn)
    ds=ds.where(ds.time.dt.season == 'JJA',drop=True)
    mxRain=ds.seasonalMax.load()
    logger.info(f'Have {len(mxRain.time)} time points from {mxRain.time.min().values} to {mxRain.time.max().values}')
    mxRain_quant=mxRain.quantile(np.linspace(0,1,11),CPMlib.CPM_coords).load()
    # read in simulated CET

    cet= cet.where(cet.time==mxRain.time, drop=True)
    cet = cet.where(cet.ensemble_member==mxRain_quant.ensemble_member,drop=True).load()
    logger.info('Doing GEV fit')
    fit=gev_r.xarray_gev(mxRain_quant.stack(ensemble_time=['ensemble_member','time']),
                         cov=[(cet-t_today).stack(ensemble_time=['ensemble_member','time']).rename('CET')],
                         dim='ensemble_time')
    fit_spatial = gev_r.xarray_gev(mxRain.stack(ensemble_time=['ensemble_member','time']),
                         cov=[(cet-t_today).stack(ensemble_time=['ensemble_member','time']).rename('CET')],
                         dim='ensemble_time').mean(CPM_rainlib.cpm_horizontal_coords)
    logger.info('Done GEV fit')
    p_today = CPM_rainlib.comp_params(fit.Parameters, 0.0)
    p_p1k = CPM_rainlib.comp_params(fit.Parameters, 1.0)
    ratios =100*(p_p1k/p_today-1.0) # % sens per k warming,
    p_today_spatial = CPM_rainlib.comp_params(fit_spatial.Parameters, 0.0)
    p_p1k_spatial = CPM_rainlib.comp_params(fit_spatial.Parameters, 1.0)
    ratios_spatial =100*(p_p1k_spatial/p_today_spatial-1.0) # % sens per k warming,
    return ratios, ratios_spatial


logger = CPM_rainlib.logger
commonLib.init_log(logger, level='INFO')
# read in simulated CET
cet = xarray.load_dataset(CPM_rainlib.dataDir / 'CPM_ts' / 'cet.nc').tas# monthly means
cet = cet.resample(time='QS-DEC').mean('time')
obs_cet = commonLib.read_cet()
obs_cet = obs_cet.where(obs_cet.time.dt.season=='JJA',drop=True)
t_today = float(obs_cet.sel(**CPMlib.today_sel).mean())
rgn=dict(zip(CPMlib.CPM_coords,
             [slice(v-0.10,v+0.10) for v in CPMlib.carmont_drain.values()]))
files = list(CPMlib.CPM_filt_dir.glob("**/*11_30_23.nc"))
fits_filt = do_gev_fit(files,cet,rgn,t_today)

files = list(CPMlib.CPM_dir.glob("**/*11_30_23.nc"))
fits_nofilt = do_gev_fit(files,cet,rgn,t_today)
ratios, ratios_spatial = fits_filt[0], fits_filt[1]
ratios_nofilt, ratios_spatial_nofilt = fits_nofilt[0], fits_nofilt[1]
cc=5.5 # CC is roughly 5.5%/K CET
## now plot 1, 2 % 4 hr param changes
fig, axs = plt.subplots(1, 3, figsize=(8, 5),clear=True,num='carmot_gev_params_sens',layout='constrained',sharey=True,sharex=True)
cmap = plt.cm.get_cmap('GnBu', len(ratios['quantile'])+1)
for roll, ax in zip([1,2,4],axs):
    cm=ax.scatter(ratios.sel(rolling=roll,parameter='location'),ratios.sel(rolling=roll,parameter='scale'),
               c=ratios['quantile'],vmin=-0.1,vmax=1.1,cmap=cmap,ec='black',s=50,marker='o')
    cm2=ax.scatter(ratios_nofilt.sel(rolling=roll,parameter='location'),ratios_nofilt.sel(rolling=roll,parameter='scale'),
               c=ratios['quantile'],vmin=-0.1,vmax=1.1,cmap=cmap,ec='black',s=60,marker='s')
    ax.plot(ratios_spatial.sel(parameter='location',rolling=roll),ratios_spatial.sel(rolling=roll,parameter='scale'),'kx',ms=20)
    ax.plot(ratios_spatial_nofilt.sel(parameter='location', rolling=roll), ratios_spatial_nofilt.sel(rolling=roll, parameter='scale'),
            'k+', ms=20)
    ax.set_title(f'Rx{roll}h')
    ax.set_xlabel('Location $\Delta$ %/K')
    ax.set_ylabel('Scale $\Delta$ %/K')
    ax.axhline(cc,color='k',linestyle='--')
    ax.axvline(cc,color='k',linestyle='--')
    ax.set_ylim(0.,14)
fig.show()
fig.colorbar(cm,ax=axs,label='quantile',**CPMlib.kw_colorbar)
commonLib.saveFig(fig)