# plot the ensemble avg and all data fits
import xarray
import CPMlib
import numpy as np
nroll = 5 # default should be 5
fit_dir = CPMlib.CPM_filt_dir / 'fits'
ens_fit_file=fit_dir / f'carmont_rgn_fit_ens_CET.nc'
ens_cpm_gev_params = xarray.open_dataset(ens_fit_file).rolling(grid_latitude=nroll, grid_longitude=nroll, center=True).mean()
ens_cpm_gev_params = ens_cpm_gev_params.sel(**CPMlib.carmont_drain, method='nearest'
                                            )  # get the data for the carmont drain
ens_cpm_gev_params = ens_cpm_gev_params.Parameters.load()
# do the sane for the stacked data
fit_file=fit_dir / f'carmont_rgn_fit_CET.nc'
cpm_gev_params = xarray.open_dataset(fit_file).rolling(grid_latitude=nroll, grid_longitude=nroll, center=True).mean()
cpm_gev_params = cpm_gev_params.sel(**CPMlib.carmont_drain, method='nearest'
                                            )  # get the data for the carmont drain
cpm_gev_cov = (cpm_gev_params.Cov/(nroll**2)).load()
print('from cov)',np.sqrt(np.diag(cpm_gev_cov.sel(rolling=1))))
cpm_gev_params = cpm_gev_params.Parameters.load()
sd = xarray.open_dataset(fit_file).Parameters.\
    rolling(grid_latitude=nroll, grid_longitude=nroll, center=True).std().\
    sel(**CPMlib.carmont_drain, method='nearest').load()
sd = sd.load()
print('spatial sd',sd.sel(rolling=1).values)
with np.printoptions(precision=1):
    for p in ['location','scale']:
        ens_fract_change = ens_cpm_gev_params.sel(rolling=1,parameter=f'D{p}_CET')*100/ens_cpm_gev_params.sel(rolling=1,parameter=p)
        fract_change = cpm_gev_params.sel(rolling=1,parameter=f'D{p}_CET')*100/cpm_gev_params.sel(rolling=1,parameter=p)

        print(p,
              f'{ens_fract_change.mean().values:3.1f}',
              f'{ens_fract_change.std().values/np.sqrt(11.):3.2f}',
              f'{fract_change.values:3.1f}',
              ens_fract_change.values)
        
# get in the bootstrap results
boots_fit_file = fit_dir/f'carmont_rgn_fit_boot_CET.nc'
boots_cpm_gev_params = xarray.open_dataset(boots_fit_file).rolling(grid_latitude=nroll, grid_longitude=nroll, center=True).mean()
boots_cpm_gev_params = boots_cpm_gev_params.sel(**CPMlib.carmont_drain, method='nearest'
                                            )  # get the data for the carmont drain
boots_sd = boots_cpm_gev_params.Parameters.std('bootstrap').load()
print(boots_sd.sel(rolling=1).values)

# Get in data near carmont..
fit_file=fit_dir / f'carmont_rgn_fit_CET.nc'
rgn = {k:slice(v-0.1,v+0.1) for k,v in CPMlib.carmont_drain.items()}
cpm_gev_params = xarray.open_dataset(fit_file).rolling(grid_latitude=nroll, grid_longitude=nroll, center=True).mean()
cpm_gev_params = cpm_gev_params.sel(**CPMlib.carmont_drain, method='nearest'
                                            )  # get the data for the carmont drain