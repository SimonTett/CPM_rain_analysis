# compute a bunch of GEV fits for the CPM data.
import numpy as np
import numpy.random

import xarray
import CPM_rainlib
import CPMlib
import commonLib
import pathlib
from R_python import gev_r


my_logger = CPM_rainlib.logger
commonLib.init_log(my_logger, level='INFO')


def get_jja_ts(path: pathlib.Path) -> xarray.Dataset:
    """
    Get the JJA time series from monthly-mean data in netcdf file
    :param path: path to data
    :return: dataset containing JJA data
    """
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist")
    ts = xarray.open_dataset(path, drop_variables=['t', 'surface'])
    # dropping t & surface as they come from the land/sea mask and cause trouble!
    ts = ts.resample(time='QS-DEC').mean()
    ts = ts.where(ts.time.dt.season == 'JJA', drop=True)
    # remove the 2008-2022 values before doing the fit
    ts = ts - ts.sel(**CPMlib.today_sel).mean()  # average over all ensembles
    return ts.load()


recreate_fit = False  # recreate all fits. Set to False to cache -- handy if want to add to existing
ts_dir = CPM_rainlib.dataDir / 'CPM_ts'
sim_cet = get_jja_ts(ts_dir / 'cet.nc').tas.rename('CET')
sim_reg_es = get_jja_ts(ts_dir / 'rgn_svp.nc')
sim_reg_tas = get_jja_ts(ts_dir / 'reg_tas.nc')
my_logger.info("Read in the time series data")

fit_dir = CPMlib.CPM_filt_dir / 'fits'
fit_dir.mkdir(exist_ok=True, parents=True)

raw_fit_dir = CPMlib.CPM_dir / 'fits'
raw_fit_dir.mkdir(exist_ok=True, parents=True)

ds = xarray.open_mfdataset(CPMlib.CPM_filt_dir.glob("**/CPM*11_30_23.nc"), parallel=True)
L = ds.time.dt.season == 'JJA'
maxRain = ds.seasonalMax.where(L, drop=True).sel(**CPMlib.carmont_rgn).load()

raw_ds = xarray.open_mfdataset(CPMlib.CPM_dir.glob("**/CPM*11_30_23.nc"), parallel=True)
L = raw_ds.time.dt.season == 'JJA'
raw_maxRain = raw_ds.seasonalMax.where(L, drop=True).sel(**CPMlib.carmont_rgn).load()

my_logger.info("Read in the max rain data")

stack_dim = dict(t_e=['time', "ensemble_member"])

fit_cov = dict(NoCov=[], CET=[sim_cet], CET_sqr=[sim_cet, (sim_cet ** 2).rename('CET_sqr')])
for root in ['', 'land_']:
    fit_cov[root + 'Rgn_t'] = [sim_reg_tas[root + 'tas'].rename(root + 'Rgn_t')]
    fit_cov[root + 'Rgn_t_sqr'] = [sim_reg_tas[root + 'tas'].rename(root + 'Rgn_t'),
                                   (sim_reg_tas[root + 'tas']**2).rename(root + 'Rgn_t_sqr')]
    fit_cov[root + 'Rgn_svp'] = [sim_reg_es[root + 'svp'].rename(root + 'Rgn_svp')]
    fit_cov[root + 'Rgn_svp_sqr'] = [sim_reg_es[root + 'svp'].rename(root + 'Rgn_svp'),
                                   (sim_reg_es[root + 'svp']**2).rename(root + 'Rgn_svp_sqr')]

fits=dict()
fits_raw=dict()
for name, cov in fit_cov.items():
    my_logger.info(f"Computing fits for {name}")
    cov_stack = [c.stack(**stack_dim) for c in cov]
    fits[name] = gev_r.xarray_gev(maxRain.stack(**stack_dim),
                            cov=cov_stack, dim='t_e', name=f'Rgn_{name}',
                            file=fit_dir / f'carmont_rgn_fit_{name}.nc', recreate_fit=recreate_fit
                            )
    fits_raw[name] = gev_r.xarray_gev(raw_maxRain.stack(**stack_dim),
                                cov=cov_stack, dim='t_e', name=f'Rgn_{name}',
                                file=raw_fit_dir / f'carmont_fit_raw_{name}.nc', recreate_fit=recreate_fit
                                )
## bootstrap.

my_logger.info("Computing bootstrap fits")
nboot = 100 # number of bootstrap samples
fit_cov = dict(NoCov=[], CET=[sim_cet])
rng = numpy.random.default_rng()
recreate_fit_boot = False
for name, cov in fit_cov.items():
    my_logger.info(f"Computing bootstrap fits for {name}")

    mx_rain_stack = maxRain.stack(**stack_dim)
    raw_mx_rain_stack = raw_maxRain.stack(**stack_dim)
    # Generating index and then doing calculations seems very expensive
    indx = rng.choice(mx_rain_stack.sizes['t_e'], (nboot, mx_rain_stack.sizes['t_e']), replace=True)
    # make indx  a dataarray
    indx = xarray.DataArray(indx, dims=['bootstrap', 't_e'], coords={'bootstrap': np.arange(nboot)})#,'t_e': mx_rain_stack.coords['t_e']})
    boot_rain = mx_rain_stack.isel(t_e=indx)

    cov_stack = [c.stack(**stack_dim).isel(t_e=indx) for c in cov]
    fits_boot = gev_r.xarray_gev(boot_rain,
                            cov=cov_stack, dim='t_e', name=f'Rgn_boot_{name}',
                            file=fit_dir / f'carmont_rgn_fit_boot_{name}.nc', recreate_fit=recreate_fit_boot,

                            )
    boot_rain = raw_mx_rain_stack.isel(t_e=indx) # big array. Don't want it twice.
    raw_fits_boot = gev_r.xarray_gev(boot_rain,verbose=True,
                            cov=cov_stack, dim='t_e', name=f'Rgn_raw_boot_{name}',
                            file=raw_fit_dir / f'carmont_fit_raw_boot_{name}.nc', recreate_fit=recreate_fit_boot,

                            )

# no stacking -- fit for each ensemble member
fit_cov = dict(NoCov=[], CET=[sim_cet])
for name, cov in fit_cov.items():
    my_logger.info(f"Computing per-ensemble fits for {name}")
    fits_ens = gev_r.xarray_gev(maxRain,
                            cov=cov, dim='time', name=f'Rgn_ens_{name}',
                            file=fit_dir / f'carmont_rgn_fit_ens_{name}.nc', recreate_fit=recreate_fit
                            )
    fits_raw_ens = gev_r.xarray_gev(raw_maxRain,
                                cov=cov, dim='time', name=f'Rgn_ens_{name}',
                                file=raw_fit_dir / f'carmont_fit_raw_ens_{name}.nc', recreate_fit=recreate_fit
                                )
my_logger.info(f"All Done")
