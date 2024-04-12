# Print out stat fits for GEV fits. WIll explore CET, CET+CET^2, regional temp & regional SVP
# will look at ks stat and AIC for micro region!
import pathlib

import pandas as pd

import CPM_rainlib
from R_python import gev_r
import CPMlib
import xarray
import commonLib
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm


def get_jja_ts(path: pathlib.Path) -> xarray.Dataset:
    """
    Get the JJA time series from monthly-mean data in netcdf file
    :param path: path to data
    :return: dataset containing JJA data
    """
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist")
    ts = xarray.open_dataset(path)
    ts = ts.resample(time='QS-DEC').mean()
    ts = ts.where(ts.time.dt.season == 'JJA', drop=True)
    # remove the 2008-2022 values before doing the fit
    ts = ts - ts.sel(**CPMlib.today_sel).mean()  # average over all ensembles
    return ts


my_logger = CPM_rainlib.logger
commonLib.init_log(my_logger, level='INFO')
rgn = {key: slice(value - 0.4, value + 0.4) for key, value in CPMlib.carmont_drain.items()}
recreate_fit = False  # set False to use cache
ts_dir = CPM_rainlib.dataDir / 'CPM_ts'
sim_cet = get_jja_ts(ts_dir / 'cet_tas.nc').tas.rename('CET')
sim_reg_es = get_jja_ts(ts_dir / 'rgn_svp.nc').tas.rename('RegSVP')
sim_reg_tas = get_jja_ts(ts_dir / 'reg_tas.nc').tas.rename('RegTemp')
my_logger.info("Read in the time series data")

fit_dir = CPMlib.CPM_filt_dir / 'fits'
fit_dir.mkdir(exist_ok=True, parents=True)

raw_fit_dir = CPMlib.CPM_dir / 'fits'
raw_fit_dir.mkdir(exist_ok=True, parents=True)

ds = xarray.open_mfdataset(CPMlib.CPM_filt_dir.glob("**/CPM*11_30_23.nc"), parallel=True)
L = ds.time.dt.season == 'JJA'
maxRain = ds.seasonalMax.where(L, drop=True).sel(**rgn).load()

raw_ds = xarray.open_mfdataset(CPMlib.CPM_dir.glob("**/CPM*11_30_23.nc"), parallel=True)
L = raw_ds.time.dt.season == 'JJA'
raw_maxRain = raw_ds.seasonalMax.where(L, drop=True).sel(**rgn).load()

my_logger.info("Read in the max rain data")
stack_dim = dict(t_e=['time', "ensemble_member"])
fits = dict()
fits_raw = dict()
for name, cov in zip(['NoCov', 'CET', 'CET_sqr', 'Rgn_t', 'Rgn_svp'],
                     [[], [sim_cet], [sim_cet, (sim_cet ** 2).rename('CET_sqr')], [sim_reg_tas], [sim_reg_es]]
                     ):
    cov_stack = [c.stack(**stack_dim) for c in cov]
    fits[name] = gev_r.xarray_gev(maxRain.stack(**stack_dim),
                                  cov=cov_stack, dim='t_e', name=f'Rgn_{name}',
                                  file=fit_dir / f'carmont_rgn_fit_{name}.nc', recreate_fit=recreate_fit
                                  )
    fits_raw[name] = gev_r.xarray_gev(raw_maxRain.stack(**stack_dim),
                                      cov=cov_stack, dim='t_e', name=f'Rgn_{name}',
                                      file=raw_fit_dir / f'carmont_rgn_fit_{name}.nc', recreate_fit=recreate_fit
                                      )
    my_logger.info(f"Computed fits for {name}")


# make dataframes of AIC and KS stats
def comp_mn(ds: xarray.Dataset):
    mn = ds.rolling(grid_latitude=5, grid_longitude=5, center=True).mean()
    mn = mn.sel(**CPMlib.carmont_drain, method='nearest').drop_vars(
        ['grid_latitude', 'grid_longitude', 'latitude', 'longitude'], errors='ignore'
        )
    return mn


AIC = pd.concat([comp_mn(f.AIC).rename(key).to_dataframe() for key, f in fits.items()], axis=1)
AIC_raw = pd.concat([comp_mn(f.AIC).rename(key).to_dataframe() for key, f in fits_raw.items()], axis=1)

KS = pd.concat([comp_mn(f.KS).rename(key).to_dataframe() for key, f in fits.items()], axis=1)
KS_raw = pd.concat([comp_mn(f.KS).rename(key).to_dataframe() for key, f in fits_raw.items()], axis=1)
## generate tables

rename = dict(NoCov='No Covariate', CET_sqr='CET+CET$^2$', Rgn_t='Regn. T', Rgn_svp='Regn. SVP')
# print out the AIC values
with open(CPMlib.table_dir / 'aic.tex', 'w') as f:
    AIC.rename(columns=rename).rename_axis('').style.\
        relabel_index([f'Rx{r:d}h' for r in AIC.index]).format(precision=0).to_latex(
            f, label='tab:aic',
            caption="Mean AIC values for Carmont Drain",hrules=True
        )
# and the KS values
with open(CPMlib.table_dir / 'ks.tex', 'w') as f:
    KS.rename(columns=rename).rename_axis('').style. \
        relabel_index([f'Rx{r:d}h' for r in AIC.index]).\
        format(precision=2).to_latex(f, label='tab:ks',caption='Mean KS values for Carmont Drain',hrules=True)
