# Print out stat fits for GEV fits. WIll explore CET, CET+CET^2, regional temp & regional SVP
# will look at ks stat and AIC for micro region!
import pathlib
import typing

import pandas as pd

import CPM_rainlib
from R_python import gev_r
import CPMlib
import xarray
import commonLib
import numpy as np


def comp_mn(ds: xarray.Dataset):
    mn = ds.rolling(grid_latitude=5, grid_longitude=5, center=True).mean()
    mn = mn.sel(**CPMlib.carmont_drain, method='nearest').drop_vars(
        ['grid_latitude', 'grid_longitude', 'latitude', 'longitude'], errors='ignore'
    )
    return mn


def comp_land_mn(ds: xarray.Dataset, topog: xarray.DataArray):
    mn = ds.rolling(grid_latitude=5, grid_longitude=5, center=True).mean()
    mn = mn.sel(**CPMlib.carmont_drain, method='nearest').drop_vars(
        ['grid_latitude', 'grid_longitude', 'latitude', 'longitude'], errors='ignore'
    )
    mn = mn.where(topog > 1, drop=True)
    return mn





def load_dataset(f: pathlib.Path):
    ds = xarray.load_dataset(f)
    CPM_rainlib.fix_coords(ds)
    return ds


my_logger = CPM_rainlib.logger
commonLib.init_log(my_logger, level='INFO')


def sort_fn(path: pathlib.Path) -> str:
    key = path.stem
    if key == 'carmont_rgn_fit_NoCov':
        key = 'aaaaaa'
    elif key == 'carmont_rgn_fit_CET':
        key = 'aaaaab'
    elif key == 'carmont_rgn_fit_CET_sqr':
        key = 'aaaaac'
    else:
        pass
    return key


fit_files = sorted((CPMlib.CPM_filt_dir / 'fits').glob('carmont_rgn_fit_*.nc'), key=sort_fn)
raw_fit_files = sorted((CPMlib.CPM_dir / 'fits').glob('carmont_fit_raw_*.nc'), key=sort_fn)
fits = dict()
for f in fit_files:
    ds = load_dataset(f)
    fits[ds.name] = ds

raw_fits = dict()
for f in raw_fit_files:
    ds = load_dataset(f)
    raw_fits[ds.name] = ds

my_logger.info("Read in the fits")
# msk to land
topog = xarray.load_dataarray(CPM_rainlib.dataDir / 'cpm_topog_fix_c2.nc').sel(**CPMlib.carmont_rgn)

my_logger.info("Read in the topography")

# make dataframes of AIC and KS stats
# fix coords
vars_to_drop = ['t', 'surface', 'latitude', 'longitude']
fn = lambda da: da.where(topog > 0).drop_vars(vars_to_drop, errors='ignore').mean(CPM_rainlib.cpm_horizontal_coords)
AIC = pd.concat([fn(f.AIC).rename(key[4:]).to_dataframe() for key, f in fits.items()], axis=1)
AIC_raw = pd.concat([fn(f.AIC).rename(key[4:]).to_dataframe() for key, f in raw_fits.items()], axis=1)
KS = pd.concat([fn(f.KS).rename(key[4:]).to_dataframe() for key, f in fits.items()], axis=1)
KS_raw = pd.concat([fn(f.KS).rename(key[4:]).to_dataframe() for key, f in raw_fits.items()], axis=1)

## generate tables

rename = dict(NoCov='No Cov.', CET='C', CET_sqr='C+C$^2$', Rgn_t='T', Rgn_svp='S',
              land_Rgn_t='L. T', land_Rgn_svp='L. S',Rgn_t_sqr='T+T$^2$', Rgn_svp_sqr='S+S$^2$',
                land_Rgn_t_sqr='L. T+T$^2$', land_Rgn_svp_sqr='L. S+S$^2$'
              )
# print out the AIC values
with open(CPMlib.table_dir / 'aic.tex', 'w') as f:
    AIC.rename(columns=rename).rename_axis('').style. \
        relabel_index([f'Rx{r:d}h' for r in AIC.index]).format(precision=0).to_latex(
        f, label='tab:aic', hrules=True, position='ht!',
        caption="""Mean AIC values for Carmont Region for different rainfall season maxes and covariates: 
            No Covariate (No Cov.), CET (C), CET and CET$^2$ (C+C$^2$), 
            Regional mean SVP (S), SVP and SVP$^2$ (S+S$^2$), Regional mean Temperature (T), Temp and Temp$^2$ (T+T$^2$).
            Land only Regional means are show with L."""
    )
# and the KS values
with open(CPMlib.table_dir / 'ks.tex', 'w') as f:
    KS.rename(columns=rename).rename_axis('').style. \
        relabel_index([f'Rx{r:d}h' for r in AIC.index]). \
        format(precision=2).to_latex(
        f, label='tab:ks', position='ht!',
        caption='Mean KS values for Carmont Region. Columns as Table~\\ref{tab:aic}.',
        hrules=True
    )
