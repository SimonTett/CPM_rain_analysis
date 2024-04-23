# Generate event datasets from radar data
# also do GEV fit
from __future__ import annotations

import logging
import pathlib
import typing

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import CPMlib
from CPM_rainlib import common_data
import xarray
import CPM_rainlib
import numpy as np
import commonLib
import datetime
import scipy.stats
import cftime
import pandas as pd
from R_python import gev_r
import rioxarray
from commonLib import init_log

# logger = logging.getLogger(f"__name__")
logger = CPM_rainlib.logger

import re


def extract_resoln_coarsen(filename: str) -> typing.Tuple[int | None, int | None]:
    # largely from GitHub co-pilot.
    """
    Extracts the resolution and coarsening values from a given filename.

    The function uses regular expressions to search for patterns in the filename that match the expected format for
    resolution and coarsening values. The expected format is 'XXXX_nkm' or 'XXXX_nkm_cm.nc', where 'n' is the
    resolution and 'm' is the coarsening value. 'XXXX' can be any character.

    Parameters:
    filename (str): The filename from which to extract the resolution and coarsening values.
    Returns:
    tuple: A tuple containing the resolution and coarsening values. If no match is found, the function returns (None, None).
    """
    match = re.search(r'(\d+)km(?:_c(\d+))?\.nc', filename)
    if match:
        resoln = int(match.group(1))
        coarsen = int(match.group(2)) if match.group(2) else None
        return resoln, coarsen
    else:
        return None, None


def convert_to_cftime(
        date: np.datetime64, date_type: typing.Callable = cftime.DatetimeGregorian,
        has_year_zero: bool = True
):
    date = pd.to_datetime(date)
    return date_type(date.year, date.month, date.day, date.hour, date.minute, date.second, date.microsecond,
                     has_year_zero=has_year_zero
                     )


init_log(logger, level='INFO')
recreate = True

carmont = CPMlib.carmont_rgn_OSGB.copy()
carmont.update(time=slice('2008-01-01', '2023-12-31'))

# get the  summer mean CET out and force its time type to be np.datetime64
obs_cet = commonLib.read_cet()
# restrict to summer and > 1800 -- so in the range of np.datetime64. Sigh time coords are such a pain.
obs_cet_jja = obs_cet.where((obs_cet.time.dt.season == 'JJA') & (obs_cet.time.dt.year > 1800), drop=True)
# and demonstrating the calendar support is a bit dodgy in xarray. Sigh!
obs_cet_jja['time'] = obs_cet_jja.indexes['time'].to_datetimeindex()
summary_files = list(CPMlib.radar_dir.glob("summary/*.nc"))
for path in summary_files:
    root = '_'.join(path.stem.split('_')[2:])
    topog = xarray.load_dataarray(CPMlib.radar_dir / 'topography'/f'topog_{root}.nc').sel(**CPMlib.carmont_rgn_OSGB)
    resoln, coarsen = extract_resoln_coarsen(path.name)
    if resoln == 1:
        samples_per_day = 12 * 24  # 5 min samples for km resoln
    elif resoln == 5:
        samples_per_day = 4 * 24  # 15 min samples for 5km resoln
    else:
        raise ValueError(f"Unknown resolution {resoln} for {path}")
    if coarsen:
        name = f'Coarsened {resoln:d}km to {coarsen:d}km hourly'
    else:
        name = f'{resoln:d}km hourly'

    outpath = path.name.replace('summary', 'events')
    logger.info(f'Processing {path} to {outpath} at {resoln} km resolution with coarsening {coarsen}.')
    radar = xarray.open_dataset(path).sel(**carmont).where(topog > 0)
    seas_radar = radar.resample(time='QS-DEC',skipna=True).map(CPM_rainlib.time_process)
    L=radar.time.dt.season == 'JJA'
    seas_radar = seas_radar.where(L, drop=True).load()
    # need a different discretisation from CPM  as radar uses np.datetime64 times.
    grp = np.floor(((seas_radar.Radar_rain_MaxTime-np.datetime64('1970-01-01'))/np.timedelta64(1,'D'))).rename('EventTime')
    logger.info(f"Computed grouping for {name}")

    radar_events = CPM_rainlib.comp_events(seas_radar.Radar_rain_Max,
                                           seas_radar.Radar_rain_MaxTime, grp, topog, obs_cet_jja,source='RADAR')

    # add some attributes
    source_str = (f"""Processed {name} NIMROD RADAR data.
     From {path.name} to events using comp_radar_events.py on {datetime.datetime.now()}""")
    radar_events.attrs.update(source=source_str)
    opath = CPMlib.radar_dir / f"radar_events/{outpath}"
    opath.parent.mkdir(exist_ok=True)
    radar_events.to_netcdf(opath)
    print(f"Saved {radar_events.attrs['source']}\n to {opath}")

## lets check there are OK and print  out the shape/locn/scale values
files = pathlib.Path(CPMlib.radar_dir / 'radar_events').glob("*.nc")
datasets = {}
for file in files:
    datasets[file.name] = xarray.load_dataset(file)

qv = 0.5
key = list(datasets.keys())[0]
for r in datasets[key]['rolling']:
    for name, ds in datasets.items():
        p = scipy.stats.genextreme.fit(ds.max_precip.sel(quantv=qv, rolling=r).dropna('EventTime'))
        print(name, int(r), f"Shape: {p[0]:4.2f} Loc:{p[1]:4.1f} Scale:{p[2]:4.1f}")
