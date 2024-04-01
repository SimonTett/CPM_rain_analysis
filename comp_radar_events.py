# Generate event datasets from radar data
# also do GEV fit
import pathlib
import typing

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import CPMlib
import xarray
import CPM_rainlib
import numpy as np
import commonLib
import datetime
import scipy.stats
import cftime
import pandas as pd
from R_python import gev_r


def convert_to_cftime(date: np.datetime64,
                      date_type: typing.Callable = cftime.DatetimeGregorian,
                      has_year_zero: bool = True):
    date = pd.to_datetime(date)
    return date_type(date.year, date.month, date.day, date.hour, date.minute, date.second, date.microsecond,
                     has_year_zero=has_year_zero)


recreate = True
proj = ccrs.PlateCarree()
projGB = ccrs.OSGB()
carmont = CPMlib.carmont_rgn_OSGB.copy()
carmont.update(time=slice('2008-01-01', '2023-12-31'))

# get the  summer mean CET out and force its time's to be the same.
obs_cet = commonLib.read_cet()  # read in the obs CET
obs_cet_jja = obs_cet.sel(time=(obs_cet.time.dt.season == 'JJA'))
summary_files = ['5km_summary_2004_2023.nc', '1km_summary_2004_2023.nc','1km_summary_2004_2023_c2.nc',
                 '1km_summary_2004_2023_c4.nc','1km_summary_2004_2023_c5.nc','1km_summary_2004_2023_c8.nc']
summary_files = [CPMlib.radar_dir / f for f in summary_files]
for path, outpath, resoln, name in zip(summary_files,
                ['5km_events_2008_2023.nc', '1km_events_2008_2023.nc','1km_events_2008_2023_c2.nc',
                 '1km_events_2008_2023_c4.nc','1km_events_2008_2023_c5.nc','1km_events_2008_2023_c8.nc'],
                [5000, 1000, 2000, 4000, 5000,8000],
               ['5km hourly', '1km hourly',
                'coarsened 1km to 2km hourly', 'coarsened 1km to 4km hourly',
                'coarsened 1km  to 5km hourly','coarsened 1km to 8km hourly']):
    print(f'Processing {path} to {outpath} at {resoln} m resolution.')
    grid = int(resoln / 90.)
    radar_dataset = CPM_rainlib.comp_event_stats(path, region=carmont, topog_grid=grid,
                                                 height_range=slice(50., None))
    # add on the CET

    obs_cet = commonLib.read_cet()
    # restrict to summer and > 1800 -- so in the range of np.datetime64
    obs_cet_jja = obs_cet.where((obs_cet.time.dt.season == 'JJA') & (obs_cet.time.dt.year > 1800),drop=True)
    obs_cet_jja['time'] = obs_cet_jja.indexes['time'].to_datetimeindex()
    # From https://stackoverflow.com/questions/55786995/converting-cftime-datetimejulian-to-datetime/55787899
    # and demonstrating the calendar support is a bit dodgy in xarray. Sigh!
    # now get CET for all the event times
    msk = radar_dataset.t.notnull()
    cet_interp = obs_cet_jja.sel(time=radar_dataset.t,method='Nearest')
    radar_dataset['CET'] = cet_interp

    # add some attributes
    source_str = f'Processed NIMROD RADAR data from {name} to events using comp_radar_events.py on {datetime.datetime.now()}'
    radar_dataset.attrs.update(source=source_str)

    opath = CPMlib.radar_dir / f"radar_events/{outpath}"
    opath.parent.mkdir(exist_ok=True)
    radar_dataset.to_netcdf(opath)
    print(f"Saved {radar_dataset.attrs['source']} to {opath}")

## lets check there are OK and plot them
files = pathlib.Path(CPMlib.radar_dir / 'radar_events').glob("*.nc")
datasets = {}
for file in files:
    datasets[file.name] = xarray.load_dataset(file)

## now lets plot things on a common scale.
qv = 0.9
key = list(datasets.keys())[0]
fig, axis = plt.subplots(nrows=1, ncols=2, figsize=[11, 8], num='plt_radar_events', clear=True)
for r in datasets[key]['rolling']:
    bins = np.histogram_bin_edges(datasets[key].max_precip.sel(rolling=r, quantv=qv).dropna('EventTime'), 50)

    for name, ds in datasets.items():
        ds.max_precip.sel(quantv=qv, rolling=r).dropna('EventTime').plot.hist(bins=bins, density=True, histtype='step',
                                                                              label=name, ax=axis[0])
        p = scipy.stats.genextreme.fit(ds.max_precip.sel(quantv=qv, rolling=r).dropna('EventTime'))
        print(name, int(r), f"Shape: {p[0]:4.2f} Loc:{p[1]:4.1f} Scale:{p[2]:4.1f}")
axis[0].legend()
axis[0].set_yscale('log')

fig.tight_layout()
fig.show()
