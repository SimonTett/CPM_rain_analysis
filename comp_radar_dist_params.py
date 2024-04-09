# compute the distribution parameters for GEV fit to radar data,
# This is done by sampling randomly from the quantiles of each distribution
# Then fit to that with weighting given by event size. Rise and repeat to get uncert.
import typing

import matplotlib.ticker
import xarray

import CPM_rainlib
import commonLib
from R_python import gev_r
import matplotlib.pyplot as plt
import numpy as np
import numpy.random
import CPMlib
import scipy.stats
import cartopy.crs as ccrs
my_logger=CPM_rainlib.logger
commonLib.init_log(my_logger)
use_cache = False # if True  just read in the saved fits (if they exist) otherwise generate the fits and save them
def comp_fits(rng, radar_events, nsamps: int = 100) -> xarray.DataArray:
    n_events = radar_events.EventTime.shape[0]
    rand_index = rng.integers(len(radar_events.quantv), size=(n_events, nsamps))
    wt = radar_events.count_cells
    fits = []
    for indx in range(0, nsamps):  # iterate over random samples
        ind = xarray.DataArray(rand_index[:, indx],
                               coords=dict(EventTime=radar_events.coords['EventTime'])
                               )
        sample = radar_events.max_precip.isel(quantv=ind).drop_vars('quantv').assign_coords(sample=indx)
        fits.append(gev_r.xarray_gev(sample, dim='EventTime', weights=wt))
        #fits.append(gev_r.xarray_gev(sample, dim='EventTime'))
    fits = xarray.concat(fits, dim='sample')
    return fits





radar_fit_dir = CPMlib.radar_dir/'radar_rgn_fit'
radar_fit_dir.mkdir(exist_ok=True,parents=True) # make sure the directory exists
radar_fit = dict()
radar_rain = dict()
radar_fit_uncert = dict()
rain_aug2020 = dict()
if use_cache: # use the cached data
    for path in radar_fit_dir.glob('*.nc'):
        name = path.stem
        if name.endswith('_fit'):
            radar_fit[name.replace('_fit','')] = xarray.load_dataset(path)
        elif name.endswith('_fit_uncert'):
            radar_fit_uncert[name.replace('_fit_uncert','')] = xarray.load_dataarray(path)
        elif name.endswith('_carmont_rain'):
            radar_rain[name.replace('_carmont_rain','')] = xarray.load_dataarray(path)
        elif name.endswith('_aug2020'):
            rain_aug2020[name.replace('_aug2020','')] = xarray.load_dataarray(path)
        else:
            my_logger.warning(f"Unknown file {path}. Ignoring")
else: # generate the data.
    rng = numpy.random.default_rng(123456)
    nsamps = 100
    projGB = ccrs.OSGB()
    carmont = CPMlib.carmont_rgn_OSGB.copy()
    carmont.update(time=slice('2020-06-01','2020-08-31')) # include summer 2020
    radar_events = list((CPMlib.radar_dir/'radar_events').glob('*.nc'))
    for  path in radar_events:
        summary_path = path.parent.parent / 'summary' / path.name.replace('events', 'summary')
        name = "_".join(path.stem.split("_")[2:])
        radar_events = xarray.load_dataset(path)
        rain, rainMx, topog = CPM_rainlib.get_radar_data(summary_path, region=carmont,height_range=slice(1, None),
                                                         topog_grid=radar_events.attrs['topog_grid'])
        rain = rain.where(rainMx.dt.strftime('%Y-%m-%d') == '2020-08-12')
        rain_aug2020[name] = rain
        radar_rain[name] = rain.sel(**CPMlib.carmont_drain_OSGB, method='nearest')

        radar_fit[name] = comp_fits(rng, radar_events, nsamps=nsamps)
        print(f"Computed fits for {name}")
        rolling_uncert=[]
        for r in radar_fit[name]['rolling']:
            mean = radar_fit[name].Parameters.sel(rolling=r).mean('sample')
            cov = radar_fit[name].sel(rolling=r).Cov.mean('sample')
            ps = scipy.stats.multivariate_normal(mean=mean,cov=cov)
            coords = dict(sample=np.arange(0, nsamps), parameter=['location', 'scale', 'shape'])
            uncert = xarray.DataArray(ps.rvs(size=nsamps), coords=coords).assign_coords(rolling=r)
            rolling_uncert.append(uncert)
        radar_fit_uncert[name] = xarray.concat(rolling_uncert, dim='rolling')

    # save the fits, uncertainties and Carmont rain

    for key in radar_fit.keys():
        radar_fit[key].to_netcdf(radar_fit_dir / f'{key}_fit.nc')
        radar_fit_uncert[key].to_netcdf(radar_fit_dir / f'{key}_fit_uncert.nc')
        radar_rain[key].to_netcdf(radar_fit_dir / f'{key}_carmont_rain.nc')
        rain_aug2020[key].to_netcdf(radar_fit_dir / f'{key}_aug2020.nc')
# now have data -- either because we generated it or because we read it back in!
# mn_fit = fits.mean('sample')Radar_rain_Mean
## now to plot return periods and their uncertainties.
