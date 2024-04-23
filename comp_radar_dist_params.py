# compute the distribution parameters for GEV fit to radar data,
# This is done by sampling randomly from the quantiles of each distribution
# Then fit to that with weighting given by event size. Rise and repeat to get uncert.
import typing

import matplotlib.ticker
import xarray

import CPM_rainlib
import commonLib
from R_python import gev_r

import numpy.random
import CPMlib

my_logger=CPM_rainlib.logger
commonLib.init_log(my_logger,level='INFO')
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
    fits = xarray.concat(fits, dim='sample')
    return fits


radar_fit_dir = CPMlib.radar_dir/'radar_rgn_fit'
radar_fit_dir.mkdir(exist_ok=True,parents=True) # make sure the directory exists
radar_fit = dict()

rng = numpy.random.default_rng(123456)
nsamps = 1000

radar_events = list((CPMlib.radar_dir/'radar_events').glob('*.nc'))
for  path in radar_events:
    summary_path = path.parent.parent / 'summary' / path.name.replace('events', 'summary')
    name = "_".join(path.stem.split("_")[2:])
    radar_events = xarray.load_dataset(path)
    radar_fit[name] = comp_fits(rng, radar_events, nsamps=nsamps).mean('sample')
    my_logger.info(f"Computed fits for {name}")


# save the fits

for key in radar_fit.keys():
    radar_fit[key].to_netcdf(radar_fit_dir / f'{key}_fit.nc')


