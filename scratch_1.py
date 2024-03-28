# test that monthly to seasonal processing works..
from rioxarray import rioxarray

import CPMlib
import pathlib
import xarray
import typing
from CPM_rainlib import common_data,quants_locn
import numpy as np
import matplotlib.pyplot as plt


def time_process(data_set):
    """
    Process a dataset of (daily) data
    :param data_set -- Dataset to process

    """
    mn = data_set['Radar_rain_Mean']
    mx = data_set['Radar_rain_Max'].max('time', keep_attrs=True)  # max of maxes
    mx_idx = data_set['Radar_rain_Max'].fillna(0.0).argmax('time', skipna=True)  # index  of max
    mx_time = data_set['Radar_rain_MaxTime'].isel(time=mx_idx).drop_vars('time')
    time_bounds = xarray.DataArray([mn.time.min().values, mn.time.max().values], coords=dict(bounds=[0, 1])).rename(
        'time_bounds')
    mn = mn.mean('time', keep_attrs=True)
    no_samples = data_set['No_samples'].sum('time', keep_attrs=True)
    ds = xarray.merge([mn, mx, mx_time, time_bounds, no_samples])

    return ds


def get_radar_data(file: pathlib.Path, topog_grid: typing.Optional[int] = None, region: typing.Optional[dict] = None,
                   height_range: slice = slice(50, None), max_total_rain: float = 1000.) -> (
xarray.DataArray, xarray.DataArray, xarray.DataArray):
    """
    read in radar data and mask it by heights and mean rain being reasonable.
    :param topog_grid: Grid for topography
    :param file: file to be read in
    :param region: region to extract. Can include spatial and temporal co-ords
    :param height_range: Height range to use (all data strictly *between* these values will be used)
    :param max_total_rain: the maximum  rain allowed (QC)
    :return: Data masked for region requested & mxTime
    """

    radar_precip = xarray.open_dataset(file)  #
    if region is not None:
        radar_precip = radar_precip.sel(**region)
    rseas = radar_precip.resample(time='QS-Dec').map(time_process)
    rseas = rseas.sel(time=(rseas.time.dt.season == 'JJA'))  # Summer max rain
    topog = read_90m_topog(region=region, resample=topog_grid)  # read in topography and regrid
    top_fit_grid = topog.interp_like(rseas.isel(time=0).squeeze())
    htMsk = True
    if height_range.start:
        htMsk = htMsk & (top_fit_grid > height_range.start)
    if height_range.stop:
        htMsk = htMsk & (top_fit_grid < height_range.stop)

    # mask by seasonal sum < 1000.
    mskRain = ((rseas['Radar_rain_Mean'] * (30 + 31 + 31) / 4.) < max_total_rain) & htMsk
    rseasMskmax = rseas['Radar_rain_Max'].where(mskRain)
    mxTime = rseas['Radar_rain_MaxTime'].where(mskRain)

    return (rseasMskmax, mxTime, top_fit_grid)

def read_90m_topog(region: typing.Optional[dict] = None, resample=None):
    """
    Read 90m DEM data from UoE regridded OSGB data.
    Fix various problems
    :param region: region to select to.
    :param resample: If not None then the amount to coarsen by.
    :return: topography dataset
    """
    topog = rioxarray.open_rasterio(common_data / 'uk_srtm')
    topog = topog.reindex(y=topog.y[::-1]).rename(
        x='projection_x_coordinate', y='projection_y_coordinate')
    rgn = region.copy()
    rgn.pop('time')
    if region is not None:
        topog = topog.sel(**rgn)
    topog = topog.load().squeeze()
    L = (topog > -10000) & (topog < 10000)  # fix bad data. L is where data is good!
    topog = topog.where(L)
    if resample is not None:
        topog = topog.coarsen(projection_x_coordinate=resample, projection_y_coordinate=resample, boundary='pad').mean()
    return topog

def comp_event_stats(file: pathlib.Path,
                     topog_grid: typing.Optional[int] = None,
                     region: typing.Optional[dict] = None,
                     height_range: slice = slice(0, 200),
                     max_total_rain: float = 1000.,
                     source: str = 'radar'):
    """
    read in radar data and then group by day.
    :param file: File to be read in
    :param region: Region to extract.
    :param height_range: Height range to use (all data strictly *between* these values will be used)
    :param time_range: Time range to use
    :param max_total_rain: the maximum mean rain allowed (QC)
    :return: grouped dataset
    """
    rseasMskmax, mxTime, topog = get_radar_data(file, region=region,
                                                topog_grid=topog_grid,
                                                height_range=height_range,
                                                max_total_rain=max_total_rain)
    # loop over rolling co-ords
    radar_data=[]
    for r in mxTime['rolling']:
        msk_max = rseasMskmax.sel(rolling=r)
        max_time = mxTime.sel(rolling=r)
        grp = ((max_time.dt.dayofyear - 1) + (max_time.dt.year-1980)*1000).rename('EventTime')
        grp = grp.where(~rseasMskmax.isnull(),  0).rename('EventTime')
        dataset = event_stats(msk_max, max_time, grp, source=source).\
            sel(EventTime=slice(1, None))
        ht = topog.sel(
                 projection_x_coordinate=dataset.x,
                 projection_y_coordinate=dataset.y)
        # drop unnneded coords
        coords_to_drop = [c for c in ht.coords if c not in ht.dims]
        ht = ht.drop_vars(coords_to_drop)
        dataset['height'] = ht
        radar_data.append(dataset)
    breakpoint()
    radar_dataset=xarray.concat(radar_data,'rolling')
    return radar_dataset

def event_stats(max_precip: xarray.Dataset,
                max_time: xarray.Dataset,
                group,
                source: str = "CPM"):
    if source == 'CPM':
        x_coord = 'grid_longitude'
        y_coord = 'grid_latitude'
    elif source == 'radar':
        x_coord = 'projection_x_coordinate'
        y_coord = 'projection_y_coordinate'
    else:
        raise ValueError(f"Unknown source {source}")

    ds = xarray.Dataset(dict(maxV=max_precip, maxT=max_time))
    grper = ds.groupby(group)
    quantiles = np.linspace(0, 1, 21)
    dataSet = grper.map(quants_locn, quantiles=quantiles, x_coord=x_coord, y_coord=y_coord).rename(quant='max_precip')
    grper2 = max_precip.groupby(group)
    count = grper2.count().rename("# Cells")
    dataSet['count_cells'] = count
    return dataSet

carmont = CPMlib.carmont_rgn_OSGB.copy()
carmont.update(time=slice('2008-01-01', '2023-12-31'))
summary_files = ['5km_summary_2004_2023.nc']
summary_files = [CPMlib.radar_dir / f for f in summary_files]
for path, resoln, name in zip(summary_files, [5000, 5000, 1000, 2000, 4000, 8000],
                              ['5km hourly', 'coarsened 1km  to 5km hourly', '1km hourly',
                               'coarsened 1km to 2km hourly', 'coarsened 1km to 4km hourly',
                               'coarsened 1km to 8km hourly']):
    out_file = path.name.replace("_summary", "")
    grid = int(resoln / 90.)
    #rain, max_time, topog = get_radar_data(path, topog_grid=grid, region=carmont)
    radar_dataset = comp_event_stats(path, region=carmont, topog_grid=grid,
                                                 height_range=slice(50., None))

