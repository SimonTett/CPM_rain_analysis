"""
Provides classes, variables and functions used across the project
"""
import copy
import re

import fiona
import rioxarray
import iris
import iris.fileformats.nimrod_load_rules
import tarfile
import gzip
import zlib  # so we can trap an error
import tempfile
import shutil

import scipy
import xarray
import pathlib
import cartopy
import cartopy.feature
import cartopy.io
import cartopy.io.shapereader
import pandas as pd
import numpy as np
import platform
import cftime
import cf_units
import logging
import iris.exceptions
import typing
import cartopy.crs as ccrs

logger = logging.getLogger(__name__)

from pandas._libs import NaTType

bad_data_err = (
    zlib.error, ValueError, TypeError, iris.exceptions.ConstraintMismatchError,
    gzip.BadGzipFile)  # possible bad data
# bad_data_err = (zlib.error,iris.exceptions.ConstraintMismatchError,gzip.BadGzipFile) # possible bad data
machine = platform.node()
if ('jasmin.ac.uk' in machine) or ('jc.rl.ac.uk' in machine):
    # specials for Jasmin cluster or LOTUS cluster
    dataDir = pathlib.Path("/gws/nopw/j04/edin_cesd/stett2/Scotland_extremes")
    nimrodRootDir = pathlib.Path("/badc/ukmo-nimrod/data/composite")  # where the nimrod data lives
    cpmDir = pathlib.Path("/badc/ukcp18/data/land-cpm/uk/2.2km/rcp85")  # where the CPM data lives
    outdir = dataDir
    common_data = pathlib.Path('~tetts/data/common_data').expanduser()
elif 'GEOS-' in machine.upper():
    dataDir = pathlib.Path(r'C:\Users\stett2\OneDrive - University of Edinburgh\data\Scotland_extremes')
    nimrodRootDir = dataDir / 'nimrod_data'
    outdir = dataDir
    common_data = pathlib.Path(r'C:\Users\stett2\OneDrive - University of Edinburgh\data\common_data')
elif machine.upper().startswith('CCRC'):  # oz machine at CCRC!
    dataDir = pathlib.Path('~z3542688/data/Scotland_extremes').expanduser()
    nimrodRootDir = dataDir / 'nimrod_data'
    outdir = dataDir
    common_data = pathlib.Path('~z3542688/data/common_data').expanduser()
else:  # don't know what to do so raise an error.
    raise Exception(f"On platform {machine} no idea where data lives")
# create the outdir
outdir.mkdir(parents=True, exist_ok=True)
horizontal_coords = ['projection_x_coordinate', 'projection_y_coordinate']  # for radar data.
cpm_horizontal_coords = ['grid_latitude', 'grid_longitude']  # horizontal coords for CPM data.

try:
    import GSHHS_WDBII

    gshhs = GSHHS_WDBII.GSHHS_WDBII()
    coastline = gshhs.coastlines(scale='full')  # so we can have high resoln coastlines.
except ModuleNotFoundError:
    coastline = cartopy.feature.NaturalEarthFeature('physical', 'coastline', '10m', edgecolor='black', facecolor='none')


def convert_date_to_cftime(
        date: typing.Union[None, int, float, NaTType],
        date_type: typing.Callable = cftime.DatetimeGregorian,
        has_year_zero: bool = False
) -> typing.Union[cftime.datetime, NaTType]:
    date = pd.to_datetime(date)
    if date is None or isinstance(date, NaTType):
        return pd.NaT
    return date_type(date.year, date.month, date.day, date.hour,
                     date.minute, date.second, date.microsecond,
                     has_year_zero=has_year_zero
                     )


## hack iris so bad times work...

def hack_nimrod_time(cube, field):
    """Add a time coord to the cube based on validity time and time-window. HACKED to ignore seconds"""
    NIMROD_DEFAULT = -32767.0

    TIME_UNIT = cf_units.Unit("seconds since 1970-01-01 00:00:00", calendar=cf_units.CALENDAR_GREGORIAN)

    if field.vt_year <= 0:
        # Some ancillary files, eg land sea mask do not
        # have a validity time.
        return
    else:
        missing = field.data_header_int16_25  # missing data indicator
        dt = [field.vt_year, field.vt_month, field.vt_day, field.vt_hour, field.vt_minute,
              field.vt_second]  # set up a list with the dt cpts
        for indx in range(3, len(dt)):  # check out hours/mins/secs and set to 0 if missing
            if dt[indx] == missing:
                dt[indx] = 0

        valid_date = cftime.datetime(*dt)  # make a cftime datetime.

    point = np.around(iris.fileformats.nimrod_load_rules.TIME_UNIT.date2num(valid_date)).astype(np.int64)

    period_seconds = None
    if field.period_minutes == 32767:
        period_seconds = field.period_seconds
    elif (not iris.fileformats.nimrod_load_rules.is_missing(field, field.period_minutes) and field.period_minutes != 0):
        period_seconds = field.period_minutes * 60
    if period_seconds:
        bounds = np.array([point - period_seconds, point], dtype=np.int64)
    else:
        bounds = None

    time_coord = iris.coords.DimCoord(points=point, bounds=bounds, standard_name="time", units=TIME_UNIT)

    cube.add_aux_coord(time_coord)


import iris.fileformats.nimrod_load_rules

iris.fileformats.nimrod_load_rules.time = hack_nimrod_time
print("WARNING MONKEY PATCHING iris.fileformats.nimrod_load_rules.time")


def get_first_non_missing(data_array: xarray.DataArray):
    """
    get the first non-missing value in a data array
    :param data_array: DataArray to be searched
    :return: first non-missing value
    """
    ind = data_array.notnull().argmax(data_array.dims)  # indx to a non-missing value
    return data_array.isel(ind)


def time_convert(DataArray, ref='1970-01-01', unit='h', set_attrs=True):
    """
    convert times to hours (etc) since reference time.
    :param DataAray -- dataArray values to be converted
    :param ref -- reference time as ISO string. Default is 1970-01-01
    :param unit -- unit default is h for hours
    :return -- returns dataarray with units reset and values converted
    """
    name_conversion = dict(m='minutes', h='hours', d='days')

    with xarray.set_options(keep_attrs=True):
        hour = (DataArray - np.datetime64(ref)) / np.timedelta64(1, unit)
    u = name_conversion.get(unit, unit)
    try:
        hour.attrs['units'] = f'{u} since {ref}'
    except AttributeError:  # hour does not have attrs.
        pass
    return hour


def extract_nimrod_day(file, region=None, QCmax=None, gzip_min=85, check_date=False):
    """
    extract rainfall data from nimrod badc archive. 
    Archive is stored as a compressed tarfile of gzipped files.
    Algorithm opens the tarfile. Iterates through files in tarfile 
      Uncompresses each file to tempfile.
       Reads tempfile then deletes it when done.
    returns a dataset of rainfall for the whole day. Note BADC archive
    seems to be missing data so some days will not be complete. 

    :param file -- pathlib path to file for data to be extracted
    :param region (default None) -- if not None then should be a dict of co-ords to be extracted.
    :param QCmax (default None) -- if not None then values > QCmax are set missing as crude QC.
    :param check_date (default False) -- if True check the dates are as expected. Complain if not but keep going
    :param gzip_min (default 85) -- if the gziped file (individual field) is less than this size ignore it -- likely becuase it is zero size.

    :example rain=extract_nimrod_day(path_to_file,
                region = dict(projection_x_coordinate=slice(5e4,5e5),
                projection_y_coordinate=slice(5e5,1.1e6)),QCmax=400.)
    """
    rain = []
    with tarfile.open(file) as tar:
        # iterate over members uncompressing them
        for tmember in tar.getmembers():
            if tmember.size < gzip_min:  # likely some problem with the gzip.
                print(f"{tmember} has size {tmember.size} so skipping")
                continue
            with tar.extractfile(tmember) as fp:
                f_out = tempfile.NamedTemporaryFile(delete=False)
                fname = f_out.name
                try:  # handle bad gzip files etc...
                    with gzip.GzipFile("somefilename", fileobj=fp) as f_in:
                        # uncompress the data writing to the tempfile
                        shutil.copyfileobj(f_in, f_out)  #
                        f_out.close()
                        # doing various transforms to the cube here rather than all at once. 
                        # cubes are quite large so worth doing. 
                        cube = iris.load_cube(fname)
                        da = xarray.DataArray.from_iris(cube)  # read data
                        if region is not None:
                            da = da.sel(**region)  # extract if requested
                        if QCmax is not None:
                            da = da.where(da <= QCmax)  # set missing all values > QCmax
                        # sort out the attributes
                        da = da.assign_attrs(units=cube.units, **cube.attributes, BADCsource='BADC nimrod data')
                        # drop forecast vars (if we have it) -- not sure why they are there!
                        da = da.drop_vars(['forecast_period', 'forecast_reference_time'], errors='ignore')
                        rain.append(da)  # add to the list
                except bad_data_err:
                    print(f"bad data in {tmember}")
                pathlib.Path(
                    fname
                ).unlink()  # remove the temp file.  # end loop over members          (every 15 or 5  mins)
    # end dealing with tarfile -- which will close the tar file.
    if len(rain) == 0:  # no data
        print(f"No data for {file} ")
        return None
    rain = xarray.concat(rain, dim='time')  # merge list of datasets
    rain = rain.sortby('time')
    # make units a string so it can be saved.
    rain.attrs['units'] = str(rain.attrs['units'])
    if check_date:  # check the time...
        date = pathlib.Path(file).name.split("_")[2]
        wanted_date = date[0:4] + "-" + date[4:6] + "-" + date[6:8]
        udate = np.unique(rain.time.dt.date)
        if len(udate) != 1:  # got more than two dates in...
            print(f"Have more than one date -- ", udate)
        lren = len(rain)
        rain = rain.sel(time=wanted_date)  # extract the date
        if len(rain) != lren:
            print("Cleaned data for ", file)
        if len(rain) == 0:
            print(f"Not enough data for {check_date} in {file}")

    return rain


## standard stuff for plots.


# UK local authorities.

# UK nations (and other national sub-units)
nations = cartopy.feature.NaturalEarthFeature(category='cultural', name='admin_0_map_subunits', scale='10m',
                                              facecolor='none', edgecolor='black'
                                              )

# using OS data for regions --generated by create_counties.py
try:
    regions = cartopy.io.shapereader.Reader(common_data / 'GB_OS_boundaries' / 'counties_0010.shp')
    regions = cartopy.feature.ShapelyFeature(regions.geometries(), crs=cartopy.crs.OSGB(), edgecolor='red',
                                             facecolor='none'
                                             )
except fiona.errors.DriverError:
    regions = cartopy.feature.NaturalEarthFeature(category='cultural', name='admin_1_states_provinces_lines',
                                                  scale='10m', edgecolor='red', facecolor='none'
                                                  )


# radar stations

def dms_to_dd(dms_str: str) -> float:
    """
    Convert a string in the format 'degrees°minutes'seconds"direction' to decimal degrees.
    """
    # Check if the input is a string
    if not isinstance(dms_str, str):
        raise ValueError("Input must be a string")

    dms_str = dms_str.strip()  # remove leading/trailing spaces
    # Check if the input string is in the correct format
    if not re.match(r"^\d+°\d+['’]\d+[”\"][NSWE]$", dms_str):
        raise ValueError(
            f"Input string >>{dms_str}<< is not in the correct format 'degrees°minutes'seconds\"direction'"
        )

    degrees, minutes, seconds, direction = re.split('°|\'|’|"|”', dms_str)
    dd = float(degrees) + float(minutes) / 60 + float(seconds) / (60 * 60)
    if direction in ['S', 'W']:
        dd *= -1
    return dd


radar_stations = pd.read_excel(common_data / 'radar_station_metadata.xlsx', index_col=[0], na_values=['-']).T
# convert long/lat from DMS to decimal degrees
for col in ['Latitude', 'Longitude']:
    radar_stations[col] = radar_stations[col].apply(dms_to_dd)
    logger.debug(f"Converted UK Radar data {col} to decimal degrees")
L = radar_stations.Working.str.upper() == 'Y'
radar_stations = radar_stations[L]
# rail network
data_path = common_data / 'UK_railway_DS_10283_2423/UK_Railways.zip'
if not data_path.exists():
    raise FileNotFoundError(f"Can't find {data_path}")
fname = r"zip://" + (data_path / 'Railway.shx').as_posix()
rdr = cartopy.io.shapereader.Reader(fname)
geoms = [r.geometry for r in rdr.records() if 'Standard Gauge' in r.attributes['LEGEND']]
railways = cartopy.feature.ShapelyFeature(geoms, crs=cartopy.crs.OSGB(), linewidth=3, facecolor='none',
                                          edgecolor='purple', linestyle='solid'
                                          )


def std_decorators(
        ax, showregions=True, radarNames=False, radar_col='green', grid: bool = True,
        show_railways: bool = True
):
    """
    Add a bunch of stuff to an axis
    :param ax: axis
    :param showregions: If True show the county borders
    :param radarNames: If True label the radar stations
    :return: Nada
    """

    ax.plot(radar_stations.Easting, radar_stations.Northing, marker='h', color=radar_col, ms=10, linestyle='none',
            transform=cartopy.crs.OSGB(approx=True), clip_on=True
            )  # radar stations location.
    if showregions:
        ax.add_feature(regions, edgecolor='grey', linewidth=2)
    if show_railways:
        ax.add_feature(railways)

    if grid:
        g = ax.gridlines(draw_labels=True)
        g.top_labels = False
        g.left_labels = False

    if radarNames:
        for name, row in radar_stations.iterrows():
            ax.annotate(name, (row.Easting + 500, row.Northing + 500), transform=cartopy.crs.OSGB(approx=True),
                        annotation_clip=True,  # backgroundcolor='grey',alpha=0.5,
                        fontweight='bold', zorder=100
                        )

    ax.add_feature(coastline)  # ax.add_feature(nations, edgecolor='black')


## stuff for event computation
def xarray_gen_cov_samp(
        mean: xarray.DataArray,
        cov: xarray.DataArray,
        rng: typing.Any = 123456,
        nsamp: int = 100
) -> xarray.DataArray:
    """ Generate  samples from the covariance matrix"""

    def gen_cov(mean, cov, rng=123456):
        return scipy.stats.multivariate_normal(mean, cov).rvs(nsamp, random_state=rng).T

    input_core_dims = [['parameter'], ['parameter', 'parameter2']]
    output_core_dims = [['parameter', 'sample', ]]
    samps = xarray.apply_ufunc(gen_cov, mean, cov, kwargs=dict(rng=rng),
                               input_core_dims=input_core_dims,
                               output_core_dims=output_core_dims, vectorize=True
                               )
    samps = samps.assign_coords(sample=np.arange(nsamp))
    return samps


def get_radar_data(
        file: pathlib.Path,
        topog_grid: typing.Optional[int] = None,
        region: typing.Optional[dict] = None,
        height_range: slice = slice(None, None),
        max_total_rain: float = 1000.,
        samples_per_day: int = None,
        sample_fract_limit: typing.Tuple[float, float] = (0., 1.)
) -> (
        xarray.DataArray, xarray.DataArray, xarray.DataArray):
    """
    read in radar data and mask it by heights and mean rain being reasonable.
    :param topog_grid: Grid for topography
    :param file: file to be read in
    :param region: region to extract. Can include spatial and temporal co-ords
    :param height_range: Height range to use (all data strictly *between* these values will be used)
    :param max_total_rain: the maximum total  rain allowed in a season (QC). Values larger than this will be removed.
    :param samples_per_day: number of samples per day. If provided then any month with more than 100% will be removed (and a warning printed out)
    :param sample_fract_limit:
    :return: Data masked for region requested & mxTime, topography
    """

    radar_precip = xarray.open_dataset(file)  #
    if region is not None:
        radar_precip = radar_precip.sel(**region)

    # if samples_per_day is not None then go and remove cases with too many or too few cases.
    if samples_per_day is not None:
        n = radar_precip['No_samples'] / (radar_precip.time.dt.days_in_month * samples_per_day)
        msk = (n < sample_fract_limit[0]) | (n > sample_fract_limit[1])
        if msk.any():
            logger.warning(f"Removing {msk.sum().values} months with too many or too few samples")
            # print out the missing times.
            for t in radar_precip.time[msk]:
                logger.info(f'{t.values}  had {radar_precip["No_samples"].sel(time=t).values} samples so removed.')
            radar_precip = radar_precip.where(~msk, drop=True)

    # Check for size 0 dims raising a warning for each one
    # and then failing if any are size zero.
    has_zero_dim = False
    for dim, size in radar_precip.sizes.items():
        # Check if the current dimension has a size of 0
        if size == 0:
            # Log a warning
            logger.warning(f"Dimension {dim} has size 0")
            has_zero_dim = True

    # If any dimension has size 0, raise an error
    if has_zero_dim:
        raise ValueError("One or more dimensions have size 0")
    rseas = radar_precip.resample(time='QS-Dec').map(time_process)
    rseas = rseas.where(rseas.time.dt.season == 'JJA', drop=True)  # Summer max rain
    topog = read_90m_topog(region=region, resample=topog_grid)  # read in topography and regrid
    top_fit_grid = topog.interp_like(rseas.isel(time=0).squeeze())
    htMsk = True
    if height_range.start:
        htMsk = htMsk & (top_fit_grid > height_range.start)
    if height_range.stop:
        htMsk = htMsk & (top_fit_grid < height_range.stop)

    # mask by seasonal sum < 1000.
    mskRain = ((rseas['Radar_rain_Mean'] * (30 + 31 + 31) / 4.) < max_total_rain) & htMsk
    rseasMskmax = rseas['Radar_rain_Max'].where(mskRain).squeeze(drop=True)
    mxTime = rseas['Radar_rain_MaxTime'].where(mskRain).squeeze(drop=True)

    return rseasMskmax, mxTime, top_fit_grid


def fix_coords(ds: typing.Union[xarray.DataArray, xarray.Dataset]):
    # fix coords in CPM data by rounding to 3 sf. Needed because of slightly different grid for topography and rain...

    for c in cpm_horizontal_coords:
        ds[c] = ds[c].astype(np.float32).round(3)  # round co-ords


def read_90m_topog(region: typing.Optional[dict] = None, resample=None):
    """
    Read 90m DEM data from UoE regridded OSGB data.
    Fix various problems
    :param region: region to select to.
    :param resample: If not None then the amount to coarsen by.
    :return: topography dataset
    """
    topog = rioxarray.open_rasterio(common_data / 'uk_srtm')
    topog = topog.reindex(y=topog.y[::-1]).rename(x='projection_x_coordinate', y='projection_y_coordinate')
    if region is not None:
        rgn = region.copy()
        rgn.pop('time', None)
        topog = topog.sel(**rgn)
    topog = topog.load().squeeze()
    L = (topog > -10000) & (topog < 10000)  # fix bad data. L is where data is good!
    topog = topog.where(L)
    if resample is not None:
        topog = topog.coarsen(projection_x_coordinate=resample, projection_y_coordinate=resample, boundary='pad').mean()
    return topog


from typing import Optional, Union, Dict, List, Tuple, Callable, Literal


def apply_func_recursively(
        collection: Union[Dict, List, any],
        func: Callable
) -> (
        Union)[Dict, List, any]:
    if isinstance(collection, dict):
        return {k: apply_func_recursively(v, func) for k, v in collection.items()}
    elif isinstance(collection, list):
        return [apply_func_recursively(v, func) for v in collection]
    else:
        return func(collection)


def convert(obj):
    if isinstance(obj, slice):
        return convert_slice(obj)
    elif obj is None:
        return np.nan
    else:
        return obj


def convert_slice(obj: slice) -> np.ndarray:
    # convert a slice to start and end
    result = [obj.start, obj.stop]
    if obj.step is not None:
        result.append(obj.step)
    result = [np.nan if r is None else r for r in result]

    return result


# noinspection PyShadowingNames
def comp_event_stats(file: pathlib.Path
                     ,
                     source: typing.Literal['RADAR', 'CPM'] = 'RADAR',
                     **kwargs
                     ):
    """
    read in  data and then group by day.
    :param file: File to be read in
    :param source: Source of data -- RADAR or CPM
        remaining arguments passed through to get_radar_data
    :return: grouped dataset
    """
    rseasMskmax, mxTime, topog = get_radar_data(file, **kwargs)
    # loop over rolling co-ords
    radar_data = []
    for r in mxTime['rolling']:
        msk_max = rseasMskmax.sel(rolling=r)
        max_time = mxTime.sel(rolling=r)
        grp = ((max_time.dt.dayofyear - 1) + (max_time.dt.year - 1980) * 1000).rename('EventTime')
        grp = grp.where(~msk_max.isnull(), 0).rename('EventTime')
        dataset = event_stats(msk_max, max_time, grp, source=source).sel(EventTime=slice(1, None))
        ht = topog.sel(projection_x_coordinate=dataset.x, projection_y_coordinate=dataset.y)
        # drop unneeded coords
        coords_to_drop = [c for c in ht.coords if c not in ht.dims]
        ht = ht.drop_vars(coords_to_drop)
        dataset['height'] = ht
        dataset['Event_Size'] = dataset.EventTime.size
        # add in rolling and modify event time.
        radar_data.append(dataset.assign_coords(dict(rolling=[r], EventTime=np.arange(1, len(dataset.EventTime) + 1))))

    radar_dataset = xarray.concat(radar_data, 'rolling').assign_attrs(rseasMskmax.attrs)
    # get rid of coords that are not in dims,

    radar_dataset = radar_dataset.drop_vars(set(radar_dataset.coords) - set(radar_dataset.dims))
    # add to the attributes the kwargs
    args_fix = copy.deepcopy(kwargs)
    args_fix.pop('region')  # have the co-ord info and converting it would need more work!
    args_fix = apply_func_recursively(args_fix, convert)
    radar_dataset = radar_dataset.assign_attrs(args_fix
                                               )  # update the attributes with the various kwargs used to get radar data.
    return radar_dataset

def source_coords(source: str) -> tuple[str, ...]:
    """
    Return coords for different source
    :param source:  Source CPM or RADAR
    :return: co-ord names s a tuple.
    :rtype:
    """
    if source == 'CPM':
        return tuple(cpm_horizontal_coords)
    elif source == 'RADAR':
        return tuple(horizontal_coords)
    else:
        raise ValueError(f"Unknown source {source}")

def event_stats(max_precip: xarray.DataArray, max_time: xarray.DataArray, group, source: str = "CPM"):
    x_coord, y_coord = source_coords(source)
    ds = xarray.Dataset(dict(maxV=max_precip, maxT=max_time))
    grper = ds.groupby(group)
    quantiles = np.linspace(0, 1, 21)
    dataSet = grper.map(quants_locn, quantiles=quantiles, x_coord=x_coord, y_coord=y_coord).rename(quant='max_precip')
    count = grper.count().maxV.rename("# Cells")
    dataSet['count_cells'] = count
    return dataSet


def time_process(data_set: xarray.Dataset,
                 mean_var: str = 'Radar_rain_Mean',
                 max_var: str = 'Radar_rain_Max',
                 max_time_var: str = 'Radar_rain_MaxTime',
                 samples_var: str = 'No_samples',
                 time_dim: str = 'time') -> xarray.Dataset:
    """
    Process a dataset of radar data (though could be anything by appropriate setting of the variable names
    :param data_set -- Dataset to process
    :param mean_var -- name of mean variable
    :param max_var -- name of max variable
    :param max_time_var - name of time of max variable
    :param samples_var -- name of samples var (if not found a warnign will be printed)
    returns a dataset containing mean of means, max of maxes, time of max, time_bounds and total number of samples.
    Used to process the data for a season.

    """
    mn = data_set[mean_var]
    mx = data_set[max_var].max(time_dim, keep_attrs=True)  # max of maxes
    mx_idx = data_set[max_var].idxmax(time_dim, skipna=True)  # index  of max
    mx_time = data_set[max_time_var].sel({time_dim: mx_idx}, method='ffill').drop_vars('time')
    time_bounds = xarray.DataArray([mn[time_dim].min().values, mn[time_dim].max().values],
                                   coords=dict(bounds=[0, 1])).rename(
        f'{time_dim}_bounds'
    )
    mn = mn.mean(time_dim, keep_attrs=True)
    ds = xarray.merge([mn, mx, mx_time, time_bounds])
    try:
        no_samples = data_set[samples_var].sum(time_dim, keep_attrs=True)
        ds = xarray.merge([ds, no_samples])
    except KeyError:
        logger.warning(f'Failed to find {samples_var} in data_set')

    return ds


def comp_events(max_values: xarray.DataArray,
                max_times: xarray.DataArray,
                grp: xarray.DataArray,
                topog: xarray.DataArray,
                cet: xarray.DataArray,
                source: str = 'CPM'
                ):
    dd_lst = []
    expected_event_area = len(max_values.time) * (topog > 0).sum()
    for roll in max_values['rolling'].values:
        dd = event_stats(max_values.sel(rolling=roll),
                         max_times.sel(rolling=roll),
                         grp.sel(rolling=roll),source=source
                         ).sel(EventTime=slice(1, None))
        # at this point we have the events. Check that the total cell_count. Should be no_years*no non-nan cells in seasonalMax
        assert (int(dd.count_cells.sum('EventTime').values) == expected_event_area)
        event_time_values = np.arange(0, len(dd.EventTime))
        dd = dd.assign_coords(rolling=roll, EventTime=event_time_values)
        # get the summer mean CET out and force its time's to be the same.
        tc = np.array([f"{int(y)}-06-01" for y in dd.t.isel(quantv=0).dt.year])
        cet_extreme_times = cet.interp(time=tc).rename(dict(time='EventTime'))
        # convert EventTime to an index.
        cet_extreme_times = cet_extreme_times.assign_coords(rolling=roll, EventTime=event_time_values)
        dd['CET'] = cet_extreme_times  # add in the CET data.

        # add in hts
        coords = source_coords(source)
        sel = dict(zip(coords, [dd.x, dd.y]))
        ht = topog.sel(**sel)
        # drop unneeded coords
        coords_to_drop = [c for c in ht.coords if c not in ht.dims]
        ht = ht.drop_vars(coords_to_drop)
        dd['height'] = ht
        dd_lst.append(dd)
        logger.info(f"Processed rolling: {roll}")
    event_ds = xarray.concat(dd_lst, dim='rolling')
    return event_ds


def quants_locn(
        data_set: xarray.Dataset, dimension: typing.Optional[typing.Union[str, typing.List[str]]] = None,
        quantiles: typing.Optional[typing.Union[np.ndarray, typing.List[float]]] = None,
        x_coord: str = 'grid_longitude', y_coord: str = 'grid_latitude'
) -> xarray.Dataset:
    """ compute quantiles and locations"""
    if quantiles is None:
        quantiles = np.linspace(0.0, 1.0, 6)

    data_array = data_set.maxV  # maximum values
    time_array = data_set.maxT  # time when max occurs
    quant = data_array.quantile(quantiles, dim=dimension).rename(quantile="quantv")
    order_values = data_array.values.argsort()
    oindices = ((data_array.size - 1) * quantiles).astype('int64')
    indices = order_values[oindices]  # actual indices to the data where the quantiles are roughly
    indices = xarray.DataArray(indices, coords=dict(quantv=quantiles))  # and give them co-ords
    y = data_array[y_coord].broadcast_like(data_array)[indices]
    x = data_array[x_coord].broadcast_like(data_array)[indices]
    time = time_array[indices]
    result = xarray.Dataset(dict(x=x, y=y, t=time, quant=quant))
    # drop unneeded coords
    coords_to_drop = [c for c in result.coords if c not in result.dims]
    result = result.drop_vars(coords_to_drop)
    return result


def comp_params(
        param: xarray.DataArray,
        temperature: float = 0.0,
        log10_area: typing.Optional[float] = None,
        hour: typing.Optional[float] = None,
        height: typing.Optional[float] = None
):
    """
    Compute the location and scale parameters for a given set of covariate values
    :param param:
    :type param:
    :param temperature:
    :type temperature:
    :param log10_area:
    :type log10_area:
    :param hour:
    :type hour:
    :param height:
    :type height:
    :return:
    :rtype:
    """
    if log10_area is None:
        log10_area = np.log10(150.)
    if hour is None:
        hour = 13.0
    if height is None:
        height = 100.
    result = [param.sel(parameter='shape')]  # start with shape
    param_names = dict(log10_area=log10_area, hour=hour, hour_sqr=hour ** 2, height=height,
                       CET=temperature, CET_sqr=temperature ** 2
                       )
    for k in ['location', 'scale']:
        r = param.sel(parameter=k)
        for pname, v in param_names.items():
            p = f"D{k}_{pname}"
            try:
                r = r + v * param.sel(parameter=p)
            except KeyError:
                logger.debug(f"{k} missing {p} so ignoring")
        r = r.assign_coords(parameter=k)  #  rename
        result.append(r)
    result = xarray.concat(result, 'parameter')
    return result


def fix_cpm_topog(topog: xarray.DataArray, reference: xarray.DataArray) -> xarray.DataArray:
    """
    Fix the CPM topography data which has small errors in its co-ords relative to jasmin archive data
    :param topog: Topography data
    :param topog_grid: Grid to regrid to
    :return: Fixed topography data
    """

    max_rel_lon = float((np.abs(topog.grid_longitude.values / reference.grid_longitude.values - 1)).max())
    max_rel_lat = float((np.abs(topog.grid_latitude.values / reference.grid_latitude.values - 1)).max())

    if max_rel_lon > 1e-6:
        raise ValueError(f"Lon Grids differ by more than 1e-6. Max = {max_rel_lon}")

    if max_rel_lat > 1e-6:
        raise ValueError(f"Lat Grids differ by more than 1e-6. Max = {max_rel_lat}")
    topog = topog.interp(grid_longitude=reference.grid_longitude, grid_latitude=reference.grid_latitude
                         )  # grids differ by tiny amount.

    return topog


def radar_dx_dy(radar: xarray.Dataset) -> np.ndarray:
    """
    Compute the mean dx, dy for a radar dataset
    :param radar: radar dataset -- only uses the coords
    :return: 2 element numpy array . element 0 = mean dx,element 1 = mean dy
    """
    dx = radar.projection_x_coordinate.diff('projection_x_coordinate')
    dy = radar.projection_y_coordinate.diff('projection_y_coordinate')
    return np.array([float(dx.mean()), float(dy.mean())])
