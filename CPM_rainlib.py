"""
Provides classes, variables and functions used across the project
"""

import fiona

import iris
import tarfile
import gzip
import zlib  # so we can trap an error
import tempfile
import shutil
import xarray
import pathlib
import cartopy
import cartopy.feature
import pandas as pd
import numpy as np
import platform
import cftime
import cf_units
import matplotlib.pyplot as plt
import iris.exceptions

bad_data_err = (
zlib.error, ValueError, TypeError, iris.exceptions.ConstraintMismatchError, gzip.BadGzipFile)  # possible bad data
# bad_data_err = (zlib.error,iris.exceptions.ConstraintMismatchError,gzip.BadGzipFile) # possible bad data
machine = platform.node()
if ('jasmin.ac.uk' in machine) or ('jc.rl.ac.uk' in machine):
    # specials for Jasmin cluster or LOTUS cluster
    dataDir = pathlib.Path.cwd()
    nimrodRootDir = pathlib.Path("/badc/ukmo-nimrod/data/composite")  # where the nimrod data lives
    cpmDir = pathlib.Path("/badc/ukcp18/data/land-cpm/uk/2.2km/rcp85")  # where the CPM data lives
    outdir = pathlib.Path("/gws/nopw/j04/edin_cesd/stett2/")  # writing locally. Really need a workspace..
elif 'GEOS-' in machine.upper():
    dataDir = pathlib.Path(r'C:\Users\stett2\OneDrive - University of Edinburgh\data\EdinburghRainfall')
    nimrodRootDir = dataDir / 'nimrod_data'
    outdir = dataDir / 'output_data'
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


## hack iris so bad times work...

def hack_nimrod_time(cube, field):
    """Add a time coord to the cube based on validity time and time-window. HACKED to ignore seconds"""
    NIMROD_DEFAULT = -32767.0

    TIME_UNIT = cf_units.Unit(
        "seconds since 1970-01-01 00:00:00", calendar=cf_units.CALENDAR_GREGORIAN
    )

    if field.vt_year <= 0:
        # Some ancillary files, eg land sea mask do not
        # have a validity time.
        return
    else:
        missing = field.data_header_int16_25  # missing data indicator
        dt = [
            field.vt_year,
            field.vt_month,
            field.vt_day,
            field.vt_hour,
            field.vt_minute,
            field.vt_second]  # set up a list with the dt cpts
        for indx in range(3, len(dt)):  # check out hours/mins/secs and set to 0 if missing
            if dt[indx] == missing:
                dt[indx] = 0

        valid_date = cftime.datetime(*dt)  # make a cftime datetime.

    point = np.around(iris.fileformats.nimrod_load_rules.TIME_UNIT.date2num(valid_date)).astype(np.int64)

    period_seconds = None
    if field.period_minutes == 32767:
        period_seconds = field.period_seconds
    elif (
            not iris.fileformats.nimrod_load_rules.is_missing(field, field.period_minutes)
            and field.period_minutes != 0
    ):
        period_seconds = field.period_minutes * 60
    if period_seconds:
        bounds = np.array([point - period_seconds, point], dtype=np.int64)
    else:
        bounds = None

    time_coord = iris.coords.DimCoord(
        points=point, bounds=bounds, standard_name="time", units=TIME_UNIT
    )

    cube.add_aux_coord(time_coord)


import iris.fileformats

iris.fileformats.nimrod_load_rules.time = hack_nimrod_time
print("WARNING MONKEY PATCHING iris.fileformats.nimrod_load_rules.time")


def time_process(DS, varPrefix='daily', summary_prefix=''):
    """
    Process a dataset of (daily) data
    :param DS -- Dataset to process
    :param outFile (default None). Name of file for summary data to output. If None  nothing will be written.
            All times in attributes or "payload" will be converted used commonLib.convert_time
    :param varPrefix (default 'daily') -- variable prefix on DataArrays in datasets)
    :param summary_prefix (default '') -- prefix to be added to output maxes etc
    """
    mx = DS[varPrefix + 'Max'].max('time', keep_attrs=True).rename(f'{summary_prefix}Max')  # max of maxes
    mx_idx = DS[varPrefix + 'Max'].fillna(0.0).argmax('time', skipna=True)  # index  of max
    mx_time = DS[varPrefix + 'MaxTime'].isel(time=mx_idx).drop_vars('time').rename(f'{summary_prefix}MaxTime')
    time_max = DS[varPrefix + 'Max'].time.max().values
    mn = DS[varPrefix + 'Mean'].mean('time', keep_attrs=True).rename(f'{summary_prefix}Mean')
    # actual time. -- dropping time as has nothing and will fix later

    ds = xarray.merge([mn, mx, mx_time])

    ds.attrs['max_time'] = time_max

    return ds


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
                pathlib.Path(fname).unlink()  # remove the temp file.
            # end loop over members          (every 15 or 5  mins)
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


def time_process(DS, varPrefix='daily', summary_prefix=''):
    f"""
    Process a dataset of (daily) data
    :param DS -- Dataset to process. Should contain variables called:
      f"{varPrefix}Max", f"{varPrefix}MaxTime", f"{varPrefix}Mean" 
    :param varPrefix (default 'daily') -- variable prefix on DataArrays in datasets)
    :param summary_prefix (default '') -- prefix to be added to output maxes etc
    """
    mx = DS[varPrefix + 'Max'].max('time', keep_attrs=True).rename(f'{summary_prefix}Max')  # max of maxes
    mx_idx = DS[varPrefix + 'Max'].fillna(0.0).argmax('time', skipna=True)  # index  of max
    mx_time = DS[varPrefix + 'MaxTime'].isel(time=mx_idx).drop_vars('time').rename(f'{summary_prefix}MaxTime')
    time_max = DS[varPrefix + 'Max'].time.max().values
    mn = DS[varPrefix + 'Mean'].mean('time', keep_attrs=True).rename(f'{summary_prefix}Mean')
    # actual time. -- dropping time as has nothing and will fix later

    ds = xarray.merge([mn, mx, mx_time])

    ds.attrs['max_time'] = time_max

    return ds


## standard stuff for plots.


# UK local authorities.

# UK nations (and other national sub-units)
nations = cartopy.feature.NaturalEarthFeature(
    category='cultural', name='admin_0_map_subunits',
    scale='10m', facecolor='none', edgecolor='black')

# using OS data for regions --generated by create_counties.py
try:
    regions = cartopy.io.shapereader.Reader(dataDir / 'GB_OS_boundaries' / 'counties_0010.shp')
    regions = cartopy.feature.ShapelyFeature(regions.geometries(), crs=cartopy.crs.OSGB(), edgecolor='red',
                                             facecolor='none')
except fiona.errors.DriverError:
    regions = cartopy.feature.NaturalEarthFeature(
        category='cultural',
        name='admin_1_states_provinces_lines',
        scale='10m',
        edgecolor='red',
        facecolor='none')

# radar stations
metadata = pd.read_excel('radar_station_metadata.xlsx', index_col=[0], na_values=['-']).T
L = metadata.Working.str.upper() == 'Y'
metadata = metadata[L]


def std_decorators(ax, showregions=True, radarNames=False):
    """
    Add a bunch of stuff to an axis
    :param ax: axis
    :param showregions: If True show the county borders
    :param radarNames: If True label the radar stations
    :return: Nada
    """

    ax.plot(metadata.Easting, metadata.Northing, marker='h', color='orange', ms=10, linestyle='none',
            transform=cartopy.crs.OSGB(approx=True), clip_on=True)  # radar stations location.
    # ax.gridlines(draw_labels=False, x_inline=False, y_inline=False)
    if showregions:
        ax.add_feature(regions, edgecolor='red')
    if radarNames:
        for name, row in metadata.iterrows():
            ax.annotate(name, (row.Easting + 500, row.Northing + 500), transform=cartopy.crs.OSGB(approx=True),
                        annotation_clip=True)

    ax.add_feature(coastline)
    # ax.add_feature(nations, edgecolor='black')
