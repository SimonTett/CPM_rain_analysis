# library for cpm ANALYSIS
import numpy as np
import xarray
import cftime
import typing
import pathlib
import functools
import cartopy.crs as ccrs
import CPM_rainlib
import scipy.stats

CPM_coords = ['grid_longitude', 'grid_latitude']

CPM_dir = CPM_rainlib.dataDir / "CPM_scotland"  # processed CPM data
CPM_filt_dir = CPM_rainlib.dataDir / "CPM_scotland_filter"  # processed CPM data
table_dir = pathlib.Path('paper/tables')
radar_dir = CPM_rainlib.dataDir / "radar"  # radar data
fit_dir = CPM_rainlib.outdir / 'fits'  # where the fits go
# make all the directories
for direct in [CPM_dir, CPM_filt_dir, table_dir, radar_dir, fit_dir]:
    direct.mkdir(exist_ok=True, parents=True)
fit_dir.mkdir(exist_ok=True, parents=True)
time_unit = 'hours since 1980-01-01'
projRot = ccrs.RotatedPole(pole_longitude=177.5, pole_latitude=37.5)
projOSGB = ccrs.OSGB()
ll = ccrs.PlateCarree()
cc_scale = 5.7  # computed in plot_scatter
cc_uncert = 0.04
cc_dist = scipy.stats.norm(loc=cc_scale, scale=cc_uncert)
# colours for different radars.
radar_cols = {'1km': 'black', '5km': 'green', '1km_c4': 'blue', '1km_c5': 'cornflowerblue'}
# compute various carmont co-ords
carmont_long_lat = (-2.3197, 56.95248)  # location of derailment from openstreetmap.
carmont_drain_long_lat = (-2.3266852553075017, 56.951548724582096)  # "field" where rainfell
carmont = dict(zip(CPM_coords, projRot.transform_point(*carmont_long_lat, ll)))
carmont[CPM_coords[0]] += 360.

carmont_OSGB = dict(zip(["projection_x_coordinate", "projection_y_coordinate"],
                        projOSGB.transform_point(*carmont_long_lat, ll)
                        )
                    )
carmont_drain_OSGB = dict(zip(["projection_x_coordinate", "projection_y_coordinate"],
                              projOSGB.transform_point(*carmont_drain_long_lat, ll)
                              )
                          )
carmont_drain = dict(zip(CPM_coords,
                         projRot.transform_point(*carmont_drain_long_lat, ll)
                         )
                     )
carmont_drain[CPM_coords[0]] += 360.
carmont_rgn_OSGB = {k: slice(v - 75e3, v + 75e3) for k, v in carmont_drain_OSGB.items()}
carmont_rgn = {k: slice(v - 0.75, v + 0.75) for k, v in carmont_drain.items()}
carmont_rgn_extent = []
for v in carmont_rgn.values():
    carmont_rgn_extent.extend([v.start, v.stop])
kw_colorbar = dict(orientation='horizontal', fraction=0.1, aspect=40, pad=0.05, extend='both')

today_sel = dict(time=slice('2008', '2023'))  # so "today" is common throughout!
PI_sel = dict(time=slice('1851', '1900'))  # so "PI" is common throughout!


def discretise(time: xarray.DataArray) -> xarray.DataArray:
    """

    param time: time of max
    :type time:
    :return:
    :rtype:
    """

    def disc(time, time_unit):
        times = cftime.date2num(time, time_unit).astype('int')
        result = 24 * (times // 24).astype('int')  # +((times%24)/24)*24
        return result.astype('int')

    return xarray.apply_ufunc(disc, time, time_unit)  # do the discretisation and return the discrete times


def inv_disc(disc_time: xarray.DataArray) -> xarray.DataArray:
    """
    Invert
    :param half_day:
    :type half_day:
    :return:
    :rtype:
    """

    result = cftime.num2date(disc_time, units=time_unit, calendar='360day')
    return result


def time_convert(DataArray, ref_yr=1970, set_attrs=True, calendar='360_day'):
    """
    convert times to hours (etc) since 1st Jan of year
    :param DataAray -- dataArray values to be converted
    :param ref -- reference year as int. Default is 1970
    :return -- returns dataarray with units reset and values converted
    """
    # name_conversion=dict(m='minutes',h='hours',d='days')

    with xarray.set_options(keep_attrs=True):
        # Need a "null" mask
        OK = ~DataArray.isnull()
        nullT = cftime.num2date(0, units='hours since 1-1-1', calendar=calendar)  # what we use for nan
        time = DataArray.where(OK, nullT)
        hour = xarray.apply_ufunc(cftime.date2num, time,
                                  kwargs=dict(units=f'hours since {ref_yr:d}-01-01', calendar=calendar)
                                  )
    # u = name_conversion.get(unit,unit)
    try:
        hour.attrs['units'] = f'hours since {ref_yr}-01-01'
        hour.attrs['calendar'] = '360_day'  # should get that from data
        hour.attrs['note'] = 'Missing data encoded as 1-1-1'
    except AttributeError:  # hour does not have attrs.
        pass
    return hour


def location_scale(
        fit_parameters: xarray.DataArray,
        covariates: typing.Optional[typing.List[xarray.DataArray]] = None,
        scaled: bool = False
        ) -> xarray.Dataset:
    """
    Compute the location and scale parameters
    :param fit_parameters: The parameters of the distribution fit. location and scale are used
    :param covariates: A list of covariates used in the fit and for predicting the values
    :param scaled: If True compued the scaled Dlocation/Dscale parameters relative to the location and scale params computed.
    :return:A datase containing the location, scale and optionally normalised Dlocation_COV/DScale_COV values.
    """

    if covariates is None:
        covariates = []
    loc = fit_parameters.sel(parameter='location').drop_vars('parameter')
    scale = fit_parameters.sel(parameter='scale').drop_vars('parameter')
    for cov in covariates:
        pl = "Dlocation_" + str(cov.name)
        ps = "Dscale_" + str(cov.name)
        loc = loc + cov * fit_parameters.sel(parameter=pl).drop_vars('parameter')
        scale = scale + cov * fit_parameters.sel(parameter=ps).drop_vars('parameter')
    result = xarray.Dataset(dict(location=loc, scale=scale))
    if scaled and (len(covariates) > 0):  # compute scaled Dloc & Dscale values

        for cov in covariates:  # work out scalings
            key = str(cov.name)
            result["Dlocation_" + key] = fit_parameters.sel(parameter='Dlocation_' + key).drop_vars('parameter') / loc
            result["Dscale_" + key] = fit_parameters.sel(parameter='Dscale_' + key).drop_vars('parameter') / scale

    return result


def get_topog(
        rgn_all: dict,
        rgn: dict,
        lat: xarray.Variable,
        lon: xarray.Variable
        ) -> (xarray.DataArray, xarray.DataArray):
    """

    :param rgn_all: Large scale region to extract data too (for coarsening)
    :param rgn: regn to extract too after coarsening
    :param lat: lat co-ords to regrid too.
    :param lon: lon co-ords to regrid too.
    :return: coarsened min and mean values
    """
    topog = xarray.load_dataset(CPM_dir / 'orog_land-cpm_BI_2.2km.nc', decode_times=False)
    topog = topog.ht.sel(rgn_all).rename(dict(longitude='grid_longitude', latitude='grid_latitude')).squeeze()
    t = topog.coarsen(grid_longitude=2, grid_latitude=2, boundary='trim').min().sel(rgn)
    tmn = topog.coarsen(grid_longitude=2, grid_latitude=2, boundary='trim').mean().sel(rgn)
    # check rel errors < 1.e-6
    max_rel_lon = float((np.abs(t.grid_longitude.values / lon.values - 1)).max())
    max_rel_lat = float((np.abs(t.grid_latitude.values / lat.values - 1)).max())

    if max_rel_lon > 1e-6:
        raise ValueError(f"Lon Grids differ by more than 1e-6. Max = {max_rel_lon}")

    if max_rel_lat > 1e-6:
        raise ValueError(f"Lat Grids differ by more than 1e-6. Max = {max_rel_lat}")
    t = t.interp(grid_longitude=lon, grid_latitude=lat)  # grids differ by tiny amount.
    tmn = tmn.interp(grid_longitude=lon, grid_latitude=lat)

    return t, tmn


def plot_carmont(ax: "cartopy.mpl.geoaxes.GeoAxes", **kwargs):
    """
    Plot the carmont drain location
    :param ax: Axis on which to plot
    : all other keywords passed through to plot command
    :return:
    """
    kwargs = kwargs.copy()
    default_values = dict(transform=ccrs.PlateCarree(),
                          marker='o', ms=8,
                          mec='cornflowerblue',
                          alpha=0.8,
                          markerfacecolor='none',
                          markeredgewidth=2,
                          zorder=80) # default values
    for key, value in default_values.items():
        kwargs[key] = kwargs.get(key, value)

    ax.plot(*carmont_drain_long_lat, **kwargs)
