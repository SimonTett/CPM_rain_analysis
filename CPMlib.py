# library for cpm ANALYSIS
import numpy as np
import xarray
import cftime
import typing
import pathlib
import functools
import cartopy.crs as ccrs
import CPM_rainlib

#datadir=pathlib.Path(r"C:\Users\stett2\OneDrive - University of Edinburgh\data\Scotland_extremes")
CPM_dir=CPM_rainlib.dataDir/"CPM_scotland" # processed CPM data
fit_dir = CPM_rainlib.outdir/'fits'  # where the fits go
fit_dir.mkdir(exist_ok=True,parents=True)
time_unit='hours since 1980-01-01'
projRot = ccrs.RotatedPole(pole_longitude=177.5,pole_latitude=37.5)
projOSGB = ccrs.OSGB()
stonehaven=dict(grid_longitude=360.16, grid_latitude=4.46) # co-ords on rotated grid
stonehaven_rgn={k:slice(v-0.75,v+0.75) for k,v in stonehaven.items()}
carmont_long_lat=(-2.321111,56.952500)
ll=ccrs.PlateCarree()
carmont=dict(zip(['grid_longitude','grid_latitude'],projRot.transform_point(*carmont_long_lat,ll)))
carmont['grid_longitude'] += 360.

carmont_OSGB = dict(zip(["projection_x_coordinate","projection_y_coordinate"],
                        projOSGB.transform_point(*carmont_long_lat,ll)))
def discretise(time:xarray.DataArray) -> xarray.DataArray:
    """

    param time: time of max
    :type time:
    :return:
    :rtype:
    """
    def disc(time,time_unit):
        times= cftime.date2num(time,time_unit).astype('int')
        result = 24*(times//24).astype('int')#+((times%24)/24)*24
        return result.astype('int')

    return xarray.apply_ufunc(disc,time,time_unit) # do the discretisation and return the discrete times

def inv_disc(disc_time:xarray.DataArray) -> xarray.DataArray:
    """
    Invert
    :param half_day:
    :type half_day:
    :return:
    :rtype:
    """

    result =cftime.num2date(disc_time,units=time_unit,calendar='360day')
    return result





def time_convert(DataArray, ref_yr=1970, set_attrs=True,calendar='360_day'):
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
        hour = xarray.apply_ufunc(cftime.date2num,time,kwargs=dict(units=f'hours since {ref_yr:d}-01-01',calendar=calendar))
    # u = name_conversion.get(unit,unit)
    try:
        hour.attrs['units'] = f'hours since {ref_yr}-01-01'
        hour.attrs['calendar'] = '360_day'  # should get that from data
        hour.attrs['note'] = 'Missing data encoded as 1-1-1'
    except AttributeError:  # hour does not have attrs.
        pass
    return hour

def location_scale(fit_parameters:xarray.DataArray,
                covariates:typing.Optional[typing.List[xarray.DataArray]]=None,
               scaled:bool=False) -> xarray.Dataset:
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
        pl="Dlocation_"+str(cov.name)
        ps="Dscale_"+str(cov.name)
        loc = loc + cov*fit_parameters.sel(parameter=pl).drop_vars('parameter')
        scale = scale + cov*fit_parameters.sel(parameter=ps).drop_vars('parameter')
    result=xarray.Dataset(dict(location=loc,scale=scale))
    if scaled and (len(covariates) > 0): # compute scaled Dloc & Dscale values

        for cov in covariates:  # work out scalings
            key = str(cov.name)
            result["Dlocation_"+key] = fit_parameters.sel(parameter='Dlocation_' + key).drop_vars('parameter') / loc
            result["Dscale_"+key] = fit_parameters.sel(parameter='Dscale_' + key).drop_vars('parameter') / scale

    return result
