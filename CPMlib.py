# library for cpm ANALYSIS
import numpy as np
import xarray
import cftime
import typing
import pathlib
import functools
import cartopy.crs as ccrs

datadir=pathlib.Path(r"C:\Users\stett2\OneDrive - University of Edinburgh\data\Scotland_extremes")
CPM_dir=datadir/"CPM_scotland"
fit_dir = datadir/'fits'
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

def quants_locn(data_set:xarray.Dataset,
                dimension:typing.Optional[typing.Union[str,typing.List[str]]]=None,
                quantiles:typing.Optional[typing.Union[np.ndarray,typing.List[float]]]=None,
                x_coord:str = 'grid_longitude',
                y_coord:str = 'grid_latitude') -> xarray.Dataset:
    """ compute quantiles and locations"""
    if quantiles is None:
        quantiles=np.linspace(0.0,1.0,6)

    data_array = data_set.maxV # maximum values
    time_array = data_set.maxT # time when max occurs
    quant = data_array.quantile(quantiles,dim=dimension).rename(quantile="quantv")
    order_values = data_array.values.argsort()
    oindices = ((data_array.size-1)*quantiles).astype('int64')
    indices = order_values[oindices] # actual indices to the data where the quantiles are roughly
    indices = xarray.DataArray(indices, coords=dict(quantv=quantiles)) # and give them co-ords
    y = data_array[y_coord].broadcast_like(data_array)[indices]
    x = data_array[x_coord].broadcast_like(data_array)[indices]
    time = time_array[indices]
    result = xarray.Dataset(dict(x=x,y=y,t=time,quant=quant))
    # drop unneeded coords
    coords_to_drop = [c for c in result.coords if c not in result.dims]
    result = result.drop_vars(coords_to_drop)
    return result

def event_stats(max_precip, max_time,group, dim,source=None):

    if source is None:
        source = 'CPM'
    if source == 'CPM':
        x_coord='grid_longitude'
        y_coord = 'grid_latitude'
    elif source == 'radar':
        x_coord='projection_x_coordinate'
        y_coord = 'projection_y_coordinate'
    else:
        raise ValueError(f"Unknown source {source}")

    ds=xarray.Dataset(dict(maxV=max_precip,maxT=max_time))
    grper= ds.groupby(group)
    quantiles = np.linspace(0,1,21)
    dataSet = grper.map(quants_locn,quantiles=quantiles,x_coord=x_coord,y_coord=y_coord).rename(quant='max_precip')
    grper2 = max_precip.groupby(group)
    count = grper2.count().rename("# Cells")
    dataSet['count_cells']=count
    return dataSet


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