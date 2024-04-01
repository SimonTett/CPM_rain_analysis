# test that gev_r.xarray_gev works with weights
import CPMlib
import commonLib
from R_python import gev_r
import xarray
import typing
import numpy as np
import pandas as pd
import rpy2.robjects as robjects
import rpy2.robjects.packages as rpackages
import rpy2.robjects.pandas2ri as rpandas2ri
import rpy2.rinterface_lib
from rpy2.robjects import numpy2ri
import cftime


base = rpackages.importr('base')  # to give us summary
def gev_fit(*args: typing.Tuple[np.ndarray], # being calle like gev_fit(arg1,arg2) which should get slurped up
            use_weights:bool = False,
            shapeCov: bool = False):
    """
    Do GEV fit using R and return named tuple of relevant values.
    :param x: Data to be fit
    :param cov: co-variate value (if None then not used)
    :param returnType. Type to return, allowed is:
        named -- return a named tuple (default)
        tuple -- return a tuple -- useful for apply_ufunc
        DataSet -- return a DataSet
    :param shapeCov -- If True allow the shape to vary with the co-variate.
    :return: A dataset of the parameters of the fit.
    #TODO rewrite this to directly call the R fn.
    """
    x = args[0]
    if use_weights:
        wts= args[-1]
        args = args[0:-1]

    ncov = len(args) - 1
    L = ~np.isnan(x)
    df_data = [x[L]]  # remove nan from data]
    wts=wts[L] # remove nan from wts.
    cols = ['x']
    npts = 3 + 2 * ncov
    if shapeCov:
        npts += ncov
        # end of dealing with covariance and trying to get to the right shape.
    r_code = 'fevd(x=x,data=df'
    for indx, cov in enumerate(args[1:]):
        df_data.append(cov[L])  # remove places where x was nan from cov.
        cols.append(f"cov{indx:d}")
    if len(cols) > 1:
        cov_expr = "~" + " + ~".join(cols[1:])  # expression for covariances
        r_code = r_code + ",location.fun=" + cov_expr + ",scale.fun=" + cov_expr
        if shapeCov:
            r_code += ',shape.fun=' + cov_expr
    if use_weights is not None:# got weights so include that in the R code.
        r_code += ',weights=wt'
    r_code += ')'  # add on the trailing bracket.
    df = pd.DataFrame(np.array(df_data).T, columns=cols)
    if use_weights is not None:
        wts = robjects.vectors.FloatVector(wts)  # convert to a R vector.
        robjects.globalenv['wt'] = wts  # and push into the R environment.
    with (robjects.default_converter + rpandas2ri.converter+numpy2ri.converter).context():
        robjects.globalenv['df'] = df  # push the dataframe with info into R

    try:
        r_fit = robjects.r(r_code)  # do the fit
        fit = base.summary(r_fit, silent=True)  # get the summary fit info
        # extract the data
        params = fit.rx2('par')  # get the parameters.
        se = fit.rx2('se.theta')  # get the std error
        cov_params = fit.rx2('cov.theta')  # get the covariance.
        if isinstance(params, rpy2.rinterface_lib.sexp.NULLType):
            # params not present (for some reason) set everything to nan
            params = np.broadcast_to(np.nan, npts)
        if isinstance(se, rpy2.rinterface_lib.sexp.NULLType):
            # std err not present (for some reason) set everything to nan
            se = np.broadcast_to(np.nan, npts)
        if isinstance(cov_params, rpy2.rinterface_lib.sexp.NULLType):
            # cov not present (for some reason) set everything to nan
            cov_params = np.broadcast_to(np.nan, [npts, npts])
        params = np.array(params)
        se = np.array(se)
        cov_params = np.array(cov_params)

        # ordering is:
        # loc, loc-cov1, loc-cov2,... scale, scale-cov1, scale-cov-2, shape, [shape-cov1 etc]
        # negate shape params as R and python conventions differ.
        start_shape = -1
        if shapeCov:
            start_shape = -ncov
        params[start_shape:] = params[start_shape:] * (-1)
        nllh = np.array(fit.rx2('nllh'))
        aic = np.array(fit.rx2('AIC'))
    except rpy2.rinterface_lib.embedded.RRuntimeError:
        # some problem in R with the fit. Set everything to nan.
        params = np.broadcast_to(np.nan, npts)
        se = np.broadcast_to(np.nan, npts)
        cov_params = np.broadcast_to(np.nan, [npts, npts])
        nllh = np.array([np.nan])
        aic = np.array([np.nan])

    return (params, se, cov_params, nllh, aic)  # return the data.
def xarray_gev(data_array: xarray.DataArray,
               cov: typing.Optional[typing.List[xarray.DataArray]] = None,
               shape_cov=False,
               dim: [typing.List[str], str] = 'time_ensemble',
               file=None, recreate_fit: bool = False,
               verbose: bool = False,
               name: typing.Optional[str] = None,
               weights: typing.Optional[xarray.DataArray] = None,
               **kwargs):
    #
    """
    Fit a GEV to xarray data using R.

    :param data_array: dataArray for which GEV is to be fit
    :param cov: covariate (If None not used) --a list of dataarrays or None.
    :param shape_cov: If True then allow the shape to vary with the covariate.
    :param weights: Weights for each sample. If not specified, no weighting will be done.
    :param dim: The dimension(s) over which to collapse.
    :param file -- if defined save fit to this file. If file exists then read data from it and so not actually do fit.
    :param recreate_fit -- if True even if file exists compute fit.
    :param verbose -- be verbose if True
    :param name: Name of the fit. Stored in result attributes under name.
    :param kwargs: any kwargs passed through to the fitting function
    :return: a dataset containing:
        Parameters -- the parameters of the fit; location, location wrt cov, scale, scale wrt cov, shape, shape wrt cov
        StdErr -- the standard error of the fit -- same parameters as Parameters
        nll -- negative log likelihood of the fit -- measure of the quality of the fit
        AIC -- aitkin information criteria.
    """
    if (file is not None) and file.exists() and (
            not recreate_fit):  # got a file specified, it exists and we are not recreating fit
        data_array = xarray.load_dataset(file)  # just load the dataset and return it
        if verbose:
            print(f"Loaded existing data from {file}")
        return data_array

    kwargs['shapeCov'] = shape_cov

    if cov is None:
        cov = []

    cov_names = [c.name for c in cov]

    ncov = len(cov)
    input_core_dims = [[dim]] * (1 + ncov)
    output_core_dims = [['parameter']] * 2 + [['parameter', 'parameter2'], ['NegLog'], ['AIC']]
    args = [data_array]
    if len(cov):
        args += cov
    if weights is not None:
        args += [weights]
        args = xarray.broadcast(*args)
        input_core_dims+= [[dim]]
        kwargs['use_weights'] = True

    params, std_err, cov_param, nll, AIC = xarray.apply_ufunc(gev_fit, *args,
                                                              input_core_dims=input_core_dims,
                                                              output_core_dims=output_core_dims,
                                                              vectorize=True, kwargs=kwargs)
    pnames = []
    for n in ['location', 'scale', 'shape']:
        pnames += [n]
        if not shape_cov and n == 'shape':
            continue  # no Dshape_dX
        for cn in cov_names:
            pnames += ['D' + n + '_' + cn]

    # name variables and then combine into one dataset.

    params = params.rename("Parameters")
    std_err = std_err.rename("StdErr")
    cov_param = cov_param.rename("Cov")
    nll = nll.rename('nll').squeeze()
    AIC = AIC.rename('AIC').squeeze()
    data_array = xarray.Dataset(dict(Parameters=params, StdErr=std_err, Cov=cov_param, nll=nll, AIC=AIC)).assign_coords(
        parameter=pnames, parameter2=pnames)
    if name:
        data_array.attrs.update(name=name)
    if file is not None:
        file.parent.mkdir(exist_ok=True, parents=True)  # make directory
        data_array.to_netcdf(file)  # save the dataset.
        if verbose:
            print(f"Wrote fit information to {file}")
    return data_array



def convert_to_cftime(date):
    if np.isnat(date):
        breakpoint()
        return None  # or return a default cftime.DatetimeGregorian instance
    date = date.astype('datetime64[s]').tolist()  # Convert to datetime.datetime object
    return cftime.DatetimeGregorian(date.year, date.month, date.day, date.hour, date.minute, date.second)

def convert_to_cftime(date,date_type=cftime.DatetimeGregorian,has_year_zero=True):
    print(type(date))
    date = pd.to_datetime(date)
    return date_type(date.year, date.month, date.day, date.hour, date.minute, date.second,date.microsecond,has_year_zero=has_year_zero)


radar_events = xarray.load_dataset(CPMlib.radar_dir/'radar_events/5km_events_2008_2023.nc')
wts = radar_events.count_cells
fit = xarray_gev(radar_events.max_precip,weights=wts,dim='EventTime')
from xarray.coding.times import convert_time_or_go_back,convert_times
fn = lambda date: convert_time_or_go_back(date,cftime.DatetimeGregorian)
msk= radar_events.t.isnull()
date = radar_events.t#.where(~msk,np.datetime64('1970-01-01'))
#cftime_da = xarray.apply_ufunc(convert_to_cftime,date,vectorize=True).where(~msk,cftime.DatetimeProlepticGregorian(-9999,1,1,has_year_zero=True))  # convert and then convert back to missing data.
cftime_da = xarray.apply_ufunc(convert_to_cftime,date,vectorize=True,kwargs=dict(has_year_zero=False)).where(~msk,np.nan) # mask
obs_cet = commonLib.read_cet()
obs_cet_jja = obs_cet.sel(time=(obs_cet.time.dt.season == 'JJA'))
# now get CET for all the event times
msk =cftime_da.isnull()
cet_interp = obs_cet_jja.sel(time=cftime_da.where(~msk,cftime.DatetimeGregorian(9999,1,1)),method='nearest').where(~msk,np.nan)
radar_events['CET']=cet_interp
#o = obs_cet_jja.sel(time=cftime_da)


