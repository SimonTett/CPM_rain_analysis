#!/bin/env python
"""
Compute monthly-mean timeseries for covariates and t/s of interest from CPM.
The following are computed:
1) Saturated vap pressure from daily/grid point temperature for carmont region
2) CET, carmont regional temperature and CPM temperature
3) carmont region and CPM region mean precip.

all ts compute avg and and land only ts.


"""
import CPMlib
import CPM_rainlib
import cartopy.crs as ccrs
import xarray
import typing
import numpy as np
import commonLib
import logging
my_logger = CPM_rainlib.logger
commonLib.init_log(my_logger,level='INFO')
test = False # set True if want to run in test mode -- limited time periods and number of files

def SVP(temperature, Huang: bool = False):
    """
    SVP from temperature.
    :param temperature: temperature (in degrees c)
    :param Huang -- if True use the Huang formula (eqn 19 )
       otherwise use the improved Magnus form (Eqn 3)
       Note improved Magnus calculation is about twice as fast as Huang calculation.
    :return: saturated humidity vp (in Pa)
       see  https://journals.ametsoc.org/view/journals/apme/57/6/jamc-d-17-0334.1.xml
    """
    if Huang:
        es = np.exp(34.494 - (4924.99 / (temperature + 237.1)))
        es /= (temperature + 105.0) ** 1.57
    else:
        es = 610.942 * np.exp(17.625 * temperature / (temperature + 243.04))
    return es


def comp_mean_svp(temperature: xarray.DataArray,
                  dimensions: typing.Optional[typing.List[str]] = None,
                  Huang: bool = False) -> xarray.DataArray:
    """
    Compute the mean SVP over all dimensions from TAS. Uses SVP function to do calculatin.
    :param temperature -- temperature (in C) to compute SVP from
    :dimensions -- list of dimensions over which to compute mean.
        If not specified mean will be computed over all dimensions
    :parm Huang -- passed through to SVP function.
    Returns SVP using SVP function 
    """
    result = SVP(temperature, Huang=Huang)  # SVP wants units of C.
    result = result.mean(dimensions)
    return result


wts = dict(
    Rothampsted=1.0 / 3,
    Malvern=1.0 / 3.,
    Squires_Gate=1.0 / 6,
    Ringway=1.0 / 6
)
coords = dict(
    carmont=CPMlib.carmont_drain_long_lat,
    Rothampsted=(-0.35, 51.8),
    Malvern=(-2.317, 52.117),
    Squires_Gate=(-3.033, 53.767),
    Ringway=(-2.267, 53.35)
)

proj = ccrs.PlateCarree()
# compute rotated co-ords
rotated_coords = {k: CPMlib.projRot.transform_point(*c, proj) for k, c in coords.items()}
# add 360 to the x coord to match co-ords of grid
rotated_coords = {k: [c[0] + 360, c[1]] for k, c in rotated_coords.items()}

# get in the topographry
sim_topog=xarray.load_dataset(CPM_rainlib.dataDir/'orog_land-cpm_BI_2.2km.nc',decode_times=False).ht.squeeze()
# and fix co-ord names
sim_topog=sim_topog.rename(dict(longitude='grid_longitude',latitude='grid_latitude'))
# and grid
ref_file = next((CPM_rainlib.cpmDir/'01/tas/mon/latest').glob('*.nc')) # get a file to get the grid from
ref_da = xarray.open_dataset(ref_file)['tas']
sim_topog = CPM_rainlib.fix_cpm_topog(sim_topog, ref_da)
## now to read the ensemble data.

offset = 0.75  # degrees offset. 1 degree is ~ 100 km
region = dict(zip(['grid_longitude', 'grid_latitude'],
                  [slice(v - offset, v + offset) for v in rotated_coords['carmont']]))


chunks = dict(grid_longitude=10, grid_latitude=10)
outdir=CPM_rainlib.dataDir/'CPM_ts'
outdir.mkdir(parents=True,exist_ok=True)
if test:
    file_pattern = '*199*.nc' # for testing
    region.update(time=slice('1989-12-01', '1992-11-30'))
    my_logger.warning('Running in test mode')
else:
    file_pattern='*.nc'
# Compute SVP from daily mean temperature.


mn_svp=[]
for p in CPM_rainlib.cpmDir.glob('[0-9][0-9]'):
    pth = p / 'tas/day/latest'
    ncfiles = list(pth.glob(file_pattern))
    my_logger.info(f"opening {len(ncfiles)} files for svp calculation")
    da = xarray.open_mfdataset(ncfiles,
                               chunks=chunks, parallel=True,
                               concat_dim='time', combine='nested',
                               data_vars='minimal', coords='minimal',
                               compat='override')['tas']
    my_logger.info(f"Opened files for ensemble {int(da.ensemble_member.values[0]):d} {str(da.ensemble_member_id.values[0])}")
    da = da.sel(**region)
    my_logger.info('Selected region')
    ts= da.resample(time='MS').map(comp_mean_svp,dimensions=['time','grid_latitude','grid_longitude']).load()
    ts=ts.rename('Reg SVP')
    ts_land = da.where(sim_topog > 0.0).resample(time='MS').map(comp_mean_svp,dimensions=['time','grid_latitude','grid_longitude']).load()
    ts_land = ts_land.rename('Land Reg SVP')
    my_logger.info('Computed mean SVP from daily data')
    ds_svp = xarray.Dataset(dict(svp=ts,land_svp=ts_land))
    mn_svp.append(ds_svp)

mn_svp=xarray.concat(mn_svp,dim='ensemble_member')
# write the data out
outfile=outdir/'rgn_svp.nc'
mn_svp.to_netcdf(outfile)
my_logger.info(f'Wrote regional SVP to {outfile}')

for var in ['tas','pr']:
    cpm_list = []
    rgn_list = []
    cet_list = []
    for p in CPM_rainlib.cpmDir.glob('[0-9][0-9]'):
        pth = p / var / 'mon/latest'
        pth_rain = p / 'pr/mon/latest'
        ncfiles = list(pth.glob(file_pattern))
        my_logger.info(f"opening {len(ncfiles)} files for {var}")
        da = xarray.open_mfdataset(ncfiles,
                                   chunks=chunks, parallel=True,
                                   concat_dim='time', combine='nested',
                                   data_vars='minimal', coords='minimal',
                                   compat='override')[var]
        my_logger.info(
            f"Opened files for {var} ensemble {int(da.ensemble_member.values[0]):d} {str(da.ensemble_member_id.values[0])}")

        # not too bad performance! The extra args come from the xarray doc.
        if var == 'tas':  # CET is temperature
            cet = 0.0
            for key, wt in wts.items():
                coords = rotated_coords[key]
                ts = da.sel(method='nearest', tolerance=0.1,
                            grid_longitude=coords[0],
                            grid_latitude=coords[1]).load()
                cet += (ts * wt)

            cet_list.append(cet)  # append the cet
            my_logger.info(f'Computed CET')
        rgn_da = da.sel(**region)
        rgn_ts = rgn_da.mean(CPM_rainlib.cpm_horizontal_coords).load().rename('Reg '+var)
        rgn_ts_land = rgn_da.where(sim_topog > 0.0).mean(CPM_rainlib.cpm_horizontal_coords).load().rename('Land Reg '+var)

        rgn_list.append(xarray.Dataset({var: rgn_ts, 'land_'+var: rgn_ts_land}))

        my_logger.info(f"Done with {p} for {var}")  # end loop over ensemble members

    if var == 'tas':
        my_logger.info("Dealing with CET")
        cet = xarray.concat(cet_list, dim='ensemble_member')
        cet.to_netcdf(outdir/'cet.nc')

    rgn_data = xarray.concat(rgn_list, dim='ensemble_member')
    outpath    = outdir/f'reg_{var}.nc'
    rgn_data.to_netcdf(outpath)

    my_logger.info(f"Done with {var}. Data written to {outpath}")
    # end loop over variables
