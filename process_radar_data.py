#!/bin/env python

"""
Process radar data on JASMIN. Generate monthly-means of rain, monthly extremes and time of monthly extreme.


"""
import pathlib
import CPM_rainlib
import xarray
import numpy as np
import pandas as pd
import argparse
import resource  # so we can see memory usage every month.
import warnings
import logging
import typing
import sys

warnings.filterwarnings('ignore', message='.*Vertical coord 5 not yet handled.*')
# filter out annoying warning from iris.

my_logger = logging.getLogger(__name__)  # for logging
max_memory = 0.0 # used as a global value
def memory_use() -> str:
    """
    Report on Memory Use.
    Returns: a string with the memory use in Gbytes or "Mem use unknown" if unable to load resource
     module.

    """
    # Based on https://medium.com/survata-engineering-blog/monitoring-memory-usage-of-a-running-python-program-49f027e3d1ba
    global max_memory # every time we sample memory need to update the global value.
    try:
        import resource
        mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
        max_memory = max(mem,max_memory)
        mem = f"Mem: {mem:7.2f} Max Mem: {max_memory:7.2f} Gbytes"
    except ModuleNotFoundError:
        mem = 'Mem use unknown'
    return mem
    

def time_process(da:xarray.DataArray,
                 rolling:typing.List[int]=[1]):
    """
    Process a dataArray of (daily) data
    :param dataarray -- Dataset to process
    :param rolling. List of rolling periods to use.
    """

    mx=[]
    mx_time=[]
    for roll in rolling:
        if roll == 1: # skip rolling!
            roll_var =  da
        else:
            roll_var =  da.rolling(time=roll,center=True).mean()
        mx_r = roll_var.max('time', keep_attrs=True).rename(f'{da.name}_Max')  # max
        mx_time_r = roll_var.where(mx_r >0).idxmax('time').rename(f'{da.name}_MaxTime') # only want > 0
        
        mx.append(mx_r.assign_coords(rolling=roll))
        mx_time.append(mx_time_r.assign_coords(rolling=roll))
    
    mx=xarray.concat(mx,dim='rolling')
    mx_time=xarray.concat(mx_time,dim='rolling')

    time_max = da.time.max().values
    time_min = da.time.min().values
    # TODO modify to make a total.
    mn = da.mean('time', keep_attrs=True).rename(f'{da.name}_Mean')
    ds = xarray.merge([mn, mx, mx_time])
    ds['time_bounds']=xarray.DataArray([time_min,time_max],coords=dict(bounds=[0,1]))

    return ds

def end_period_process(dailyData:typing.List,
                       outDir:pathlib.Path, 
                       resoln:str, 
                       period:str='1M', 
                       coarsens:typing.List[int]=[1],
                       rolling:typing.List[int]=[1]):
    """
    Deal with data processing at end of a period -- normally a calendar month.
    :param dailyData -- input list of daily datasets
    :param outDir -- output directory where data will be written to
    :param  resoln -- resolution string
    :param period (default '1M'). Time period over which data to be resampled to
    :param coarsens: List of coarsenings to apply to spatial data.
    :param rolling. List of rolling periods to apply to data in time.
    """
    if len(dailyData) == 0:
        return []  # nothing to do so return empty list 
    name_keys = {'1D': 'daily', '1M': 'monthly'}
    no_days = len(dailyData)
    summary_prefix = name_keys.get(period, period)
    outDir.mkdir(exist_ok=True,parents=True) # create the outdir
    rain=xarray.concat(dailyData,dim='time')
    # pull out no of samples and compute total number of samples in period
    No_samples=rain['no_samples'].resample(time=period).sum() # no of samples
    fmt_str = '%Y-%m-%dT%H'
    start_time = rain.time.min().dt.strftime(fmt_str).values.tolist()
    end_time = rain.time.max().dt.strftime(fmt_str).values.tolist()



    # Handle Coarsening
    resultDS=dict()
    rain_var = "Radar_rain"
    for coarsen in coarsens:
        if coarsen == 1:
            coarsened_da = rain[rain_var] # no coarsening needed!
            key = f'{resoln}'
        else:
            c_dict=dict(projection_x_coordinate=coarsen,
                        projection_y_coordinate=coarsen)
            coarsened_da =rain[rain_var].coarsen(boundary='pad',**c_dict).mean().\
                          assign_coords(dict(coarsen=coarsen))
            key = f'{resoln}_c{coarsen}'

        outFile = outDir / "_".join(
            ["radar_rain", start_time,end_time, key+'.nc'])

        resampDS = coarsened_da.resample(time=period).map(time_process, rolling=rolling)
        encoding = dict()
        comp = dict(zlib=True)
        for v in resampDS.data_vars:
            encoding[v] = comp

        resampDS['No_samples'] = No_samples # add in the number of samples
        resampDS.to_netcdf(outFile, encoding=encoding)
        my_logger.info(f"Wrote summary data for {len(dailyData)} days for {resampDS.time.max().values} to {outFile}")
        resultDS[key]=resampDS

    return resultDS  # return summary dataset

def init_log(log: logging.Logger,
             level: str,
             log_file: typing.Optional[typing.Union[pathlib.Path, str]] = None,
             mode: str = 'a'):
    """
    Set up logging on a logger! Will clear any existing logging
    :param log: logger to be changed
    :param level: level to be set.
    :param log_file:  if provided pathlib.Path to log to file
    :param mode: mode to open log file with (a  -- append or w -- write)
    :return: nothing -- existing log is modified.
    """
    log.handlers.clear()
    log.setLevel(level)
    formatter = logging.Formatter('%(asctime)s %(levelname)s:  %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')
    ch = logging.StreamHandler(sys.stderr)
    ch.setFormatter(formatter)
    log.addHandler(ch)
    # add a file handler.
    if log_file:
        if isinstance(log_file, str):
            log_file = pathlib.Path(log_file)
        log_file.parent.mkdir(exist_ok=True, parents=True)
        fh = logging.FileHandler(log_file, mode=mode + 't')  #
        fh.setLevel(level)
        fh.setFormatter(formatter)
        log.addHandler(fh)
    log.propagate = False
# read cmd line args.
parser = argparse.ArgumentParser(
    description="Process UK Nimrod Radar Data to compute hourly maxes on monthly timescales")

parser.add_argument('year', type=int, nargs='+', help='years to process')
parser.add_argument('--resolution', '-r', type=str,
                    help='Resolution wanted', choices=['1km', '5km'], default='5km')
parser.add_argument('--glob', type=str, help='Pattern for glob month/day matching -- i.e. 0[6-8][0-3][0-9]',
                    default='[0-1][0-9][0-3][0-9]')
parser.add_argument('--test', '-t', action='store_true',
                    help='If set run in test mode -- no data read or generated')
parser.add_argument('--outdir', '-o', type=str, help='Name of output directory')
parser.add_argument('--log_level', '-l', type=str,
                    choices=['DEBUG','INFO','WARNING'],default='WARNING',
                    help='logging level')
parser.add_argument('--region', nargs=4, type=float, help='Region to extract (x0, x1,y0,y1)')

parser.add_argument('--minhours', type=int,
                    help='Minium number of unique hours in data to process daily data (default 6 hours)', default=6)
parser.add_argument('--minsamples',type=int,help='Minimum number of samples in a time sample')
parser.add_argument("--resample", type=str, help='Time to resample input radar data to (default = 1h)', default='1h')
parser.add_argument("--coarsen",type=int,nargs="+",
                    help='Spatial coarsenings to apply *before* time resampling',default=[1])
parser.add_argument("--rolling",type=int,nargs="+",help='Rolling times to apply before computing maxes',default=[1])
args = parser.parse_args()

init_log(my_logger, args.log_level)


glob_patt = args.glob
test = args.test
resoln = args.resolution
outdir = args.outdir
if outdir is None:  # default
    outdir = CPM_rainlib.outdir / f'summary_{resoln}'
else:
    outdir = pathlib.Path(outdir)


region = None
if args.region is not None:
    region = dict(
        projection_x_coordinate=slice(args.region[0], args.region[1]),
        projection_y_coordinate=slice(args.region[2], args.region[3])
    )

resample_prd = args.resample
# log cmd args. 
my_logger.info("Command line args: \n"+repr(args))

minhours = args.minhours
coarsens=args.coarsen # list of coarsening values.

if test:
    print(f"Would create {outdir}")
else:
    outdir.mkdir(parents=True, exist_ok=True)  # create directory if needed

# initialise -- 
last_month = None
last_time = None
for year in args.year:
    dataYr = CPM_rainlib.nimrodRootDir / f'uk-{resoln}/{year:04d}'
    # initialise the list...
    files = sorted(list(dataYr.glob(f'metoffice*{year:02d}{glob_patt}*-composite.dat.gz.tar')))
    daily_rain=[] 
    for f in files:
        if test:  # test mode
            print(f"Would read {f} but in test mode")
            continue
        rain = CPM_rainlib.extract_nimrod_day(f, QCmax=400, check_date=True, region=region)
        if (rain is None) or (len(np.unique(rain.time.dt.hour)) <= minhours):  # want min hours hours of data
            my_logger.warning(f"Not enough data for {f} ")
            if rain is None:
                my_logger.warning("No data at all")
            else:
                my_logger.warning(f"Only {len(np.unique(rain.time.dt.hour))} unique hours")

            last_time = None  # no last data for the next day
            continue  # no or not enough data so onto the next day
        # now we can process a days worth of data.    
        no_samples = rain.time.resample(time=resample_prd).count().rename('no_samples')
        
        rain = rain.resample(time=resample_prd).mean(keep_attrs=True).rename('Radar_rain')  # mean rain/hour units mm/hr
        if args.minsamples:
            rain = rain.where(no_samples >= args.minsamples)
        ds=xarray.merge([rain,no_samples])
        my_logger.debug(memory_use())
        my_logger.debug(f"{f} @  {rain.time.min().values}")

        # this block will only be run if last_month is not None and month 
        # has changed. Taking advantage of lazy evaluation.
        if (last_month is not None) and   (rain[0].time.dt.month != last_month): 
            # change of month -- could be quarter etc
            # starting a new month so save data and start again.
            my_logger.info("Starting a new period. Summarizing and writing data out")
            # process and write out data

            summaryDS = end_period_process(daily_rain, outdir,resoln,period='1M',
                                           coarsens=coarsens, rolling=args.rolling)

            my_logger.info(memory_use()) # print out memory use.


            # new month so clean up daily_rain
            daily_rain=[]

        daily_rain.append(ds) # append the ds we just generated to list to process at end of next block!

        #  figure our last period and month.
        last_time = rain.time[-1]  # get last hour.
        last_month = last_time.dt.month
        # print("End of loop",last_month)
        # done loop over files for this year
    # end loop over years.

# and we might have data to write out!
summaryDS = end_period_process(daily_rain,outdir,resoln, period='1M',
                               coarsens=coarsens,rolling=args.rolling) # process and write out data

my_logger.info("All done")
my_logger.info(memory_use())
