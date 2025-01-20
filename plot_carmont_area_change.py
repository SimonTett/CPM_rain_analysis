# plot the "restricted" area (5x5 region) centred on carmont drain area.
import xarray
import matplotlib.pyplot as plt
import CPMlib
import CPM_rainlib
import commonLib
import pathlib
import flox # make sure we have it!
import numpy as np
my_logger = CPM_rainlib.logger
commonLib.init_log(my_logger,level='DEBUG')
ds = xarray.open_mfdataset(CPMlib.CPM_filt_dir.glob("*/CPM*11_30_23.nc"), parallel=True)
CPM_rainlib.fix_coords(ds)
L = ds.time.dt.season == 'JJA'
topog = xarray.load_dataset(CPM_rainlib.dataDir/'cpm_topog_fix_c2.nc').ht.sel(**CPMlib.carmont_rgn)
ds = ds.where(L,drop=True).sel(**CPMlib.carmont_rgn)
my_logger.info('opened and filtered data')
dayofyear = ds.seasonalMaxTime.dt.dayofyear.where(topog > 0)
mx_rain = ds.seasonalMax.where(topog > 0)
my_logger.info('computed day of year and max rain')
# extra args seems to be needed when flox running.
cnt_doy=dayofyear.groupby(dayofyear).count(CPM_rainlib.cpm_horizontal_coords,fill_value=np.nan,method='map-reduce')
mn_rain = mx_rain.groupby(dayofyear).mean(CPM_rainlib.cpm_horizontal_coords,fill_value=np.nan,method='map-reduce')
my_logger.info('computed counts and means')
q_area = cnt_doy.quantile([0,0.05,0.5,0.95,1.0],'dayofyear',method='nearest').load()
q_mn = mn_rain.quantile([0,0.05,0.5,0.95,1.0],'dayofyear').load()
my_logger.info('computed quantiles')