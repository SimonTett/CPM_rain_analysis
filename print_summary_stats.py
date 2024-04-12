#  print out some summary info on events and rain.
import pandas as pd

import CPMlib
import xarray
import dask
import CPM_rainlib
import commonLib
my_logger = CPM_rainlib.logger
commonLib.init_log(my_logger, level='INFO')
chunks = {"time": 4, "ensemble_member": 1, "grid_latitude": None, "grid_longitude": None}
medians=dict()
# model data
for key,files in zip(['Raw CPM','Filt. CPM'],
                     [CPMlib.CPM_dir.glob("CPM*/*[0-9]*.nc"),
                      CPMlib.CPM_filt_dir.glob("CPM*/*[0-9]*23.nc"),
                      ]):
    my_logger.info(f'Loading {key}')
    with dask.config.set(**{'array.slicing.split_large_chunks': True}):
        maxRain = xarray.open_mfdataset(files, chunks=chunks,parallel=True).seasonalMax
        maxRain=maxRain.sel(**CPMlib.carmont_rgn).sel(**CPMlib.today_sel)
    L = maxRain.time.dt.season == 'JJA'
    maxRain = maxRain.where(L,drop=True)
    maxRain.load() # need to load it to compute median
    med = maxRain.median(['grid_latitude','grid_longitude','time','ensemble_member'])
    medians[key]=med*med['rolling']
    my_logger.info(f'Loaded {key}')

## get in the radar data

carmont_rgn_OSGB={k:slice(v-75e3,v+75e3) for k,v in CPMlib.carmont_drain_OSGB.items()}
for key in ['5km','1km_c5','1km_c4']:
    file = CPMlib.radar_dir/f"summary/summary_2008_{key}.nc"
    print('Dealing with ', key)
    with dask.config.set(**{'array.slicing.split_large_chunks': True}):
        maxRain = xarray.open_dataset(file, chunks=chunks).Radar_rain_Max.resample(time='QS-Dec').max()
        maxRain = maxRain.sel(**CPMlib.carmont_rgn_OSGB)
    L = maxRain.time.dt.season == 'JJA'
    maxRain = maxRain.where(L,drop=True).sel(**CPMlib.today_sel)
    maxRain.load()  # need to load it to compute median
    med = maxRain.median(['projection_y_coordinate','projection_x_coordinate','time'])
    medians[key] = med*med['rolling']
    my_logger.info(f'Loaded {key}')
## Now to print them out

median_max = pd.concat([ds.drop_vars('coarsen',errors='ignore').rename(k).to_dataframe()
                        for k,ds in medians.items()],axis=1)
rename={v:v.replace('_','-') for v in median_max.columns if '_' in v}
with open(CPMlib.table_dir / 'max_rain.tex', 'w') as f:
    median_max.rename(columns=rename).rename_axis('').style.\
        relabel_index([f'Rx{r:d}h' for r in median_max.index]).format(precision=1).to_latex(
            f, label='tab:max_rain',
            caption="Regional Median Seasonal Max Total Rain (mm)",hrules=True
        )
