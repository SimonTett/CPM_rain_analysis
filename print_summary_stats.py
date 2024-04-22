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
medians = dict()
areas = dict()
event_count = dict()
# model data
topog = xarray.load_dataarray(CPM_rainlib.dataDir / 'cpm_topog_fix_c2.nc').sel(**CPMlib.carmont_rgn)

for key, files, event_file in zip(['Raw CPM', 'Filt. CPM'],
                                  [CPMlib.CPM_dir.glob("CPM*/*[0-9]*.nc"),
                                   CPMlib.CPM_filt_dir.glob("CPM*/*[0-9]*23.nc"),
                                   ],
                                  [CPMlib.CPM_filt_dir / 'CPM_all_events.nc',
                                   CPMlib.CPM_dir / 'CPM_all_events.nc']
                                  ):
    my_logger.info(f'Loading {key}')
    with dask.config.set(**{'array.slicing.split_large_chunks': True}):
        maxRain = xarray.open_mfdataset(files, chunks=chunks, parallel=True).seasonalMax
        maxRain = maxRain.sel(**CPMlib.carmont_rgn).sel(**CPMlib.today_sel)
    L = maxRain.time.dt.season == 'JJA'
    maxRain = maxRain.where(L, drop=True)
    CPM_rainlib.fix_coords(maxRain)
    maxRain = maxRain.where(topog > 0.0)  # land points
    maxRain.load()  # need to load it to compute median
    med = maxRain.median(['grid_latitude', 'grid_longitude', 'time', 'ensemble_member'])
    medians[key] = med * med['rolling']
    my_logger.info(f'Computed median accumulations for {key}')
    # and the event info.
    events = xarray.load_dataset(CPMlib.CPM_dir / 'CPM_all_events.nc')
    time = events.t.isel(quantv=0)
    yr = time.dt.year
    msk = (yr >= 2008) & (yr <= 2023) & (time.dt.season == 'JJA')
    area = events.count_cells.where(msk, drop=True) * (4.4 ** 2)
    areas[key] = area.median(['EventTime', 'ensemble_member'])  # median area
    event_count[key] = area.notnull().sum(['EventTime', 'ensemble_member']) / (16 * 12)  # mean no of events
    my_logger.info(f'Computed event info for {key}')

## get in the radar data

carmont_rgn_OSGB = {k: slice(v - 75e3, v + 75e3) for k, v in CPMlib.carmont_drain_OSGB.items()}
for key in ['5km', '1km_c5', '1km_c4']:
    file = CPMlib.radar_dir / f"summary/summary_2008_{key}.nc"
    print('Dealing with ', key)
    with dask.config.set(**{'array.slicing.split_large_chunks': True}):
        maxRain = xarray.open_dataset(file, chunks=chunks).Radar_rain_Max.resample(time='QS-Dec').max()
        maxRain = maxRain.sel(**CPMlib.carmont_rgn_OSGB)
    L = maxRain.time.dt.season == 'JJA'
    maxRain = maxRain.where(L, drop=True).sel(**CPMlib.today_sel)
    topog = xarray.open_dataarray(CPMlib.radar_dir / 'topography' / f'topog_{key}.nc').sel(**carmont_rgn_OSGB)
    maxRain = maxRain.where(topog > 0.0)  # land points
    maxRain.load()  # need to load it to compute median
    med = maxRain.median(['projection_y_coordinate', 'projection_x_coordinate', 'time'])
    medians[key] = med * med['rolling']
    my_logger.info(f'Processed radar max rain for {key}')
    # now to do the events
    events = xarray.load_dataset(CPMlib.radar_dir / 'radar_events' / f'events_2008_{key}.nc')
    time = events.t.isel(quantv=0)
    yr = time.dt.year
    msk = (yr >= 2008) & (yr <= 2023) & (time.dt.season == 'JJA')
    dx_dy = CPM_rainlib.radar_dx_dy(maxRain)
    area = events.count_cells.where(msk, drop=True) * (dx_dy[0] * dx_dy[1]) / 1e6
    areas[key] = area.mean(['EventTime'])  # mean area
    event_count[key] = area.notnull().sum(['EventTime']) / 16  # mean no of events
    my_logger.info(f'Computed event info for {key}')

## Now to print them out

median_max = pd.concat([ds.drop_vars('coarsen', errors='ignore').rename(k).to_dataframe()
                        for k, ds in medians.items()], axis=1
                       )
rename = {v: v.replace('_', '-') for v in median_max.columns if '_' in v}
with open(CPMlib.table_dir / 'max_rain.tex', 'w') as f:
    median_max.rename(columns=rename).rename_axis('').style. \
        relabel_index([f'Rx{r:d}h' for r in median_max.index]).format(precision=1).to_latex(
        f, label='tab:max_rain', position='ht!',
        caption="Regional Median Seasonal Max Total Rain (mm)", hrules=True
    )

## print out the areas and event_count in a multi-index
df_areas = pd.concat([ds.rename(k).drop_vars(['quantv']).to_dataframe() for k, ds in areas.items()], axis=1)
df_counts = pd.concat([ds.rename(k).drop_vars(['quantv']).to_dataframe() for k, ds in event_count.items()], axis=1)
merged_df = pd.concat([df_areas, df_counts], axis=1, keys=['Area', 'Count']).swaplevel(0, 1, axis=1)
merged_df = merged_df.sort_index(axis=1)
with open(CPMlib.table_dir / 'event_stats.tex', 'w') as f:
    merged_df.rename(columns=rename).rename_axis('').style. \
        relabel_index([f'Rx{r:d}h' for r in merged_df.index]). \
        format(precision=0). \
        to_latex(f,    column_format='l'+'|ll'*5,
                 label='tab:event_stats', position='ht!',
                 caption="Area (km$^2$) and Event count/year", hrules=True
                 )
