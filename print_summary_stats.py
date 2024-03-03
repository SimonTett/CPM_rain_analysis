#  print out some summary info on events and rain.
import CPMlib
import xarray
import dask

chunks = {"time": 4, "ensemble_member": 1, "grid_latitude": None, "grid_longitude": None}
medians=dict()
# model data
for key,files in zip(['Raw CPM','Filt. CPM'],
                     [CPMlib.CPM_dir.glob("CPM*/*[0-9]*.nc"),
                      CPMlib.CPM_filt_dir.glob("CPM*/*[0-9]*23.nc"),
                      ]):
    print('Dealing with ',key)
    with dask.config.set(**{'array.slicing.split_large_chunks': True}):
        maxRain = xarray.open_mfdataset(files, chunks=chunks,parallel=True).seasonalMax
        if "CPM" in key: # CP<M data -- select to region.
            maxRain=maxRain.sel(**CPMlib.stonehaven_rgn)
    time_slice = slice('2005', '2022')
    L = maxRain.time.dt.season == 'JJA'
    maxRain = maxRain.sel(time=L).sel(time=time_slice)
    maxRain.load() # need to load it to compute median
    med = float(maxRain.median())
    medians[key]=med
    print('Done with ',key)

## get in the radar data
stonehaven_OSGB=dict(zip(["projection_x_coordinate","projection_y_coordinate"],
                        [387171.,785865.])) # from wikipedia https://geohack.toolforge.org/geohack.php?pagename=Stonehaven&params=56.964_N_2.211_W_region:GB_type:city(11150)

stonehaven_rgn_OSGB={k:slice(v-75e3,v+75e3) for k,v in stonehaven_OSGB.items()}
for key,file in zip(['radar_5km','radar_1km_c5','radar_1km_c4'],
    [CPMlib.radar_dir/"summary_5km_1hr_scotland.nc",
    CPMlib.radar_dir/"summary_1km/1km_c5_summary.nc",
    CPMlib.radar_dir/"summary_1km/1km_c4_summary.nc"
        ]):
    print('Dealing with ', key)
    with dask.config.set(**{'array.slicing.split_large_chunks': True}):
        maxRain = xarray.load_dataset(file, chunks=chunks).monthlyMax.resample(time='QS-Dec').max()
        maxRain = maxRain.sel(**stonehaven_rgn_OSGB)
    time_slice = slice('2005', '2022')
    L = maxRain.time.dt.season == 'JJA'
    maxRain = maxRain.sel(time=L).sel(time=time_slice)
    maxRain.load()  # need to load it to compute median
    med = float(maxRain.median())
    medians[key] = med
    print('Done with ', key)
## Now to print them
print(" ".join([f"{k}: {med:3.1f} mm/h" for k,med in medians.items()]))
