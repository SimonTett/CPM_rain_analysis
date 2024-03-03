# Generate event datasets from radar data
# also do GEV fit
import pathlib
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import CPMlib
import xarray
import CPM_rainlib
import numpy as np
import commonLib
import datetime
import scipy.stats
from R_python import gev_r

recreate=False
proj=ccrs.PlateCarree()
projGB=ccrs.OSGB()
stonehaven_OSGB=dict(zip(["projection_x_coordinate","projection_y_coordinate"],
                        [387171.,785865.])) # from wikipedia https://geohack.toolforge.org/geohack.php?pagename=Stonehaven&params=56.964_N_2.211_W_region:GB_type:city(11150)

stonehaven_rgn={k:slice(v-75e3,v+75e3) for k,v in stonehaven_OSGB.items()}


# get the  summer mean CET out and force its time's to be the same.
obs_cet = commonLib.read_cet()  # read in the obs CET
obs_cet_jja = obs_cet.sel(time=(obs_cet.time.dt.season == 'JJA'))
summary_files = ['summary_5km_1hr_scotland.nc','summary_1km/1km_c5_summary.nc','summary_1km/1km_summary.nc','summary_1km/1km_c2_summary.nc',
                 'summary_1km/1km_c4_summary.nc','summary_1km/1km_c8_summary.nc']
summary_files = [CPMlib.radar_dir/f for f in summary_files]
for path,resoln,name in zip(summary_files,
                          [5000,5000,1000,2000,4000,8000],
                          ['5km hourly','coarsened 1km  to 5km hourly','1km hourly','coarsened 1km to 2km hourly','coarsened 1km to 4km hourly','coarsened 1km to 8km hourly']):
    out_file = path.name.replace("_summary", "")
    grid = int(resoln/90.)
    radar_dataset = CPM_rainlib.comp_event_stats(path, region=stonehaven_rgn, topog_grid=grid, height_range=slice(1, 300))
    # get the CET to the right times,
    tc = np.array([f"{int(y)}-06-01" for y in radar_dataset.t.isel(quantv=0).dt.year])
    cet = obs_cet_jja.interp(time=tc).rename(dict(time='EventTime'))
    # convert EventTime to an index.
    etime = np.arange(0, len(radar_dataset.EventTime))
    cet = cet.assign_coords(dict(EventTime=etime))
    radar_dataset = radar_dataset.assign_coords(dict(EventTime=etime))
    radar_dataset['CET'] = cet  # add in the CET data.


    # remove unneded dimensions
    dims = list(radar_dataset.dims)
    coords_to_drop = [c for c in radar_dataset.coords if c not in dims]
    radar_dataset = radar_dataset.drop_vars(coords_to_drop)
    # add some attributes
    source_str=f'Processed NIMROD RADAR data from {name} to events using comp_radar_events.py on {datetime.datetime.now()}'
    radar_dataset.attrs.update(source=source_str)

    opath = CPMlib.radar_dir/f"radar_events_{out_file}"
    opath.parent.mkdir(exist_ok=True)
    radar_dataset.to_netcdf(opath)
    print(f"Saved {radar_dataset.attrs['source']} to {opath}")

    # do GEV fits
    out_dir =  CPMlib.fit_dir/'radar'
    out_dir.mkdir(parents=True,exist_ok=True)
    ln_radar_area = (np.log10(radar_dataset.count_cells * (resoln/1000)**2)).rename('log10_area')
    opath = out_dir/f'fit_{out_file}'
    fit_nocov = gev_r.xarray_gev(radar_dataset.max_precip, dim="EventTime", file=opath, recreate_fit=recreate,
                                 verbose=True, name='No_cov')

    opath = out_dir/f'fit_area_{out_file}'

    fit_area = gev_r.xarray_gev(radar_dataset.max_precip, cov=[ln_radar_area], dim="EventTime", file=opath,
                                recreate_fit=recreate, verbose=True, name='lnA')
    opath = out_dir/f'fit_area_ht_{out_file}'
    fit_area_ht = gev_r.xarray_gev(radar_dataset.max_precip, cov=[ln_radar_area, radar_dataset.height], dim="EventTime",
                                   file=opath, recreate_fit=recreate, verbose=True, name='lnA+z')

    print("DOne fits")

## lets check there are OK and plot them
files=pathlib.Path(CPMlib.radar_dir).glob("*.nc")
datasets={}
for file in files:
    datasets[file.name]=xarray.load_dataset(file)

# now lets plot things on a common scale.
qv=0.9
key=list(datasets.keys())[0]

bins = np.histogram_bin_edges(datasets[key].max_precip.sel(quantv=qv),50)
fig,axis = plt.subplots(nrows=1,ncols=2,figsize=[11,8],num='plt_radar_events',clear=True)
for name,ds in datasets.items():
    ds.max_precip.sel(quantv=qv).plot.hist(bins=bins,density=True,histtype='step',label=name,ax=axis[0])
    p = scipy.stats.genextreme.fit(ds.max_precip.sel(quantv=qv))
    print(name,f"Shape: {p[0]:4.2f} Loc:{p[1]:4.1f} Scale:{p[2]:4.1f}")
axis[0].legend()
axis[0].set_yscale('log')

fig.tight_layout()
fig.show()



