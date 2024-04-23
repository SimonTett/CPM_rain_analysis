# actually analyse the stonehaven event/
import scipy.stats
import xarray
import numpy as np
import edinburghRainLib
import CPMlib
import cartopy.crs as ccrs
import commonLib
import typing
import logging



proj=ccrs.PlateCarree()
projGB=ccrs.OSGB()
v=tuple(CPMlib.stonehaven.values())
stonehaven = projGB.transform_point(*v,CPMlib.projRot)
var_names = ["projection_x_coordinate","projection_y_coordinate"]
stonehaven = dict(zip(var_names,stonehaven))
stonehaven_rgn={k:slice(v-75e3,v+75e3) for k,v in stonehaven.items()}
# load the data wanted.
path = CPMlib.datadir/"radar_events.nc"
radar_events = xarray.load_dataset(path)
CPM_events = xarray.load_dataset(CPMlib.CPM_dir/"CPM_all_events.nc")
radar_fit = xarray.load_dataset(CPMlib.fit_dir/'radar_fit_lnarea.nc')
CPM_fit = xarray.load_dataset(CPMlib.fit_dir/'fit_cet_lnarea.nc')
# work out Dparam

CPM_lnarea=np.log10(CPM_events.count_cells*4.4**2).stack(indx=[...]).dropna('indx').rename('log10_area')
CPM_lnarea_quants=CPM_lnarea.quantile(CPM_events.quantv).rename(quantile='area_quant')
radar_lnarea=np.log10(radar_events.count_cells*5**2).rename('log10_area')
radar_lnarea_quants=radar_lnarea.quantile(radar_events.quantv).rename(quantile='area_quant')
L=(radar_events.t.sel(quantv=0).dt.strftime("%Y-%m-%d") == '2020-08-12').drop_vars("quantv")
path=CPMlib.datadir/'summary_5km_1hr_scotland.nc'
rseasMskmax, mxTime=edinburghRainLib.get_radar_data(path,region=stonehaven_rgn,height_range=slice(1,300))
rseasMskmax=rseasMskmax.sel(time='2020-06')
mxTime = mxTime.sel(time='2020-06')
print(radar_events.max_precip.sel(EventTime=L).values)
carmont_plus = CPMlib.carmont_OSGB.copy()
carmont_plus['projection_y_coordinate']+=500.
carmont_plus['projection_x_coordinate']+=000.

rains=[]
for locn in [CPMlib.carmont_OSGB,carmont_plus]:
    rains.append(float(rseasMskmax.sel(locn,method='nearest')))
rains = np.array(rains).reshape((-1,1,1))

# compute location, scale and shapes for now.
obs_cet=commonLib.read_cet() # read in the obs CET
obs_cet_jja = obs_cet.sel(time=(obs_cet.time.dt.season == 'JJA')).rename("CET")
t_radar = float(obs_cet_jja.sel(time=slice('2005','2023')).mean())
t_today = obs_cet_jja.sel(time=slice('2014','2023')).mean()
t_PI = float(obs_cet_jja.sel(time=slice('1850','1899')).mean())
# compute the fits!



covariates=[t_today,CPM_lnarea_quants]
CPM_today=CPMlib.location_scale(CPM_fit.Parameters,covariates=covariates,scaled=True)


radar_covariates=[radar_lnarea_quants]
radar_today =CPMlib.location_scale(radar_fit.Parameters,covariates=radar_covariates,scaled=True)
delta_PI = t_PI-t_radar
delta_today = t_today-t_radar
delta_P2K = t_PI+2-t_today
PI_location = radar_today.location*(1+CPM_today.Dlocation_CET*delta_PI)
PI_scale = radar_today.scale*(1+CPM_today.Dscale_CET*delta_PI)
today_location = radar_today.location*(1+CPM_today.Dlocation_CET*delta_today)
today_scale = radar_today.scale*(1+CPM_today.Dscale_CET*delta_today)
P2K_location = radar_today.location*(1+CPM_today.Dlocation_CET*delta_P2K)
P2K_scale = radar_today.scale*(1+CPM_today.Dscale_CET*delta_P2K)

gev_PI=scipy.stats.genextreme(radar_fit.Parameters.sel(parameter='shape'),loc=PI_location,scale=PI_scale)
gev_today =scipy.stats.genextreme(radar_fit.Parameters.sel(parameter='shape'),loc=today_location,scale=today_scale)
gev_P2K = scipy.stats.genextreme(radar_fit.Parameters.sel(parameter='shape'),loc=P2K_location,scale=P2K_scale)
# prob ratio!
p_today = gev_today.sf(rains).mean(axis=-1).mean(axis=-1)
p_PI=gev_PI.sf(rains).mean(axis=-1).mean(axis=-1)
p_P2K=gev_P2K.sf(rains).mean(axis=-1).mean(axis=-1)
PR= p_today/p_PI
PR_P2K= p_P2K/p_PI

# now compute the intensity values for today, PI and P+2K
I_today = gev_today.isf(p_today.reshape(rains.shape)).mean(axis=-1).mean(axis=-1)
logging.warning("Today intensity != rains. Understand! ")
I_PI = gev_PI.isf(p_today.reshape(rains.shape)).mean(axis=-1).mean(axis=-1)
I_P2K = gev_P2K.isf(p_today.reshape(rains.shape)).mean(axis=-1).mean(axis=-1)
I_today_ratio = I_today/I_PI-1
I_P2K_ratio = I_P2K/I_PI-1
for indx,r in enumerate(rains.flatten()):
    rp_str=f"RP PI: {1.0/float(p_PI[indx]):3.0f} RP today : {1.0/float(p_today[indx]):3.0f} Summers"
    pr_str =f"PR today: {PR[indx]:3.2f} PR+2K: {PR_P2K[indx]:3.2f}"
    Int_str=f"Int change Today: {I_today_ratio[indx]:3.2f} +2K: {I_P2K_ratio[indx]:3.2f}"
    print(f"Rain: {float(r):3.1f} mm/hr",rp_str,pr_str,Int_str)

