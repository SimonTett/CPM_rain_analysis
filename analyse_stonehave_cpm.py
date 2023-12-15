# Analyse the Stonehaven event using only the CPM data.
# Rain coming from the radar but that is it!
# get the radar data and slice out what we wanT
import cartopy.crs as ccrs
import CPMlib
import edinburghRainLib
import commonLib
import xarray
import scipy.stats
import numpy as np
from R_python import gev_r

proj=ccrs.PlateCarree()
projGB=ccrs.OSGB()
v=tuple(CPMlib.stonehaven.values())
stonehaven = projGB.transform_point(*v,CPMlib.projRot)
var_names = ["projection_x_coordinate","projection_y_coordinate"]
stonehaven = dict(zip(var_names,stonehaven))
stonehaven_rgn={k:slice(v-75e3,v+75e3) for k,v in stonehaven.items()}
path=CPMlib.datadir/'summary_5km_1hr_scotland.nc'
rseasMskmax, mxTime=edinburghRainLib.get_radar_data(path,region=stonehaven_rgn,height_range=slice(1,300))
rseasMskmax=rseasMskmax.sel(time='2020-06')
mxTime = mxTime.sel(time='2020-06')
carmont_plus = CPMlib.carmont_OSGB.copy()
carmont_plus['projection_y_coordinate']+=500.
carmont_plus['projection_x_coordinate']+=000.

rains=[]
for locn in [CPMlib.carmont_OSGB,carmont_plus]:
    rains.append(float(rseasMskmax.sel(locn,method='nearest'))) # rains we want.
rains=np.array(rains)

obs_cet=commonLib.read_cet() # read in the obs CET
obs_cet_jja = obs_cet.sel(time=(obs_cet.time.dt.season == 'JJA'))

cpm_cet=xarray.load_dataset(CPMlib.CPM_dir/"cet_tas.nc")
cpm_cet=cpm_cet.tas.resample(time='QS-DEC').mean() # resample to seasonal means
cpm_cet_jja = cpm_cet.sel(time=(cpm_cet.time.dt.season=='JJA')) # and pull out summer.

# Load up CPM extreme rainfall data, select to region of interest and mask
rgn_all=dict(longitude=slice(357.5,361.0),latitude=slice(1.5,7.5)) # region for which extraction was done.


paths = sorted((CPMlib.CPM_dir).glob('CPM*/*.nc'))
print("Opening data")
ds=xarray.open_mfdataset(paths)
ds=ds.sel(time=(ds.time.dt.season=='JJA')) # get summer only
ds = ds.sel(CPMlib.carmont,method='nearest') # reduce to rgn of interest
ds=ds.stack(indx=[...]).dropna('indx').load()
cet=cpm_cet_jja.T.stack(indx=["time","ensemble_member"]).rename("CET")
coords_to_drop=['grid_latitude',
 'grid_longitude',
 'ensemble_member_id',
 'latitude',
 'longitude',
 'month_number',
 'year',
 'yyyymmddhh',
]
cet = cet.drop_vars(coords_to_drop,errors='ignore')
ds=ds.drop_vars(coords_to_drop,errors='ignore')
fit = gev_r.xarray_gev(ds.seasonalMax, cov=[cet], dim='indx')

t_today = obs_cet_jja.sel(time=slice('2014','2023')).mean()
t_PI = float(obs_cet_jja.sel(time=slice('1850','1899')).mean())
# compute the fits!

delta_PI = t_PI-t_today
delta_P2K = t_PI+2-t_today

covariates=[t_today]
CPM_today=CPMlib.location_scale(fit.Parameters,covariates=covariates,scaled=True)

PI_location = CPM_today.location*(1+CPM_today.Dlocation_CET*delta_PI)
PI_scale = CPM_today.scale*(1+CPM_today.Dscale_CET*delta_PI)
P2K_location =  CPM_today.location*(1+CPM_today.Dlocation_CET*delta_P2K)
P2K_scale =  CPM_today.scale*(1+CPM_today.Dscale_CET*delta_P2K)

gev_PI=scipy.stats.genextreme( fit.Parameters.sel(parameter='shape'),loc=PI_location,scale=PI_scale)
gev_today =scipy.stats.genextreme(fit.Parameters.sel(parameter='shape'),loc=CPM_today.location,scale=CPM_today.scale)
gev_P2K = scipy.stats.genextreme( fit.Parameters.sel(parameter='shape'),loc=P2K_location,scale=P2K_scale)

p_today = gev_today.sf(rains)
p_PI=gev_PI.sf(rains)
p_P2K=gev_P2K.sf(rains)
PR= p_today/p_PI
PR_P2K= p_P2K/p_PI

# now compute the intensity values for today, PI and P+2K
I_today = gev_today.isf(p_today)

I_PI = gev_PI.isf(p_today)
I_P2K = gev_P2K.isf(p_today)
I_today_ratio = I_today/I_PI-1
I_P2K_ratio = I_P2K/I_PI-1
for indx,r in enumerate(rains.flatten()):
    rp_str=f"RP PI: {1.0/float(p_PI[indx]):3.0f} RP today : {1.0/float(p_today[indx]):3.0f} Summers"
    pr_str =f"PR today: {PR[indx]:3.2f} PR+2K: {PR_P2K[indx]:3.2f}"
    Int_str=f"Int change Today: {I_today_ratio[indx]:3.2f} +2K: {I_P2K_ratio[indx]:3.2f}"
    print(f"Rain: {float(r):3.1f} mm/hr",rp_str,pr_str,Int_str)