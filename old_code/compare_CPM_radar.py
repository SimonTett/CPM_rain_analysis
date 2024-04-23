# Compare the radar  and CPM fit (for today)
from R_python import gev_r
import CPMlib
import CPM_rainlib
import xarray
import matplotlib.pyplot as plt
import commonLib

save_file_cpm=CPM_rainlib.datadir/'fits/cpm_fit.nc'
fit_cpm = xarray.load_dataset(save_file_cpm)
save_radar_file=CPMlib.datadir/'fits/radar_fit.nc'

fit_radar = xarray.load_dataset(save_radar_file)
obs_cet=commonLib.read_cet() # read in the obs CET
obs_cet_jja = obs_cet.sel(time=(obs_cet.time.dt.season == 'JJA'))
t_today = obs_cet_jja.sel(time=slice('2010','2020')).mean()
params_today = gev_r.param_at_cov(fit_cpm.Parameters, t_today)

fig=plt.figure(num='radar_CPM_dist',clear=True)
ax=fig.add_subplot(111)
ax.errorbar(fit_radar.Parameters.sel(parameter="location"),fit_radar.Parameters.sel(parameter="scale"),
            xerr=fit_radar.StdErr.sel(parameter="location"),yerr=fit_radar.StdErr.sel(parameter="scale"),
            linestyle='none',label='RADAR')
ax.errorbar(params_today.sel(parameter="location"),params_today.sel(parameter="scale"),
            xerr=fit_cpm.StdErr.sel(parameter="location"),yerr=fit_cpm.StdErr.sel(parameter="scale"),
            linestyle='none',label='CPM')
ax.set_xlabel("location")
ax.set_ylabel("scale")
ax.legend()
fig.show()

