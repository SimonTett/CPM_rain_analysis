# Plot filtered and  raw CPM data and compare with Radar data.
# Will consider the fit with CET^2 and ln area in both cases
import logging
import typing

import scipy.stats

import CPM_rainlib
import xarray
import commonLib
import numpy as np
import matplotlib.pyplot as plt
from R_python import gev_r



def comp_today(fit: xarray.Dataset, t_today: float, log10_area: float):
    result = [fit.Parameters.sel(parameter='shape')] # start with shape
    for k in ['location', 'scale']:
        r = fit.Parameters.sel(parameter=k)
        try:
            r = r + t_today * fit.Parameters.sel(parameter=f"D{k}_CET")
            r = r + t_today ** 2 * fit.Parameters.sel(parameter=f"D{k}_CET_sqr")
        except KeyError:
            logging.warning(f"{k}:No DCET so ignoring")
        r = r + log10_area * fit.Parameters.sel(parameter=f"D{k}_log10_area")
        r = r.assign_coords(parameter=k)  #  rename
        result.append(r)
    result=xarray.concat(result,'parameter')
    return result

def gev_isf(param:xarray.DataArray,
            probs:typing.List,
            name:str,
            dist:typing.Optional[typing.Callable]=None) -> xarray.DataArray:
    if dist is None:
        dist = scipy.stats.genextreme

    fd = dist(param.sel(parameter='shape'), loc=param.sel(parameter='location'), scale=param.sel(parameter='location'))
    values=[]
    for p in probs:
        r=fd.isf(p).reshape(-1,1)

        r=xarray.DataArray(data=r,dims=['quantv','return_period'],
                           coords=dict(quantv=param.quantv,return_period=[1./p]))
        values.append(r)
    result = xarray.concat(values,dim="return_period").rename(name)
    return result

file = 'fit_cet_sqr_lnarea.nc'
fits = dict()
for dir, name in zip(['CPM_scotland', 'CPM_scotland_filter'], ['Raw CPM', 'Filtered CPM']):
    path = CPM_rainlib.dataDir / dir / "fits" / file
    fits[name] = xarray.load_dataset(path)
fits['radar_5km'] = xarray.load_dataset(CPM_rainlib.dataDir / 'radar/fits/fit_area_summary_5km_1hr_scotland.nc')
fits['radar_1km_c5'] = xarray.load_dataset(CPM_rainlib.dataDir / 'radar/fits/fit_area_1km_c5.nc')
fits['radar_1km_c4'] = xarray.load_dataset(CPM_rainlib.dataDir / 'radar/fits/fit_area_1km_c4.nc')
# want values for today. So read in the obs summer CET and then compute the avg for 2005-2023
obs_cet = commonLib.read_cet()  # read in the obs CET
obs_cet_jja = obs_cet.sel(time=(obs_cet.time.dt.season == 'JJA'))
# compute the fits!
t_today = obs_cet_jja.sel(time=slice('2005', '2023')).mean()

params = {k: comp_today(v, t_today, np.log10(500.)) for k, v in fits.items()}
params_p1k = {k: comp_today(v, t_today + 1, np.log10(500.)) for k, v in fits.items() if 'radar' not in k}
params_p2k = {k: comp_today(v, t_today + 2, np.log10(500.)) for k, v in fits.items() if 'radar' not in k}
## now to make the plot!
fig, ((ax, ax_scale), (ax_int_today, ax_int_change)) = plt.subplots(nrows=2, ncols=2, num='scale_locn', figsize=[8, 7],
                                                                    clear=True)
rtn_prd=100.
for k, v in params.items():
    ax.scatter(v.sel(parameter='location'), v.sel(parameter='scale'), label=k, linewidths=1)
    isf = gev_isf(v,[1/rtn_prd],'R1hrx (mm/hr)')
    isf.plot(ax=ax_int_today,x='quantv',label=k)
    if k in params_p1k.keys():
        p1k = params_p1k[k]
        p2k = params_p2k[k]
        r1 = ( p1k/ v)-1
        r2 = ((p2k/ v)-1)/2
        ax_scale.scatter(r1.sel(parameter='location'), r1.sel(parameter='scale'), label="+1K " + k, linewidths=1, marker='x')
        ax_scale.scatter(r2.sel(parameter='location'), r2.sel(parameter='scale'), label="+2K " + k, linewidths=1, marker='h')

        # and now the intensity ratios
        isf1k= gev_isf(p1k,[1/rtn_prd],'R1hrx (mm/hr)')
        isf2k = gev_isf(p2k, [1 / rtn_prd], 'R1hrx (mm/hr)')
        ri1 = (isf1k/isf)-1
        ri2 = ((isf2k / isf) - 1)/2
        ri1.plot(ax=ax_int_change, x='quantv', label="+1K " + k)
        ri2.plot(ax=ax_int_change, x='quantv', label="+2K " + k)
ax.set_xlabel('location (mm/hr)')
ax.set_ylabel('scale (mm/hr)')
ax_scale.set_xlabel('Location frac. change/K')
ax_scale.set_ylabel('Scale frac. change/K')
ax_scale.legend()
ax.legend()
ax_int_today.set_yscale('log')
ax_int_today.legend()

ax_int_change.set_ylabel("Intensity frac change/K")
ax_int_change.legend()
# now to compute scales from the models.
fig.tight_layout()
fig.show()
commonLib.saveFig(fig)
