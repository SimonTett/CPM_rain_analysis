# fit the CPM rainfall data.
from R_python import gev_r
import commonLib
import matplotlib.pyplot as plt
import numpy as np
import xarray
import CPMlib
import CPM_rainlib
import pandas as pd
filtered=True # Set true to use filtered rather than raw data.
recreate=False # set True to recreate the fits
verbose=True
if filtered:
    event_path = CPM_rainlib.dataDir/'CPM_scotland_filter'/"CPM_filter_all_events.nc"
    fit_dir = CPM_rainlib.dataDir/'CPM_scotland_filter'/"fits"
    base_img_name='filter'
else:
    event_path = CPMlib.CPM_dir/"CPM_all_events.nc"
    fit_dir = CPM_rainlib.dataDir/'CPM_scotland'/"fits"
    base_img_name='raw'

dataset = xarray.load_dataset(event_path) # load the processed events
dataset= dataset.stack(idx=['ensemble_member','EventTime']).dropna('idx')





# first get in CET -- our covariate

obs_cet=commonLib.read_cet() # read in the obs CET
obs_cet_jja = obs_cet.sel(time=(obs_cet.time.dt.season == 'JJA'))
# compute the fits!


t_today = obs_cet_jja.sel(time=slice('2010','2020')).mean()
#dd=dataset.sel(ensemble_member=1); dim='EventTime' # test case.
dd=dataset ; dim='idx'
ht=dd.height


ln_area = np.log10(dd.count_cells*4.4**2).rename('log10_area')
area = (dd.count_cells*4.4**2).rename('area')
hour = dd.t.dt.hour.rename('hour')
hour_sqr = (hour**2).rename('hour_sqr')
cet_sqr = (dd.CET**2).rename("CET_sqr")
precip = dd.max_precip

fit = gev_r.xarray_gev(precip, dim=dim, file=fit_dir / 'fit_no_cov.nc', recreate_fit=recreate, verbose=verbose, name='None')
fit_cet = gev_r.xarray_gev(precip, cov=[dd.CET], dim=dim, file=fit_dir / 'fit_cet.nc', recreate_fit=recreate,
                           verbose=verbose, name='C')

fit_cet_sqr = gev_r.xarray_gev(precip, cov=[dd.CET, cet_sqr], dim=dim, file=fit_dir / 'fit_cet_sqr.nc',
                               recreate_fit=recreate, verbose=verbose, name='C+C^2')

fit_cet_area = gev_r.xarray_gev(precip, cov=[dd.CET, area], dim=dim, file=fit_dir / 'fit_cet_area.nc',
                                 recreate_fit=recreate, verbose=verbose, name='C+A')
fit_cet_ln_area = gev_r.xarray_gev(precip, cov=[dd.CET, ln_area], dim=dim, file=fit_dir / 'fit_cet_lnarea.nc',
                                    recreate_fit=recreate, verbose=verbose, name='C+lnA')
fit_ln_area = gev_r.xarray_gev(precip, cov=[ln_area], dim=dim, file=fit_dir / 'fit_lnarea.nc',
                                    recreate_fit=recreate, verbose=verbose, name='lnA')
fit_area = gev_r.xarray_gev(precip, cov=[area], dim=dim, file=fit_dir / 'fit_area.nc',
                                    recreate_fit=recreate, verbose=verbose, name='A')

fit_cet_sqr_ln_area = gev_r.xarray_gev(precip, cov=[dd.CET, cet_sqr, ln_area], dim=dim,
                                        file=fit_dir / 'fit_cet_sqr_lnarea.nc', recreate_fit=recreate, verbose=verbose,
                                        name='C+C^2+lnA')

fit_cet_area_z = gev_r.xarray_gev(precip, cov=[dd.CET, area, ht], dim=dim,
                                    file=fit_dir / 'fit_cet_area_z.nc', recreate_fit=recreate, verbose=verbose, name='C+A+z')
fit_cet_ln_area_z = gev_r.xarray_gev(precip, cov=[dd.CET, ln_area, ht], dim=dim,
                                       file=fit_dir / 'fit_cet_lnarea_z.nc', recreate_fit=recreate, verbose=verbose,
                                       name='C+lnA+z')

# try using hour of day.

fit_cet_sqr_ln_area_hr_sqr = gev_r.xarray_gev(precip, cov=[dd.CET, cet_sqr, ln_area, hour, hour_sqr], dim=dim,
                                               file=fit_dir / 'fit_cet_sqr_lnarea_hr_sqr.nc', recreate_fit=recreate,
                                               verbose=verbose,name='C+C^2+lnA+hr+hr^2')

fit_cet_sqr_ln_cells_hr_sqr_ht = gev_r.xarray_gev(precip, cov=[dd.CET, cet_sqr, ln_area, hour, hour_sqr, ht], dim=dim,
                                                  file=fit_dir / 'fit_cet_sqr_lnarea_hr_sqr_ht_z.nc',
                                                  recreate_fit=recreate, verbose=verbose,name='C+C^2+lnA+hr+hr^2+z')
## plot everything
fig,ax=plt.subplots(nrows=1,ncols=1,num=f'{base_img_name}_AIC',clear=True,figsize=(5,3),layout='constrained')
AIC=dict()

for f,marker in zip(
            [fit,
             fit_cet,
             fit_area,fit_ln_area,
             fit_cet_ln_area,

             #fit_ht,
             #fit_cet_cells_ht,
             # fit_cet_ln_cells_ht,
             fit_cet_sqr_ln_area,
            # commented out these more complex fits till figure out what to do with them when comparing with obs.
             #fit_cet_ln_cells_ht_hr,fit_cet_sqr_ln_cells_hr,fit_cet_sqr_ln_cells_hr_sqr,fit_cet_sqr_ln_cells_hr_sqr_ht
             ],

                          ['x','+','o']*20):
    title = f.attrs.get('name')
    if 'z' in title:
        alpha=0.5
        linewidth=1
    else:
        alpha=1
        linewidth=2
    (f.AIC/1000).plot(label="$"+title+"$",linewidth=linewidth,marker=marker,alpha=alpha,ax=ax)
    ax.set_title(f"{base_img_name} AIC")
    ax.set_ylabel("AIC/1000")
    AIC[title]=f.AIC
ax.legend(ncol=2)

fig.show()
commonLib.saveFig(fig)
df_AIC = pd.DataFrame(AIC,index=f.quantv).T
print(df_AIC.idxmin())
# plot the raw and adjusted CET sensitivity.


# and then express as fraction but only for two models -- cet & cet_ln_area
delta_t = float(t_today - obs_cet_jja.sel(time=slice('1850', '1899')).mean())

fig, ax = plt.subplots(nrows=1, ncols=1, clear=True, num=f'{base_img_name}_scatter_CPM_changes', figsize=[11, 8])
for fit,title,color in zip([fit_cet,fit_cet_ln_area],['CET','CET & log10(area)'],['red','purple']):
    ref = fit.Parameters.sel(parameter=['location','scale'])
    D=fit.Parameters.sel(parameter=['Dlocation_CET','Dscale_CET']).assign_coords(parameter=['location','scale'])
    Derr=fit.StdErr.sel(parameter=['Dlocation_CET','Dscale_CET']).assign_coords(parameter=['location','scale'])
    params_today = ref + D*float(t_today)
    if 'area' in title:
        D_area= fit.Parameters.sel(parameter=['Dlocation_log10_area','Dscale_log10_area']).assign_coords(parameter=['location','scale'])
        params_today += D_area*float(ln_area.mean())
    ratio_Dparam = D/params_today
    ratio_Dparam_err = Derr/params_today# think about this
    # plot them
    ax.errorbar(ratio_Dparam.sel(parameter='location'),ratio_Dparam.sel(parameter='scale'),
                xerr=ratio_Dparam_err.sel(parameter='location'), yerr=ratio_Dparam_err.sel(parameter='scale'),
                color=color,label=title)
    # for q in range(0,len(fit.quantv)):
    #         c=rat[q,:]
    #         ax.scatter(c[0],c[1],s=20)
    #         ax.annotate(f"{float(fit.quantv[q]):3.2f}",(c[0],c[1]),xytext=(1,0),textcoords="offset fontsize")


ax.set_xlabel("Dlocation/location")
ax.set_ylabel("Dscale/scale")
ax.set_title("DLocation vs DScale")
cc=0.06*delta_t
ax.axhline(cc,linestyle='dotted',linewidth=4)
ax.axvline(cc,linestyle='dotted',linewidth=4)
ax.legend()
fig.tight_layout()
fig.show()
commonLib.saveFig(fig)


