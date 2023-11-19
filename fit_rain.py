# fit the CPM rainfall data.
import gev_r
import commonLib
import matplotlib.pyplot as plt
import numpy as np
import xarray
import CPMlib
dataset = xarray.load_dataset(CPMlib.CPM_dir/"CPM_all_events.nc") # load the processed events
dataset= dataset.stack(idx=['ensemble_member','EventTime']).dropna('idx')

radar_datset= xarray.load_dataset(CPMlib.datadir/"radar_events.nc")
recreate=False # set True to recreate the fits






# first get in CET -- our covariate

obs_cet=commonLib.read_cet() # read in the obs CET
obs_cet_jja = obs_cet.sel(time=(obs_cet.time.dt.season == 'JJA'))
# compute the fits!


t_today = obs_cet_jja.sel(time=slice('2010','2020')).mean()
#dd=dataset.sel(ensemble_member=1); dim='EventTime' # test case.
dd=dataset ; dim='idx'
ht=dd.height


ln_area = np.log10(dd.count_cells*4.4**2).rename('log10_area')
precip = dd.max_precip

ln_radar_area  = (np.log10(radar_datset.count_cells * 25)).rename('log10_area')
fit_radar = gev_r.xarray_gev(radar_datset.max_precip,dim='EventTime',verbose=True,
                             file=CPMlib.fit_dir/'radar_fit_no_cov.nc',recreate_fit=recreate)

fit_radar = gev_r.xarray_gev(radar_datset.max_precip,dim='EventTime',verbose=True,
                             file=CPMlib.fit_dir/'radar_fit_no_cov.nc',recreate_fit=recreate)
fit_radar_lnarea = gev_r.xarray_gev(radar_datset.max_precip,cov=[ln_radar_area],dim='EventTime',verbose=True,
                             file=CPMlib.fit_dir/'radar_fit_lnarea.nc',recreate_fit=recreate)
fit = gev_r.xarray_gev(precip,dim=dim,verbose=True,file=CPMlib.fit_dir/'fit_no_cov.nc',recreate_fit=recreate)
fit_cet = gev_r.xarray_gev(precip,
                       cov=[dd.CET],dim=dim,verbose=True,file=CPMlib.fit_dir/'fit_cet.nc',recreate_fit=recreate)
fit_cet_cells = gev_r.xarray_gev(precip,
                       cov=[dd.CET,dd.count_cells],
                         dim=dim,verbose=True,file=CPMlib.fit_dir/'fit_cet_cells.nc',recreate_fit=recreate)
fit_cet_ln_cells = gev_r.xarray_gev(precip,
                       cov=[dd.CET,ln_area],
                         dim=dim,verbose=True,file=CPMlib.fit_dir/'fit_cet_lnarea.nc',recreate_fit=recreate)

fit_cells = gev_r.xarray_gev(precip,
                       cov=[dd.count_cells],
                         dim=dim,verbose=True,file=CPMlib.fit_dir/'fit_cells.nc',recreate_fit=recreate)

fit_ht = gev_r.xarray_gev(precip,cov=[ht],dim=dim,verbose=True,file=CPMlib.fit_dir/'fit_ht.nc',recreate_fit=recreate)
fit_cet_cells_ht = gev_r.xarray_gev(precip,cov=[dd.CET,dd.count_cells,ht],dim=dim,verbose=True,
                                    file=CPMlib.fit_dir/'fit_cet_cells_ht.nc',recreate_fit=recreate)
fit_cet_ln_cells_ht = gev_r.xarray_gev(precip,
                                      cov=[dd.CET,ln_area,ht],
                                      dim=dim,verbose=True,file=CPMlib.fit_dir/'fit_cet_lnarea_ht.nc',recreate_fit=recreate)

fig=plt.figure(num='AIC',clear=True)
for f,title,marker in zip(
            [fit,fit_cet,fit_cells,fit_cet_cells,fit_cet_ln_cells,fit_ht,fit_cet_cells_ht,fit_cet_ln_cells_ht],
                   ['None','CET','cells','CET & cells','CET & ln Cells','ht','CET, Cells & Ht','C lnCe ht'],
                          ['x','+','o']*10):
    if 'ht' in title:
        alpha=0.5
        linewidth=1
    else:
        alpha=1
        linewidth=2
    f.AIC.plot(label=title,linewidth=linewidth,marker=marker,alpha=alpha)
fig.legend(ncol=3)
fig.show()
commonLib.saveFig(fig)
# plot the raw and adjusted CET sensitivity.


# and then express as fraction but only for two models -- cet & cet_ln_area
delta_t = float(t_today - obs_cet_jja.sel(time=slice('1850', '1899')).mean())

fig, ax = plt.subplots(nrows=1, ncols=1, clear=True, num='scatter_CPM_changes', figsize=[11, 8])
for fit,title,color in zip([fit_cet,fit_cet_ln_cells],['CET','CET & log10(area)'],['red','purple']):
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


