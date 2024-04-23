# do straight fit
#
# ts to 1980-1989 & 2070-2080 CPM event data.
# and then plot them. This shows that they are different (or not)
import xarray
import CPMlib
from R_python import gev_r
import matplotlib.pyplot as plt
import commonLib

path = CPMlib.CPM_filt_dir/"CPM_filter_all_events.nc"
ds = xarray.load_dataset(path)
# using ds.t to select the data.
ds = ds.stack(indx=['ensemble_member','EventTime']).dropna('indx')
season=ds.t.isel(quantv=0).dt.season.drop('quantv')
L=(season== 'JJA')
ds=ds.sel(indx=L)
yr = ds.isel(quantv=0).t.dt.year.drop('quantv')
L = (1981 <= yr) & (yr <= 2000)
ds1980_90s = ds.sel(indx=L)
L = (2061 <= yr) & (yr <= 2080)
ds2060_70s = ds.sel(indx=L)
fit_2060_70s = gev_r.xarray_gev(ds2060_70s.max_precip, dim='indx', name='Filt 2061-2080')
fit_1980_90s = gev_r.xarray_gev(ds1980_90s.max_precip, dim='indx', name='Filt 1981-2000')
delta_cet=ds2060_70s.CET.mean() - ds1980_90s.CET.mean()
per_change = 100*(fit_2060_70s.Parameters/fit_1980_90s.Parameters-1)/delta_cet
err_change = 2*100*(fit_2060_70s.StdErr+fit_1980_90s.StdErr)/(fit_1980_90s.Parameters*delta_cet)

## plot
label = commonLib.plotLabel()
fig,axis = plt.subplots(nrows=1,ncols=3,clear=True,
                        num='fit_ratio',figsize=[9,4],
                        layout='tight')

for p,ax in zip(['location','scale'],axis):
    #per_change.sel(parameter=p).plot(ax=ax)
    f=per_change.sel(parameter=p)
    e=err_change.sel(parameter=p)
    ax.errorbar(f.quantv,f,e,capsize=4)
    ax.set_title(p.capitalize())
    ax.set_ylabel("% Change/CET")
    ax.set_xlabel("Event Quantile")
    ax.axhline(f.mean(),color='black',linestyle='dotted')
    label.plot(ax)
# plot the shapes.
p='shape'
ax=axis[-1]
for name,fit in zip(['1981-2000','2061-2080'],
                    [fit_1980_90s,fit_2060_70s]):
    f=fit.Parameters.sel(parameter=p)
    err = fit.StdErr.sel(parameter=p)*2 # 2 sigma
    ax.errorbar(f.quantv,f,err,label=name,capsize=4)
label.plot(ax)
ax.set_title("Shape")
ax.set_xlabel("Event Quantile")
ax.set_ylabel("Shape")
ax.legend()
ax.axhline(0,linestyle='dashed',color='black')
fig.show()
commonLib.saveFig(fig)
