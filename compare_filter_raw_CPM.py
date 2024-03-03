# Plot filtered and  raw CPM data and compare with Radar data.
# Will consider the fit with CET^2 and ln area in both cases
# will also compare with expected CC
import logging
import math
import typing

import scipy.stats

import CPM_rainlib
import CPMlib
import xarray
import commonLib
import numpy as np
import matplotlib.pyplot as plt
from R_python import gev_r

def qsat(temperature):
    """
    Saturated humidity from temperature.
    :param temperature: temperature (in degrees c)
    :return: saturated humidity vp
    """
    es = 6.112 * np.exp(17.6 * temperature / (temperature + 243.5))
    return es

def comp_today(fit: xarray.Dataset, t_today: float,
               log10_area: typing.Optional[float]=None,
               hour:typing.Optional[float] = None,
               height:typing.Optional[float] = None):
    if log10_area is None:
        log10_area = math.log10(150.)
    if hour is None:
        hour = 13.0
    if height is None:
        height = 100.
    result = [fit.Parameters.sel(parameter='shape')] # start with shape
    param=dict(log10_area=log10_area,hour=hour,hour_sqr=hour**2,height=height,
                   CET=t_today,CET_sqr=t_today**2)
    for k in ['location', 'scale']:
        r = fit.Parameters.sel(parameter=k)
        for pname,v in param.items():
            p = f"D{k}_{pname}"
            try:
                r = r + v * fit.Parameters.sel(parameter=p)
            except KeyError:
                logging.warning(f"{k} missing {p} so ignoring")
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

    fd = dist(param.sel(parameter='shape'), loc=param.sel(parameter='location'), scale=param.sel(parameter='scale'))
    values=[]
    for p in probs:
        r=fd.isf(p).reshape(-1,1)

        r=xarray.DataArray(data=r,dims=['quantv','return_period'],
                           coords=dict(quantv=param.quantv,return_period=[1./p]))
        values.append(r)
    result = xarray.concat(values,dim="return_period").rename(name)
    return result

def gev_support(param:xarray.DataArray,
            dist:typing.Optional[typing.Callable]=None) -> xarray.DataArray:
    if dist is None:
        dist = scipy.stats.genextreme

    fd = dist(param.sel(parameter='shape'), loc=param.sel(parameter='location'), scale=param.sel(parameter='scale'))
    support=fd.support()
    result=xarray.DataArray(data=np.array(support),coords=dict(support=['min','max'],quantv=param.quantv)).rename(param.name)
    return result

# read in the regional temp and compute summer qstat then compute seasonal means
def comp_summer_mn(da:xarray.DataArray) -> xarray.DataArray:
    seas = da.resample(time='QS-DEC').mean()
    seas = seas.sel(time=(seas.time.dt.season=='JJA'))
    return seas
sim_reg_tas = xarray.load_dataset(CPMlib.CPM_dir/'stonehaven_reg_tas.nc').tas
sim_reg_es = comp_summer_mn(qsat(sim_reg_tas))
sim_reg_tas = comp_summer_mn(sim_reg_tas)
sim_cet=xarray.load_dataset(CPMlib.CPM_dir/'cet_tas.nc').tas
sim_cet = comp_summer_mn(sim_cet)
import statsmodels.api as sm
def regress(temp:xarray.DataArray,es:xarray.DataArray):
    # compute fit using statsmodels.
    X = temp.values.flatten()
    X = sm.add_constant(X)
    Y = es.values.flatten()
    model = sm.OLS(Y, X)
    fit = model.fit()
    return fit

# now compute the regression coefficient.
fit_cet = regress(sim_cet, sim_reg_es)
fit_reg = regress(sim_reg_tas, sim_reg_es)
X = sim_cet.values.flatten()
X = sm.add_constant(X)
model_cet_reg = sm.OLS(sim_reg_tas.values.flatten(),X)
fit_cet_reg = model_cet_reg.fit()





file='fit_cet_lnarea.nc'
fits = dict()
for dir, name in zip(['CPM_scotland', 'CPM_scotland_filter'], ['Raw CPM', 'Filt CPM']):
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
t_today_reg = fit_cet_reg.params[0]+t_today*fit_cet_reg.params[1] # best est for todays temperature in region.
today_es = float(fit_cet.get_prediction([1, t_today]).predicted_mean)
today_reg_es = float(fit_reg.get_prediction([1, t_today_reg]).predicted_mean)
scale_cet=float(fit_cet.params[1]/today_es)*100 # how much as a fraction would expect ES to increase for a 1 k increase in CET
scale_reg=float(fit_reg.params[1]/today_reg_es)*100 # how much as a fraction would expect ES to increase for a 1 k increase in CET
print(f"CET scale:{scale_cet:3.1f} %/K   Reg scale:{scale_reg:3.1f} %/K")

## plot scatters between CET and Regional Temp and CET & regional SVP (hPa)
fig_scatter,axis = plt.subplots(nrows=1,ncols=2,num='cet_scatter',clear=True,figsize=[8,3],layout='constrained')
axis[0].scatter(sim_cet,sim_reg_tas)
axis[1].scatter(sim_cet, sim_reg_es)
# plot best fit lines...
npts = 50
for fit,ax in zip([fit_cet_reg,fit_cet],axis):
    x=np.linspace(12,24,num=npts)
    best_ft=fit.params.dot(np.array([np.repeat(1,npts),x]))
    ax.plot(x,best_ft,color='black',linewidth=3)

# plot "today"
axis[0].plot(t_today,t_today_reg,marker='h',ms=15,color='red')
axis[1].plot(t_today,today_es,marker='h',ms=15,color='red')

for a in axis:
    a.set_xlabel(r"CET ($^\circ$C)")
axis[0].set_ylabel(r"Reg. Temp ($^\circ$C)")
axis[1].set_ylabel("Sat Vap. P (hPa)")
axis[0].set_title("CET vs Stonehaven Reg. Temp")
axis[1].set_title("CET vs Stonehaven Reg. Sat VP")
fig_scatter.show()
commonLib.saveFig(fig_scatter)
##
params = {k: comp_today(v, t_today) for k, v in fits.items()}
params_p1k = {k: comp_today(v, t_today + 1) for k, v in fits.items() if 'radar' not in k}
#params_p2k = {k: comp_today(v, t_today + 2) for k, v in fits.items() if 'radar' not in k}
## now to make the plot!
# set up styles for different lines etc.
plot_styles={'Raw CPM':dict(color='royalblue',marker='d'),
             'Filt CPM':dict(color='darkblue',marker='d'),
             'radar_5km':dict(color='orange',marker='o'),
             'radar_1km_c5':dict(color='brown',marker='h'),
             'radar_1km_c4':dict(color='red',marker='h')}
fig, ((ax, ax_scale), (ax_int_today, ax_int_change)) = plt.subplots(nrows=2, ncols=2, num='scale_locn', figsize=[8, 7],
                                                                    clear=True,layout="constrained")

label = commonLib.plotLabel()

# add on a scale param plot.
fig_shape,ax_shape = plt.subplots(nrows=1,ncols=1,clear=True,num='shape_plot',figsize=(8,3.5),layout="constrained")
rtn_prd=10.
lw=2
for k, v in params.items():
    pstyle = plot_styles[k]
    err_style = pstyle.copy()
    err_style.update(marker=None,linestyle='None')
    #pstyle.update(s=(v.quantv*40).values,linestyle='solid')
    print(k,gev_support(v).sel(quantv=0.5).values)
    size=dict(s=(v.quantv[::2]*50).values+5)
    ax.scatter(v.sel(parameter='location')[::2], v.sel(parameter='scale')[::2], linewidth=lw,**err_style,**size)
    ax.plot(v.sel(parameter='location'), v.sel(parameter='scale'), label=k, linewidth=lw, **pstyle)
    shape = -fits[k].Parameters.sel(parameter='shape')# negate shape to have consitency with Cole book.
    se=fits[k].StdErr.sel(parameter='shape')
    ax_shape.errorbar(shape.quantv[::4],shape[::4],yerr=2*se[::4],capthick=2,capsize=4,**err_style)
    ax_shape.plot(shape.quantv, shape, **pstyle,markevery=2,label=k)
    #shape.plot(label=k,linewidth=lw,**pstyle,ax=ax_shape)
    # and plot (shaded) the uncertainty range


    isf = gev_isf(v,[1/rtn_prd],'R1hrx (mm/hr)')
    isf.plot(ax=ax_int_today,x='quantv',label=k,**pstyle,markevery=2)
    if k in params_p1k.keys():
        p1k = params_p1k[k]
        #p2k = params_p2k[k]
        r1 = ( p1k/ v)-1
        r1 *= 100 # convert to %
        #r2 = ((p2k/ v)-1)/2
        style = pstyle.copy()
        style.update(marker='h')
        ax_scale.plot(r1.sel(parameter='location'), r1.sel(parameter='scale'), label="+1K " + k, linewidth=lw, **pstyle,markevery=2)
        ax_scale.scatter(r1.sel(parameter='location')[::2], r1.sel(parameter='scale')[::2],  linewidth=lw, **err_style,**size)
        #ax_scale.plot(r2.sel(parameter='location'), r2.sel(parameter='scale'), label="+2K " + k, linewidth=lw, **style)

        # and now the intensity ratios
        isf1k= gev_isf(p1k,[1/rtn_prd],'R1hrx (mm/hr)')
        #isf2k = gev_isf(p2k, [1 / rtn_prd], 'R1hrx (mm/hr)')
        ri1 = (isf1k/isf)-1
        ri1 *= 100 # convert to %
        #ri2 = ((isf2k / isf) - 1)/2
        ri1.plot(ax=ax_int_change, x='quantv', label="+1K " + k,**pstyle,markevery=2)
        #ri2.plot(ax=ax_int_change, x='quantv', label="+2K " + k,**style)
ax.set_xlabel('location (mm/hr)')
ax.set_ylabel('scale (mm/hr)')
ax_scale.set_xlabel('Location % change/K')
ax_scale.set_ylabel('Scale % change/K')

# add on the CC lines
style=dict(linestyle='dotted',color='black')
ax_scale.axhline(scale_cet, **style)
ax_scale.axvline(scale_cet, **style)
ax_int_change.axhline(scale_cet,**style)

ax_int_today.set_yscale('log')
tv=[5,10,20,30,40,50,70]
ax_int_today.set_yticks(tv,labels=[str(t) for t in tv])
ax_int_change.set_ylabel("Intensity % change/K")

ax_shape.set_ylabel('Shape')
# titles
for a,title in zip([ax,ax_int_today,ax_scale,ax_int_change,ax_shape],['GEV params','Intensity','%. Parameter change','% Intensity change',
                                                                      'Shape']):
    a.set_title(title)
    label.plot(a)
# legends
ax_int_change.legend()
ax.legend(ncols=2)
ax_shape.legend(ncols=5)
for a in (ax_shape,ax_int_today,ax_int_change):
    a.set_xlabel("Event Quantile")
# add on zero line for shape plot
ax_shape.axhline(0,**style)


fig.show()
fig_shape.show()
commonLib.saveFig(fig)
commonLib.saveFig(fig_shape)