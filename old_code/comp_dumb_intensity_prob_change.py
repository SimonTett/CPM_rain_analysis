# Dumb calculation of changing intensity and probabilty ratio.
# uses filtered and raw CPM at carmont point, fits GEV with covariate
import matplotlib.pyplot as plt
import numpy as np

import CPM_rainlib
import CPMlib
import xarray

import commonLib
from R_python import gev_r
import typing
import math
import logging
import scipy
import matplotlib.colors as mcolors
import matplotlib.cm as mcm
import cartopy.crs as ccrs
import CPM_rainlib

def comp_params(fit: xarray.Dataset,
                t_today: float,
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

def gev_isf(param: xarray.DataArray,
            probs: np.ndarray,
            name: typing.Optional[str] = None,
            dist: typing.Optional[typing.Callable] = None) -> xarray.DataArray:
    if dist is None:
        dist = scipy.stats.genextreme

    fd = dist(param.sel(parameter='shape'), loc=param.sel(parameter='location'), scale=param.sel(parameter='scale'))
    r = fd.isf(probs)
    result = xarray.DataArray(data=r, dims=['return_period'], coords=dict( return_period=1. / probs))
    if name:
        result = result.rename(name)
    return result
##
def gev_sf(param: xarray.DataArray,
            values: np.ndarray,
            name: typing.Optional[str] = None,
            dist: typing.Optional[typing.Callable] = None) -> xarray.DataArray:
    if dist is None:
        dist = scipy.stats.genextreme
    params = [np.expand_dims(param.sel(parameter=k),-1) for k in ['shape','location','scale']]
    fd = dist(*params)
    r = fd.sf(values)
    # need to deal with possible dimensions.
    dims = list(param.dims)
    dims.remove('parameter')
    coords = {c:param[c] for c in dims}
    dims.append('values')
    coords.update(values=values.squeeze())

    result = xarray.DataArray(data=r,  coords=coords)
    if name:
        result = result.rename(name)
    return result
##
fit_dir = CPM_rainlib.dataDir/'CPM_scotland_filter'/"fits"
obs_cet = commonLib.read_cet()  # read in the obs CET
obs_cet_jja = obs_cet.where(obs_cet.time.dt.season == 'JJA',drop=True)
temps=dict()
temps["PI"] = obs_cet_jja.sel(time=slice('1850','1899')).mean()
temps["1980-1989"] = obs_cet_jja.sel(time=slice('1980','1989')).mean()
temps["2012-2021"] = obs_cet_jja.sel(time=slice('2012', '2021')).mean()
temps["PI+2K"] = 2 * 0.94 + temps["PI"]
plot_styles = { # styles for plotting
    '1980-1989':dict(color='royalblue',linestyle='solid'),
    '2012-2021': dict(color='blue', linestyle='solid'),
    'PI+2K': dict(color='red', linestyle='solid')}
# compute the fits!
ds=xarray.open_mfdataset(CPMlib.CPM_filt_dir.glob("**/CPM*11_30_23.nc"),parallel=True)
L=ds.time.dt.season == 'JJA'
rgn = CPMlib.stonehaven_rgn
maxRain = ds.seasonalMax.sel(**rgn)
carmont = ds.seasonalMax.sel(**CPMlib.carmont,method='nearest').where(L,drop=True).load()
rgn_all=dict(longitude=slice(357.5,361.0),latitude=slice(1.5,7.5)) # region for which extraction was done.

topog_min,topog_mn = CPMlib.get_topog(rgn_all,rgn,lat=maxRain.grid_latitude,lon=maxRain.grid_longitude)
maxRain =maxRain.where(L,drop=True).load()
cpm_cet=xarray.load_dataset(CPMlib.CPM_dir/"cet_tas.nc")
cpm_cet=cpm_cet.tas.resample(time='QS-DEC').mean() # resample to seasonal means
cpm_cet_jja = cpm_cet.where(cpm_cet.time.dt.season=='JJA',drop=True) # and pull out summer.
stack_dim=dict(t_e=['time',"ensemble_member"])
fit=gev_r.xarray_gev(carmont.stack(**stack_dim), cov=[cpm_cet_jja.rename('CET').stack(**stack_dim)],
                     dim='t_e',name='Carmont_C',file=fit_dir/'carmont_fit_cet.nc')
fit_extra=gev_r.xarray_gev(maxRain.stack(**stack_dim),
                           cov=[cpm_cet_jja.rename('CET').stack(**stack_dim)],
                           dim='t_e',name='Rgn_c',file=fit_dir/'rgn_fit_cet.nc')
fits=dict()
fits_extra=dict()

dists=dict()
dists_extra=dict()
probs=dict()
probs_extra =dict()
delta_intens= dict()
delta_intens_extra=dict()
pr=dict()
for key,t in temps.items():
    fits[key] = comp_params(fit, t)
    dists[key]= gev_r.xarray_gev_isf(fits[key], 1.0 / np.geomspace(2, 100, 200))
    probs[key] = gev_r.xarray_gev_sf(fits[key],np.linspace(10,30,num=200))
    # extra stuff!
    fits_extra[key] = comp_params(fit_extra,t)
    dists_extra[key] = gev_r.xarray_gev_isf(fits_extra[key],1.0 / np.geomspace(2, 100, 200))
    probs_extra[key] = gev_r.xarray_gev_sf(fits_extra[key], np.linspace(10, 30, num=200))
for key, t in temps.items():
    if key != 'PI':
        delta_intens[key] = (dists[key]/dists['PI']-1)*100.
        pr[key] = (probs[key]/probs['PI'])*100.
        # extra stuff -- whole domain.
        delta_intens_extra[key] = (dists_extra[key]/dists_extra['PI']-1)*100.
        probs_extra[key] = (probs_extra[key]/probs_extra['PI'])*100.

## plot the Carmont values.
msk= (topog_mn < 300) & (topog_min > 0)
key='2012-2021'
fig_carmont_int,ax_c_int = plt.subplots(nrows=1,ncols=1,figsize=(8,5),
                                 clear=True,num='Dumb_carmont_intensity',layout='constrained')
dists[key].plot(ax=ax_c_int,x='return_period',**plot_styles[key])
zz=dists_extra[key].where(msk).stack(idx=['grid_longitude', 'grid_latitude']).\
        quantile([0.05,0.5, 0.95], dim='idx')
lstyle=plot_styles[key].copy()
lstyle.update(linestyle='dashdot')
zz.isel(quantile=1).plot(ax=ax_c_int,x='return_period',**lstyle)
ax_c_int.fill_between(x=zz.return_period,y1=zz.isel(quantile=0),y2=zz.isel(quantile=-1),
                      alpha=0.5,color=plot_styles[key]['color'])
ax_c_int.set_title('Carmont Intensity (mm/h')
ax_c_int.set_xlabel('Return Period (summers)')
ax_c_int.set_ylabel('Intensity (mm/h)')
fig_carmont_int.show()
commonLib.saveFig(fig_carmont_int,figtype='.pdf')
## Now plot the intensity and prob changes.
scale_cet = 5.61 # Hard wired from earlier comp.

label = commonLib.plotLabel()
fig,(ax_ir,ax_pr) = plt.subplots(nrows=2,ncols=2,figsize=(8,5),
                                 clear=True,num='Dumb_carmont_changes',layout='constrained')
for key in list(delta_intens.keys())[::-1]:
    delta_intens[key].plot(ax=ax_ir,x='return_period',**plot_styles[key],label=key)
    # plot the range
    zz=delta_intens_extra[key].where(msk).stack(idx=['grid_longitude', 'grid_latitude']).\
        quantile([0.05,0.5, 0.95], dim='idx')
    lstyle=plot_styles[key].copy()
    lstyle.update(linestyle='dashdot')
    zz.isel(quantile=1).plot(ax=ax_ir,x='return_period',**lstyle)
    ax_ir.fill_between(x=zz.return_period,y1=zz.isel(quantile=0),y2=zz.isel(quantile=-1),alpha=0.5,color=plot_styles[key]['color'])
    pr[key].plot(ax=ax_pr, **plot_styles[key], label=key)
    zz = probs_extra[key].where(msk).stack(idx=['grid_longitude', 'grid_latitude']).\
        quantile([0.05, 0.5,0.95], dim='idx')
    lstyle=plot_styles[key].copy()
    lstyle.update(linestyle='dashdot')
    zz.isel(quantile=1).plot(ax=ax_pr,**lstyle)
    ax_pr.fill_between(x=zz.threshold, y1=zz.isel(quantile=0), y2=zz.isel(quantile=-1), alpha=0.5,
                       color=plot_styles[key]['color'])
    dt = temps[key]-temps['PI']
    ax_ir.axhline(dt*scale_cet,color=plot_styles[key]['color'],linestyle='dashed')
ax_ir.legend()
ax_ir.set_xlabel("Return period (summers)")
ax_ir.set_ylabel("Intensity change (%)")
ax_ir.set_title("Intensity change")
ax_pr.set_xlabel('Seasonal Max Rain (mm/h)')
ax_pr.set_ylabel("PR (%)")
ax_pr.set_title("Probability Ratio")

for ax in [ax_ir,ax_pr]:
    label.plot(ax)

fig.show()
commonLib.saveFig(fig)

## define region



## plot "todays" intensity for return periods of 1:20 & 1:100
fig_today,axis_today = plt.subplots(nrows=1,ncols=2,figsize=(8,4),clear=True,num='map_intensity_today',
                        subplot_kw=dict(projection=CPMlib.projRot))
label = commonLib.plotLabel()

kw_colorbar = dict(orientation='horizontal', fraction=0.1, aspect=40, pad=0.05,
                   spacing='uniform', label='mm/h')
intensity_levels=np.arange(15,35.,2.5)
intensity_levels = np.arange(24,30)
norm_intensity = mcolors.BoundaryNorm(intensity_levels, ncolors=256, extend='both')
cmap = 'RdYlBu'
for ax,rtn_prd in zip(axis_today,[100]):
    intensity = dists_extra['2012-2021'].sel(pvalues=1.0/rtn_prd,method='nearest')
    carmont = float(intensity.sel(**CPMlib.carmont_drain, method='Nearest'))
    msk = (intensity < carmont * 1.1) * (intensity > carmont * 0.9)
    #di = di.rolling(grid_latitude=smooth,grid_longitude=smooth,center=True).mean()
    intensity.plot(ax=ax, cmap=cmap, levels=intensity_levels, add_colorbar=False, alpha=0.4)
    intensity.where(msk).plot(ax=ax,cmap=cmap, levels=intensity_levels,add_colorbar=False,alpha=0.75)

    # topog_mn.plot.contour(ax=ax,colors=['blue','palegreen','green','brown'],
    #                       levels=[200,300,500,800],linewidths=1)
    ax.set_title(f'Intensity (2012-2022) rp={rtn_prd} ')
    ax.set_extent(CPMlib.stonehaven_rgn_extent)
    CPM_rainlib.std_decorators(ax,radar_col='green',show_railways=True)
    g = ax.gridlines(draw_labels=True)
    g.top_labels = False
    g.left_labels = False
    label.plot(ax)
    # add on carmont
    ax.plot(*CPMlib.carmont_long_lat, transform=ccrs.PlateCarree(), marker='*', ms=10, color='black')
cm = mcm.ScalarMappable(norm=norm_intensity, cmap=cmap)
fig_today.colorbar(cm, ax=axis_today, boundaries=intensity_levels, **kw_colorbar)
fig_today.show()
commonLib.saveFig(fig_today)

## plot maps of intensity and PR change for today.



smooth= 1
fig,axis = plt.subplots(nrows=2,ncols=2,figsize=(8,8),clear=True,num='map_intensity_prob',
                        subplot_kw=dict(projection=CPMlib.projRot))
label = commonLib.plotLabel()

# plot the intensities first
kw_colorbar = dict(orientation='horizontal', fraction=0.1, aspect=40, pad=0.05, spacing='uniform', label='% increase')
intensity_levels=np.arange(4,13.,1)
norm_intensity = mcolors.BoundaryNorm(intensity_levels, ncolors=256, extend='both')
cmap = 'RdYlBu'


for ax,rtn_prd in zip(axis[0,:],[20,100]):
    di = delta_intens_extra['2012-2021'].sel(pvalues=1.0/rtn_prd,method='nearest')
    di = di.rolling(grid_latitude=smooth,grid_longitude=smooth,center=True).mean()
    di.plot(ax=ax,cmap=cmap, levels=intensity_levels,add_colorbar=False)

    topog_mn.plot.contour(ax=ax,colors=['blue','palegreen','green','brown'],
                          levels=[200,300,500,800],linewidths=1)
    ax.set_title(f'Intensity Increase  rp={rtn_prd} ')

cm = mcm.ScalarMappable(norm=norm_intensity, cmap=cmap)
fig.colorbar(cm, ax=list(axis[0,:]), boundaries=intensity_levels, **kw_colorbar)
# now to plot the pr's
rr_levels=np.arange(130,200,10)
norm_rr = mcolors.BoundaryNorm(rr_levels, ncolors=256, extend='both')
kw_colorbar.update(label='%')
for ax,intensity in zip(axis[1,:],[20,30]):
    rr = probs_extra['2012-2021'].sel(threshold=intensity,method='nearest')
    rr = rr.rolling(grid_latitude=smooth,grid_longitude=smooth,center=True).mean()
    rr.plot(ax=ax,cmap=cmap, levels=rr_levels,add_colorbar=False)

    topog_mn.plot.contour(ax=ax,colors=['blue','palegreen','green','brown'],
                          levels=[200,300,500,800],linewidths=1)
    ax.set_title(f'PR for {intensity} mm/h')

cm = mcm.ScalarMappable(norm=norm_rr, cmap=cmap)
fig.colorbar(cm, ax=list(axis[1,:]), boundaries=rr_levels, **kw_colorbar)
for ax in axis.flatten():
    ax.set_extent(rgn_stonehaven_lst)
    ax.coastlines()
    g = ax.gridlines(draw_labels=True)
    g.top_labels = False
    g.left_labels = False
    label.plot(ax)
    # add on carmont
    ax.plot(*CPMlib.carmont_long_lat,transform=ccrs.PlateCarree(),marker='*',ms=10,color='black')
    # and stonehaven
    #ax.plot(*CPMlib.stonehaven_long_lat,transform=ccrs.PlateCarree(),marker='o',ms=10,color='black')
    CPM_rainlib.std_decorators(ax,radar_col='green')
fig.show()
commonLib.saveFig(fig)
