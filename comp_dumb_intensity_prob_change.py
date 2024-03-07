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
fit=gev_r.xarray_gev(carmont.stack(**stack_dim), cov=[cpm_cet_jja.rename('CET').stack(**stack_dim)], dim='t_e',name='Carmont_C')
fit_extra=gev_r.xarray_gev(maxRain.stack(**stack_dim), cov=[cpm_cet_jja.rename('CET').stack(**stack_dim)], dim='t_e',name='Carmont_C')
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
    dists[key]= gev_isf(fits[key], 1.0 / np.geomspace(2, 100, 200))
    probs[key] = gev_sf(fits[key],np.linspace(10,30,num=200))
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

## Now plot the intensity and prob changes.
scale_cet = 5.61 # Hard wired from earlier comp.
label = commonLib.plotLabel()
fig,(ax_ir,ax_pr) = plt.subplots(nrows=1,ncols=2,figsize=(8,5),
                                 clear=True,num='Dumb_carmont_changes',layout='constrained')
for key in list(delta_intens.keys())[::-1]:
    delta_intens[key].plot(ax=ax_ir,**plot_styles[key],label=key)
    pr[key].plot(ax=ax_pr, **plot_styles[key], label=key)
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

## plot maps of intensity and PR change for today.

smooth= 3
fig,axis = plt.subplots(nrows=2,ncols=2,figsize=(8,8),clear=True,num='map_intensity_prob',
                        subplot_kw=dict(projection=CPMlib.projRot))
label = commonLib.plotLabel()

# plot the intensities first
kw_colorbar = dict(orientation='horizontal', fraction=0.1, aspect=40, pad=0.05, spacing='uniform', label='% increase')
intensity_levels=np.arange(2.,13.,2)
norm_intensity = mcolors.BoundaryNorm(intensity_levels, ncolors=256, extend='both')
cmap = 'RdYlBu'


for ax,rtn_prd in zip(axis[0,:],[20,100]):
    di = delta_intens_extra['2012-2021'].sel(pvalues=1.0/rtn_prd,method='nearest')
    di = di.rolling(grid_latitude=smooth,grid_longitude=smooth,center=True).mean()
    di.plot(ax=ax,cmap=cmap, levels=intensity_levels,add_colorbar=False)
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
    ax.set_title(f'PR for {intensity} mm/h')

cm = mcm.ScalarMappable(norm=norm_rr, cmap=cmap)
fig.colorbar(cm, ax=list(axis[1,:]), boundaries=rr_levels, **kw_colorbar)
rgn_lst = []
for s in CPMlib.stonehaven_rgn.values():
    rgn_lst += [s.start,s.stop]
for ax in axis.flatten():

    ax.set_extent(rgn_lst)
    #ax.coastlines()
    g = ax.gridlines(draw_labels=True)
    g.top_labels = False
    g.left_labels = False
    label.plot(ax)
    # add on carmont
    ax.plot(*CPMlib.carmont_long_lat,transform=ccrs.PlateCarree(),marker='*',ms=10,color='black')
    # and stonehaven
    ax.plot(*CPMlib.stonehaven_long_lat,transform=ccrs.PlateCarree(),marker='o',ms=10,color='black')
    CPM_rainlib.std_decorators(ax)
fig.show()
