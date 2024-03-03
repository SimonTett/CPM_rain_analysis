"""
Make some images to illustrate how "events" are defined.
"""
import matplotlib.pyplot as plt
import xarray
import numpy as np
import CPMlib
import cartopy.crs as ccrs

import commonLib
import pandas as pd
import CPM_rainlib

projGB = ccrs.OSGB()
v=tuple(CPMlib.stonehaven.values())
stonehaven = projGB.transform_point(*v,CPMlib.projRot)
var_names = ["projection_x_coordinate","projection_y_coordinate"]
stonehaven = dict(zip(var_names,stonehaven))
stonehaven_rgn={k:slice(v-75e3,v+75e3) for k,v in stonehaven.items()}
path = CPMlib.radar_dir/'summary_5km_1hr_scotland.nc'
radar_data=xarray.load_dataset(path).monthlyMax.sel(time=slice('2020-06-01','2020-08-31')).max('time')

rseasMskmax, mxTime, top_fit_grid=CPM_rainlib.get_radar_data(path,region=stonehaven_rgn,height_range=slice(1,300))

cmap='RdYlBu'
kw_colorbar = dict(orientation='horizontal',fraction=0.1,aspect=40,pad=0.05)
fig,axis=plt.subplots(nrows=1,ncols=2,figsize=[8,5],clear=True,
                    num='illuminate_group',subplot_kw=dict(projection=projGB))

radar_data.plot(ax=axis[0],robust=True,transform=projGB,cmap=cmap,
                cbar_kwargs=kw_colorbar)
dofyear = pd.to_datetime('2020-08-12').dayofyear
rn_max = rseasMskmax.sel(time='2020').where(mxTime.sel(time='2020').dt.dayofyear == dofyear)
print(f"area of 2020-08-12 event is {float(rn_max.count())*25} km^2")
rn_max.plot(ax=axis[1],robust=True,cmap=cmap,transform=projGB,
            cbar_kwargs=kw_colorbar)
for ax,title in zip(axis,['2020 JJA Max','Masked 2020-08-12']):
    ax.set_extent([stonehaven_rgn['projection_x_coordinate'].start, stonehaven_rgn['projection_x_coordinate'].stop,
                   stonehaven_rgn['projection_y_coordinate'].start, stonehaven_rgn['projection_y_coordinate'].stop], crs=projGB)
    CPM_rainlib.std_decorators(ax,radar_col='red',radarNames=True)
    ax.plot(*tuple(stonehaven.values()),transform=projGB,marker='*',ms=8,color='black')
    ax.plot(*tuple(CPMlib.carmont_OSGB.values()),transform=projGB,marker='o',ms=8,color='black')

    # add on contours for topogrophy. Will show 300m, 500m & 800m
    cs=top_fit_grid.plot.contour(ax=ax,colors=['blue','palegreen','green','brown'],levels=[200,300,500,800],linewidths=2)
    cs.clabel(inline=True, fontsize=10)
    ax.set_title(title,size='large')
fig.show()
commonLib.saveFig(fig)

# make a small plot with just the distribution + mark the quantiles.
quants=np.array([0,0.5,1,0.2,0.5,0.8,0.9,0.95,1.0])
fig = plt.figure(clear=True,num='plot_rain',figsize=[11,3])
data = np.sort(rn_max.stack(indx=list(rn_max.coords)[0:2]).squeeze().dropna('indx'))
x=np.arange(0,len(data))
qx=(quants*(len(data)-1)).astype('int')
plt.bar(x,data,color='green')
plt.bar(qx,data[qx],color='red')
plt.ylabel("JJA Max (mm/hr)")
fig.show()
commonLib.saveFig(fig)


