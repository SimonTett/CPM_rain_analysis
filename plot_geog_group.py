"""
Plot some maps to give geography and also illustrate groups.
"""
import matplotlib.pyplot as plt
import xarray
import numpy as np
import CPMlib
import cartopy.crs as ccrs
from matplotlib_scalebar.scalebar import ScaleBar

import commonLib
import pandas as pd
import CPM_rainlib
import cartopy.io.img_tiles as cimgt
import itertools
import matplotlib.patches as mpatches
import cartopy.crs as ccrs

projGB = ccrs.OSGB()


carmont_rgn = {k: slice(v - 75e3, v + 75e3) for k, v in CPMlib.carmont_drain_OSGB.items()}
big_carmont_rgn = {k: slice(v - 90e3, v + 90e3) for k, v in CPMlib.carmont_drain_OSGB.items()}
#big_carmont_rgn_ll = {k: slice(v - 2.5, v + 2.5) for k, v in CPMlib.carmont_drain_long_lat.items()}
big_carmont_extent = [big_carmont_rgn['projection_x_coordinate'].start, big_carmont_rgn['projection_x_coordinate'].stop,
                      big_carmont_rgn['projection_y_coordinate'].start, big_carmont_rgn['projection_y_coordinate'].stop]
scalebarprops=dict(frameon=False)

path = CPMlib.radar_dir / 'summary/summary_2008_1km.nc'

radar_data = xarray.open_dataset(path).Radar_rain_Max.sel(rolling=4,time=slice('2020-06-01', '2020-08-31')).max('time').sel(**big_carmont_rgn)*4
topog = CPM_rainlib.read_90m_topog(region=big_carmont_rgn, resample=11).squeeze()  # read in topography and regrid to about 1km resoln

rseasMskmax, mxTime, top_fit_grid = CPM_rainlib.get_radar_data(path, topog_grid=11, region=carmont_rgn, height_range=slice(1, None))

cmap = 'RdYlBu'
levels = [25,50,75,100,125,150]
levels=np.linspace(5,60,12)

fig, axis = plt.subplot_mosaic([['zoom','topog'],['jja2020max','aug2020']],
                               figsize=[8, 7], clear=True,layout='constrained',
                                num='carmont_geog_group', subplot_kw=dict(projection=projGB))
# plot the accident area
scale =16
osm_img = cimgt.OSM(cache=True,user_agent='Anaconda 3')
center_pt = CPMlib.carmont_OSGB  # lat/lon of Carmont accident.
zoom = 3500  # for zooming out of center point
extent = [center_pt['projection_x_coordinate'] - zoom, center_pt['projection_x_coordinate'] + zoom,
          center_pt['projection_y_coordinate'] - zoom, center_pt['projection_y_coordinate'] + zoom]  # adjust to zoom
axis['zoom'].set_extent(extent, crs=ccrs.OSGB())  # set extents
axis['zoom'].add_image(osm_img, int(scale)) # add OSM with zoom specification
axis['zoom'].plot(*CPMlib.carmont_long_lat,color='firebrick', marker='*',
                  ms=12, transform=ccrs.PlateCarree())
scalebar = ScaleBar(1,"m",**scalebarprops)
axis['zoom'].add_artist(scalebar)
# plot the topography
topog.plot(ax=axis['topog'],cmap='terrain',
                  levels=[-200,-100,0,50,100,200,300,400,500,600,700,800],
                  cbar_kwargs=dict(label='height (m)'))
# add circles around the radar at roughly 60 and 120 km corresponding to roughly 1 km and 2km resoln.

for (range,name) in itertools.product([60,120],['Hill of Dudwick','Munduff Hill']):
    co_ords = CPM_rainlib.radar_stations.loc[name,['Easting','Northing']].astype(float)
    # Create a circle
    circle = mpatches.Circle(co_ords, radius=range*1000, transform=ccrs.OSGB(),
                             edgecolor='green',linewidth=2, facecolor='none')
    # Add the circle to the axis
    axis['topog'].add_patch(circle)
scalebar = ScaleBar(1,"m",**scalebarprops)
axis['topog'].add_artist(scalebar)


cm = radar_data.plot(ax=axis['jja2020max'], levels=levels,transform=projGB, cmap=cmap, add_colorbar=False)
dofyear = pd.to_datetime('2020-08-12').dayofyear
rn_max = rseasMskmax.sel(time='2020',rolling=4).where(mxTime.sel(rolling=4,time='2020').dt.dayofyear == dofyear)*4
print(f"area of 2020-08-12 event is {float(rn_max.count()) } km^2")
#(top_fit_grid < 50).where(True,np.nan).plot(cmap='gray_r',ax=axis['aug2020'],transform=projGB,add_colorbar=False)
rn_max.plot(ax=axis['aug2020'], cmap=cmap, transform=projGB, add_colorbar=False,levels=levels)
# plot the topog over.


# plot a box!
xstart=carmont_rgn['projection_x_coordinate'].start
xstop=carmont_rgn['projection_x_coordinate'].stop
ystart=carmont_rgn['projection_y_coordinate'].start
ystop=carmont_rgn['projection_y_coordinate'].stop
x, y = [xstart, xstart, xstop, xstop, xstart], [ystart, ystop, ystop, ystart, ystart]
for ax_name in ['aug2020','jja2020max']:
    axis[ax_name].plot(x, y, color='black', linewidth=2,transform=ccrs.OSGB())


# show 1km and 5 km grids,

for file,color in zip(["summary/summary_2008_1km.nc","summary/summary_2008_5km.nc"],
                      ['red','black']):

    ds=xarray.open_dataset(CPMlib.radar_dir/file).Radar_rain_Mean
    X,Y= np.meshgrid(ds.projection_x_coordinate,ds.projection_y_coordinate)
    axis['zoom'].scatter(X,Y,marker='+',s=100,transform=ccrs.OSGB(),color=color)
label=commonLib.plotLabel()
for ax, title in zip(axis.values(), ['Accident Site','Topography','2020 JJA Max', 'Masked 2020-08-12']):
    if title == 'Accident Site':
        pass # don't want to add anything!

    else:
        ax.set_extent(big_carmont_extent,crs=ccrs.OSGB())
    CPM_rainlib.std_decorators(ax, radar_col='green', radarNames=(title=='Topography'), show_railways=True)
    ax.plot(*CPMlib.carmont_drain_long_lat, transform=ccrs.PlateCarree(),
            marker='o', ms=6, color='cornflowerblue')

    ax.set_title(title, size='large')
    label.plot(ax)

## add an inset plot of the British Isles

axBI = axis['topog'].inset_axes([0.62,0.025,0.4,0.70],
                  projection=ccrs.OSGB())
axBI.set_extent((-11, 2, 50, 61),crs=ccrs.PlateCarree())
axBI.tick_params(labelleft=False, labelbottom=False)
axBI.plot(*CPMlib.carmont_drain_long_lat, transform=ccrs.PlateCarree(),
        marker='o', ms=6, color='cornflowerblue')
axBI.coastlines()
##
fig.colorbar(cm, ax=list(axis.values()), label='Rx4h Accumulation (mm)', **CPMlib.kw_colorbar)
fig.show()
commonLib.saveFig(fig)


