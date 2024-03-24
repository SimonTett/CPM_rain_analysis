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


projGB = ccrs.OSGB()
v = tuple(CPMlib.stonehaven.values())
stonehaven = projGB.transform_point(*v, CPMlib.projRot)
var_names = ["projection_x_coordinate", "projection_y_coordinate"]
stonehaven = dict(zip(var_names, stonehaven))
stonehaven_rgn = {k: slice(v - 75e3, v + 75e3) for k, v in stonehaven.items()}
scalebarprops=dict(frameon=False)

path = CPMlib.radar_dir / 'summary_5km_1hr_scotland.nc'
radar_data = xarray.load_dataset(path).monthlyMax.sel(time=slice('2020-06-01', '2020-08-31')).max('time')

rseasMskmax, mxTime, top_fit_grid = CPM_rainlib.get_radar_data(path, region=stonehaven_rgn, height_range=slice(1, 300))

cmap = 'RdYlBu'
levels = np.linspace(5, 30, 11)

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
top_fit_grid.plot(ax=axis['topog'],cmap='terrain',
                  levels=[-200,-100,0,100,200,300,400,500,600,700],
                  cbar_kwargs=dict(label='height (m)'))
scalebar = ScaleBar(1,"m",**scalebarprops)
axis['topog'].add_artist(scalebar)
# add an inset plot of the British Isles
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import cartopy.mpl.geoaxes
axBI = inset_axes(axis['topog'], width="45%", height="40%", loc="lower right",
                  axes_class=cartopy.mpl.geoaxes.GeoAxes, borderpad=0.,
                  axes_kwargs=dict(projection=ccrs.PlateCarree()))
axBI.set_extent((-11, 2, 50, 61),crs=ccrs.PlateCarree())
axBI.tick_params(labelleft=False, labelbottom=False)
axBI.plot(*CPMlib.carmont_drain_long_lat, transform=ccrs.PlateCarree(),
        marker='o', ms=6, color='cornflowerblue')
axBI.coastlines()

cm = radar_data.plot(ax=axis['jja2020max'], levels=levels,transform=projGB, cmap=cmap, add_colorbar=False)
dofyear = pd.to_datetime('2020-08-12').dayofyear
rn_max = rseasMskmax.sel(time='2020').where(mxTime.sel(time='2020').dt.dayofyear == dofyear)
print(f"area of 2020-08-12 event is {float(rn_max.count()) * 25} km^2")
rn_max.plot(ax=axis['aug2020'], robust=True, cmap=cmap, transform=projGB, add_colorbar=False)

# show 1km and 5 km grids,

for file,color in zip(["summary_1km/1km_summary.nc","summary_5km_1hr_scotland.nc"],
                      ['red','black']):

    ds=xarray.open_dataset(CPMlib.radar_dir/file).monthlyMax
    X,Y= np.meshgrid(ds.projection_x_coordinate,ds.projection_y_coordinate)
    axis['zoom'].scatter(X,Y,marker='+',s=100,transform=ccrs.OSGB(),color=color)
label=commonLib.plotLabel()
for ax, title in zip(axis.values(), ['Accident Site','Topography','2020 JJA Max', 'Masked 2020-08-12']):
    if title == 'Accident Site':
        pass # don't want to add anything!

    else:
        ax.set_extent(CPMlib.stonehaven_rgn_extent, crs=CPMlib.projRot)
    CPM_rainlib.std_decorators(ax, radar_col='green', radarNames=(title=='Topography'), show_railways=True)
    ax.plot(*CPMlib.carmont_drain_long_lat, transform=ccrs.PlateCarree(),
            marker='o', ms=10, color='cornflowerblue')

    ax.set_title(title, size='large')
    label.plot(ax)
fig.colorbar(cm, ax=list(axis.values()), label='Rx1h (mm/h)', **CPMlib.kw_colorbar)
fig.show()
commonLib.saveFig(fig)


