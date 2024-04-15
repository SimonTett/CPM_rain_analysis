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
import os
my_logger = CPM_rainlib.logger
commonLib.init_log(my_logger,level='INFO')


# get OS API key -- needed for detailed map
try:
    os_api_key = os.environ['OS_API_KEY']
except KeyError:
    print("OS_API_KEY not found in environment variables.")
    print("Please set it to your Ordnance Survey API key.")
    print("You can get one by registering at https://osdatahub.os.uk")
    print("If you are SFBT get it from https://osdatahub.os.uk/projects/Plot_carmont")
    raise

my_logger.info(f"Got api key {os_api_key}")

carmont_rgn = {k: slice(v - 75e3, v + 75e3) for k, v in CPMlib.carmont_drain_OSGB.items()}
big_carmont_rgn = {k: slice(v - 100e3, v + 90e3) for k, v in CPMlib.carmont_drain_OSGB.items()}
big_carmont_extent = [big_carmont_rgn['projection_x_coordinate'].start, big_carmont_rgn['projection_x_coordinate'].stop,
                      big_carmont_rgn['projection_y_coordinate'].start, big_carmont_rgn['projection_y_coordinate'].stop]
scalebarprops = dict(frameon=False)

path = CPMlib.radar_dir / 'summary/summary_2008_1km.nc'

radar_data = xarray.open_dataset(path).Radar_rain_Max.sel(rolling=4, time=slice('2020-06-01', '2020-08-31')).max('time'
                                                                                                                 ).sel(
    **big_carmont_rgn
    ) * 4
topog = CPM_rainlib.read_90m_topog(region=big_carmont_rgn, resample=11
                                   ).squeeze()  # read in topography and regrid to about 1km resoln

rseasMskmax, mxTime, top_fit_grid = CPM_rainlib.get_radar_data(path, topog_grid=11, region=carmont_rgn,
                                                               height_range=slice(1, None)
                                                               )

cmap = 'RdYlBu'
levels = [25, 50, 75, 100, 125, 150]
levels = np.linspace(5, 60, 12)
locations = dict(
    Stonehaven = (-2.211,56.964),
    Montrose = (-2.467,56.708),
    Dyce = (-2.198,57.2025),
    Aviemore=(-3.823,57.194),
    Aberdeen = (-2.11,57.15)
)

fig, axis = plt.subplot_mosaic([['topog','zoom'], ['jja2020max', 'aug2020']],
                               figsize=[8, 7], clear=True, layout='constrained',
                               num='carmont_geog_group', subplot_kw=dict(projection=CPMlib.projOSGB)
                               )
# plot the accident area
scale = 14 # empirically derived.



ord_survey_img = cimgt.OrdnanceSurvey(os_api_key, layer='Outdoor', cache=True)
center_pt = CPMlib.carmont_OSGB  # lat/lon of Carmont accident.
zoom = 1000  # for zooming out of center point
extent = [center_pt['projection_x_coordinate'] - zoom, center_pt['projection_x_coordinate'] + zoom,
          center_pt['projection_y_coordinate'] - zoom, center_pt['projection_y_coordinate'] + zoom]  # adjust to zoom
axis['zoom'].set_extent(extent, crs=ccrs.OSGB())  # set extents
axis['zoom'].add_image(ord_survey_img, int(scale))  # add OSM with zoom specification
axis['zoom'].plot(*CPMlib.carmont_long_lat, mec='firebrick', marker='*',mfc='None',mew=2,
                  ms=12, transform=ccrs.PlateCarree()
                  )
scalebar = ScaleBar(1, "m", **scalebarprops)
axis['zoom'].add_artist(scalebar)
my_logger.info("Plotted zoom")
# plot the topography
topog.plot(ax=axis['topog'], cmap='terrain',
           levels=[-200, -100, 0, 50, 100, 200, 300, 400, 500, 600, 700, 800],
           cbar_kwargs=dict(label='height (m)')
           )
## add locations of places refered to int he paper text.
for locn, coord in locations.items():
    axis['topog'].text(*coord,locn[0:2], transform=ccrs.PlateCarree(),fontweight='bold')#,backgroundcolor='grey',alpha=0.7)
# add circles around the radar at roughly 60 and 120 km corresponding to roughly 1 km and 2km resoln.

for (range, name) in itertools.product([60, 120], ['Hill of Dudwick', 'Munduff Hill']):
    co_ords = CPM_rainlib.radar_stations.loc[name, ['Easting', 'Northing']].astype(float)
    # Create a circle
    circle = mpatches.Circle(co_ords, radius=range * 1000, transform=ccrs.OSGB(),
                             edgecolor='green', linewidth=2, facecolor='none'
                             )
    # Add the circle to the axis
    axis['topog'].add_patch(circle)
scalebar = ScaleBar(1, "m", **scalebarprops)
axis['topog'].add_artist(scalebar)
my_logger.info("Plotted topog")
cm = radar_data.plot(ax=axis['jja2020max'], levels=levels, transform=CPMlib.projOSGB, cmap=cmap, add_colorbar=False)
dofyear = pd.to_datetime('2020-08-12').dayofyear
rn_max = rseasMskmax.sel(time='2020', rolling=4).where(mxTime.sel(rolling=4, time='2020').dt.dayofyear == dofyear) * 4
print(f"area of 2020-08-12 event is {float(rn_max.count())} km^2")

rn_max.plot(ax=axis['aug2020'], cmap=cmap, transform=CPMlib.projOSGB, add_colorbar=False, levels=levels)

my_logger.info("Plotted radar for aug2020")

# plot a box!
xstart = carmont_rgn['projection_x_coordinate'].start
xstop = carmont_rgn['projection_x_coordinate'].stop
ystart = carmont_rgn['projection_y_coordinate'].start
ystop = carmont_rgn['projection_y_coordinate'].stop
x, y = [xstart, xstart, xstop, xstop, xstart], [ystart, ystop, ystop, ystart, ystart]
for ax_name in ['aug2020', 'jja2020max']:
    axis[ax_name].plot(x, y, color='black', linewidth=2, transform=CPMlib.projOSGB)

label = commonLib.plotLabel()
for ax, title in zip(axis.values(), ['Topography','Accident Site',  '2020 JJA Max', 'Masked 2020-08-12']):
    if title == 'Accident Site':
        pass  # don't want to add anything!

    else:
        ax.set_extent(big_carmont_extent, crs=CPMlib.projOSGB)
        CPM_rainlib.std_decorators(ax, radar_col='green', radarNames=(title == 'Topography'), show_railways=True)

    CPMlib.plot_carmont(ax)

    ax.set_title(title, size='large')
    label.plot(ax)

## add an inset plot of the British Isles

axBI = axis['topog'].inset_axes([0.65, 0.025, 0.4, 0.65],
                                projection=ccrs.OSGB()
                                )
axBI.set_extent((-11, 2, 50, 61), crs=ccrs.PlateCarree())
axBI.tick_params(labelleft=False, labelbottom=False)
CPMlib.plot_carmont(axBI)
axBI.coastlines()
##
fig.colorbar(cm, ax=list(axis.values()), label='Rx4h Accumulation (mm)', **CPMlib.kw_colorbar)
fig.show()
commonLib.saveFig(fig)
