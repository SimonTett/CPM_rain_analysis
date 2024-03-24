# Mapping Carmont City Open Street Map (OSM) with Cartopy
# This code uses a spoofing algorithm to avoid bounceback from OSM servers
# Based on https://makersportal.com/blog/2020/4/24/geographic-visualizations-in-python-with-cartopy
#
import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt
import io
from urllib.request import urlopen, Request
from PIL import Image
import CPMlib
import commonLib


def image_spoof(self, tile): # this function pretends not to be a Python script
    url = self._image_url(tile) # get the url of the street map API
    req = Request(url) # start request
    req.add_header('User-agent','Anaconda 3') # add user agent to request
    fh = urlopen(req)
    im_data = io.BytesIO(fh.read()) # get image
    fh.close() # close url
    img = Image.open(im_data) # open image with PIL
    img = img.convert(self.desired_tile_form) # set image format
    return img, self.tileextent(tile), 'lower' # reformat for cartopy

#cimgt.OSM.get_image = image_spoof # reformat web request for street map spoofing
osm_img = cimgt.OSM() # spoofed, downloaded street map

fig = plt.figure(figsize=(12,8),clear=True,num='carmont_osm') # open matplotlib figure
ax = fig.add_subplot(111, projection=ccrs.OSGB()) # project using coordinate reference system (CRS) of street map
center_pt = CPMlib.carmont_long_lat # lat/lon of Carmont accident.
zoom = 0.03 # for zooming out of center point
extent = [center_pt[0]-(zoom*2.0),center_pt[0]+(zoom*2.0),
          center_pt[1]-zoom,center_pt[1]+zoom] # adjust to zoom
ax.set_extent(extent) # set extents

scale = np.ceil(-np.sqrt(2)*np.log(np.divide(zoom,350.0))) # empirical solve for scale based on zoom
# NOTE: zoom specifications should be selected based on extent:
# -- 2     = coarse image, select for worldwide or continental scales
# -- 4-6   = medium coarseness, select for countries and larger states
# -- 6-10  = medium fineness, select for smaller states, regions, and cities
# -- 10-12 = fine image, select for city boundaries and zip codes
# -- 14+   = extremely fine image, select for roads, blocks, buildings
scale =16
scale = (scale<20) and scale or 19 # scale cannot be larger than 19
ax.add_image(osm_img, int(scale)) # add OSM with zoom specification
for coord,color in zip([CPMlib.carmont_long_lat,CPMlib.carmont_drain_long_lat],
                       ['black','purple']):
    ax.plot(*coord,color=color, marker='*', ms=10, linewidth=4, transform=ccrs.PlateCarree())

# show 1km and 5 km grids,
import xarray
for file,color in zip(["summary_1km/1km_summary.nc","summary_5km_1hr_scotland.nc"],
                      ['red','black']):

    ds=xarray.open_dataset(CPMlib.radar_dir/file).monthlyMax
    X,Y= np.meshgrid(ds.projection_x_coordinate,ds.projection_y_coordinate)
    ax.scatter(X,Y,marker='+',s=100,transform=ccrs.OSGB(),color=color)
plt.show() # show the plot
commonLib.saveFig(fig)