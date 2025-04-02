# plot geographical context for Carmont within Scotland and the UK.
# use the cartopy land/sea image and then add topography at 5km resoln.
import cartopy.crs as ccrs
import cartopy
import CPM_rainlib

import matplotlib.pyplot as plt

import CPMlib
import commonLib

topog = CPM_rainlib.read_90m_topog(resample=11) # regrid to ~ 1km
## now actually plot it.
fig = plt.figure(figsize=(8,8),num='carmont_context',clear=True)
ax=fig.add_subplot(111,projection=ccrs.OSGB())
topog.plot(ax=ax,cmap='terrain',add_colorbar=True,levels=[-400,-200,0,100,200,300,400,500,800,1000])
ax.plot(*CPMlib.carmont_long_lat, transform=ccrs.PlateCarree(), marker='*', ms=10, color='black')
CPM_rainlib.std_decorators(ax=ax,showregions=False)
ax.set_title("Great Britain DEM @ 1km")
fig.show()
commonLib.saveFig(fig,figtype='.pdf')
