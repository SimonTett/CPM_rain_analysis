# plot railways
import cartopy.crs as ccrs
import cartopy.feature
import cartopy.io.shapereader
import pathlib
import  matplotlib.pyplot as plt
import CPM_rainlib
import CPMlib
# read in dhe data
datadir = pathlib.Path(
    r'C:\Users\stett2\OneDrive - University of Edinburgh\data\common_data\UK_railway_DS_10283_2423\UK_Railways.zip')
fname = r"zip://" + (datadir / 'Railway.shx').as_posix()
records = cartopy.io.shapereader.Reader(fname)
railways = cartopy.feature.ShapelyFeature(records.geometries(),
                                          crs=ccrs.OSGB())
## now plot them
fig=plt.figure(num='railways',figsize=(8,5),clear=True,layout='constrained')
ax=fig.add_subplot(111,projection=ccrs.OSGB())
ax.set_extent(CPMlib.stonehaven_rgn_extent,crs=CPMlib.projRot)
ax.add_feature(railways,linewidth=2,facecolor='none',edgecolor='purple')
#ax.coastlines()
CPM_rainlib.std_decorators(ax)
fig.show()