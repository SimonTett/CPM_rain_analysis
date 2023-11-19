# test code to work out how to find max lat/lon
import CPMlib
import xarray
import matplotlib.pyplot as plt
import seaborn as sns

grp = CPMlib.discretise(ds.seasonalMaxTime)
grp = xarray.where((t>1) & (tmn<300.), grp,0).rename("EventTime") # land < 300m
dd=CPMlib.event_stats(ds.seasonalMax,ds.seasonalMaxTime,grp).sel(EventTime=slice(1,None))
large= dd.count_cells > 13
dd_large = dd.sel(EventTime=large)

# now plot it.
rgn = CPMlib.stonehaven_rgn

# Now plot where the events occur.
fig,ax = plt.subplots(nrows=1,ncols=1,figsize=[6,8],clear=True,
                      num='event_medians',subplot_kw=dict(projection=CPMlib.projRot))
ax.set_extent([rgn['grid_longitude'].start,rgn['grid_longitude'].stop,
              rgn['grid_latitude'].start,rgn['grid_latitude'].stop],crs=CPMlib.projRot)
xarray.where(tmn > 0,tmn,np.nan).plot(ax=ax,transform=CPMlib.projRot,levels=[1,50,100,150,200,250,300],
                                      cbar_kwargs=dict(orientation='horizontal'))
# use seaborn to produce a KDE plot
df=dd_large[['x_max','y_max']].to_pandas() # make a dataframe
sns.kdeplot(data=df,x='x_max',y='y_max',transform=CPMlib.projRot,colors='black',levels=10,linewidths=2,linestyles=['solid','dashed','dotted']*4)
#ax.scatter(dd_large.x_max,dd_large.y_max,transform=CPMlib.projRot,
#           s=60,c=dd_large.quant_precip.sel(quant=0.9),norm='log',marker='o',alpha=0.5)
c=tuple(CPMlib.stonehaven.values())
ax.plot(c[0],c[1],marker='*',ms=12,transform=CPMlib.projRot)

fig.show()