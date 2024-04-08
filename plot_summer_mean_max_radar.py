#plot the summer mean and max rainfall.
import CPM_rainlib
import CPMlib
import xarray
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

import commonLib

rolling=4
rgn={k:slice(v-85e3,v+75e3) for k,v in CPMlib.carmont_drain_OSGB.items()}
ds=xarray.open_dataset(CPMlib.radar_dir/'summary/summary_2008_1km.nc')

ds_summer= ds.where(ds.time.dt.season=='JJA',drop=True).sel(**rgn).mean('time').sel(rolling=rolling).load()
topog = CPM_rainlib.read_90m_topog(region=rgn,resample=11)
## plot means
extent = []
for k in ['projection_x_coordinate','projection_y_coordinate']:
    extent += [rgn[k].start,rgn[k].stop]
fig_radar_summ, axis = plt.subplots(nrows=1,ncols=2,figsize=(8,5),num='radar_jja',
                                    clear=True,layout='constrained',
                                    subplot_kw=dict(projection=ccrs.OSGB()))

cmap = 'RdYlBu'
kw_colorbar = CPMlib.kw_colorbar.copy()
kw_colorbar.update(label='JJA Mean (mm/day)')
(ds_summer.Radar_rain_Mean*24).plot(ax=axis[0],cbar_kwargs=kw_colorbar,robust=True,cmap=cmap)

kw_colorbar.update(label=f'JJA Mean Rx{rolling:d}h (mm)')
(ds_summer.Radar_rain_Max*rolling).plot(ax=axis[1],cbar_kwargs=kw_colorbar,robust=True,cmap=cmap)

label = commonLib.plotLabel()
for ax in axis:
    label.plot(ax)
    ax.set_extent(extent,crs=ccrs.OSGB())
    CPM_rainlib.std_decorators(ax,radar_col='green',radarNames=True,show_railways=True)
    g=ax.gridlines(draw_labels=True)
    g.top_labels = False
    g.left_labels = False
    ax.plot(*tuple(CPMlib.carmont_drain_OSGB.values()), transform=ccrs.OSGB(), marker='o', ms=8, color='black')
    # cs=topog.plot.contour(ax=ax,colors=['blue','palegreen','green','brown'],
    #                       levels=[200,300,500,800],linewidths=1)
    # cs.clabel(inline=True, fontsize=10)

axis[0].set_title("Mean JJA rain")
axis[1].set_title("Mean JJA Monthly Max rain")
fig_radar_summ.show()
commonLib.saveFig(fig_radar_summ)

