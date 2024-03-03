#plot the summer mean and max rainfall.
import CPM_rainlib
import CPMlib
import xarray
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

import commonLib

stonehaven_OSGB=dict(zip(["projection_x_coordinate","projection_y_coordinate"],
                        [387171.,785865.])) # from wikipedia https://geohack.toolforge.org/geohack.php?pagename=Stonehaven&params=56.964_N_2.211_W_region:GB_type:city(11150)

stonehaven_rgn={k:slice(v-85e3,v+75e3) for k,v in stonehaven_OSGB.items()}
ds=xarray.open_dataset(CPMlib.radar_dir/'summary_1km/1km_summary.nc')
ds_summer= ds.where(ds.time.dt.season=='JJA',drop=True).sel(**stonehaven_rgn).mean('time').load()
topog = CPM_rainlib.read_90m_topog(region=stonehaven_rgn,resample=11)
## plot means
extent = []
for k in ['projection_x_coordinate','projection_y_coordinate']:
    extent += [stonehaven_rgn[k].start,stonehaven_rgn[k].stop]
fig_radar_summ, axis = plt.subplots(nrows=1,ncols=2,figsize=(8,5),num='radar_jja',
                                    clear=True,layout='constrained',
                                    subplot_kw=dict(projection=ccrs.OSGB()))

cmap = 'RdYlBu'
kw_colorbar = dict(orientation='horizontal',fraction=0.1,aspect=40,pad=0.05)
kw_colorbar.update(label='JJA Mean (mm/day)')
ds_summer.monthlyMean.plot(ax=axis[0],cbar_kwargs=kw_colorbar,robust=True,cmap=cmap)

kw_colorbar.update(label='JJA Mean Rx1h (mm/h)')
ds_summer.monthlyMax.plot(ax=axis[1],cbar_kwargs=kw_colorbar,robust=True,cmap=cmap)

label = commonLib.plotLabel()
for ax in axis:
    label.plot(ax)
    ax.set_extent(extent,crs=ccrs.OSGB())
    CPM_rainlib.std_decorators(ax,radar_col='red',radarNames=True)
    g=ax.gridlines(draw_labels=True)
    g.top_labels = False
    g.left_labels = False
    ax.plot(*tuple(stonehaven_OSGB.values()), transform=ccrs.OSGB(), marker='*', ms=8, color='black')
    ax.plot(*tuple(CPMlib.carmont_OSGB.values()), transform=ccrs.OSGB(), marker='o', ms=8, color='black')
    cs=topog.plot.contour(ax=ax,colors=['blue','palegreen','green','brown'],
                          levels=[200,300,500,800],linewidths=1)
    cs.clabel(inline=True, fontsize=10)

axis[0].set_title("Mean JJA rain")
axis[1].set_title("Mean JJA Monthly 1 hour max rain")
fig_radar_summ.show()
commonLib.saveFig(fig_radar_summ)

