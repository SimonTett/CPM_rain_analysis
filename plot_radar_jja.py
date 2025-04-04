#plot the summer mean and max rainfall.
import CPM_rainlib
import CPMlib
import xarray
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cartopy.crs as ccrs
import itertools
import commonLib

rolling = 4
rgn = {k: slice(v - 85e3, v + 75e3) for k, v in CPMlib.carmont_drain_OSGB.items()}
ds = xarray.open_dataset(CPMlib.radar_dir / 'summary/summary_2008_1km.nc')

ds_summer = ds.where(ds.time.dt.season == 'JJA', drop=True).sel(**rgn).sel(rolling=rolling).load()
topog = CPM_rainlib.read_90m_topog(region=rgn, resample=11)

## plot means
point_features = [(-1.35, 57.49), (-2.0, 57.15), (-1.85, 57.00), (-2.65, 56.47)]  # features that are points
# radial features.
radial_features = [(-1.77, 56.66), (-3.06, 56.82)]  # radial features -- likely radar obstructions.
alpha_features = 0.4 # alpha for plotting radar artifacts.
extent = []
for k in ['projection_x_coordinate', 'projection_y_coordinate']:
    extent += [rgn[k].start, rgn[k].stop]
fig_radar_summ, axis = plt.subplots(nrows=1, ncols=2, figsize=(8, 5), num='radar_jja',
                                    clear=True, layout='constrained',
                                    subplot_kw=dict(projection=ccrs.OSGB())
                                    )

cmap = 'RdYlBu'
kw_colorbar = CPMlib.kw_colorbar.copy()
kw_colorbar.update(label='JJA Mean (mm/day)')
(ds_summer.Radar_rain_Mean.mean('time') * 24).plot(ax=axis[0], cbar_kwargs=kw_colorbar, robust=True, cmap=cmap)

kw_colorbar.update(label=f'JJA Median Rx{rolling:d}h (mm)')
(ds_summer.Radar_rain_Max.median('time') * rolling).plot(ax=axis[1], cbar_kwargs=kw_colorbar, robust=True, cmap=cmap)

label = commonLib.plotLabel()
for ax in axis:
    label.plot(ax)
    ax.set_extent(extent, crs=ccrs.OSGB())
    CPM_rainlib.std_decorators(ax, radar_col='green', radarNames=True, show_railways=True)
    g = ax.gridlines(draw_labels=True)
    g.top_labels = False
    g.left_labels = False
    CPMlib.plot_carmont(ax)
    # add circles around the radar at roughly 60 and 120 km corresponding to roughly 1 km and 2km resoln.

    for (range, name) in itertools.product([60, 120], ['Hill of Dudwick', 'Munduff Hill']):
        co_ords = CPM_rainlib.radar_stations.loc[name, ['Easting', 'Northing']].astype(float)
        # Create a circle
        circle = mpatches.Circle(co_ords, radius=range * 1000, transform=ccrs.OSGB(),
                                 edgecolor='green', linewidth=2, facecolor='none'
                                 )
        # Add the circle to the axis
        ax.add_patch(circle)
    # plot radial features around Munduff Hill
    for co_ords in radial_features:
        x = [CPM_rainlib.radar_stations.loc['Munduff Hill', 'Longitude'], co_ords[0]]
        y = [CPM_rainlib.radar_stations.loc['Munduff Hill', 'Latitude'], co_ords[1]]
        ax.plot(x, y, transform=ccrs.PlateCarree(), color='green', linewidth=4, alpha=0.2)
    for co_ords in point_features:
        ax.plot(*co_ords, marker='o', mec='green',mfc='none',mew=2,ms=14, alpha=alpha_features, transform=ccrs.PlateCarree())

axis[0].set_title("Mean JJA rain")
axis[1].set_title("Median JJA Monthly Max rain")
fig_radar_summ.show()
commonLib.saveFig(fig_radar_summ,figtype='pdf')
