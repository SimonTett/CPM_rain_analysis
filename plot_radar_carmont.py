# plot radar for carmont.
import matplotlib.pyplot as plt

import CPM_rainlib
import CPMlib
import commonLib
import numpy as np

radar_1km_file = CPM_rainlib.nimrodRootDir / 'metoffice-c-band-rain-radar_uk_20200812_1km-composite.dat.gz.tar'  #
# aug-12 2020
radar_1km = CPM_rainlib.extract_nimrod_day(radar_1km_file, region=CPMlib.carmont_rgn_OSGB).sel(
    **CPMlib.carmont_drain_OSGB, method='nearest'
)
radar_5km_file = CPM_rainlib.nimrodRootDir / 'metoffice-c-band-rain-radar_uk_20200812_5km-composite.dat.gz.tar'  #
# aug-12 2020
radar_5km = CPM_rainlib.extract_nimrod_day(radar_5km_file, region=CPMlib.carmont_rgn_OSGB).sel(
    **CPMlib.carmont_drain_OSGB, method='nearest'
)

## Plot the data.
lab = commonLib.plotLabel()
fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(5, 3), num='radar_carmont', clear=True, sharex=True,
                        layout='constrained'
                        )


fig.get_layout_engine().execute(fig)
for radar, label, color in zip([radar_1km, radar_5km], ['1km/5min', '5km/15min'], ['red', 'blue']):
    if label.startswith('1km'):
        scale=  12.
    else:
        scale = 4.
    r = radar.sel(time=slice('2020-08-12T02:45', '2020-08-12T10:15'))
    r.plot(ax=axs[0], drawstyle='steps-mid', label=label, color=color)
    cumsum = r.cumsum('time')/scale

    mx = cumsum.values[-1]
    indx = (np.abs(cumsum-mx/2)).idxmin().values
    time_half = cumsum.time.where(cumsum < mx/2,drop=True).dt.strftime('%H:%M').values[-1]
    time_max = cumsum.time.where(cumsum == mx, drop=True).dt.strftime('%H:%M').values[0]
    value = cumsum.sel(time=indx)

    cumsum.plot(ax=axs[0], drawstyle='steps-mid',
                                   color=color, linestyle='dotted',label=None)
    axs[0].scatter(indx,value,edgecolors=color,facecolors='None',linewidth=2,marker='o', s=30,linestyle='None', label=None)
    r.resample(time='1h').mean().plot(ax=axs[1], drawstyle='steps-mid', label=label, color=color)
    print(f'{label} Rainfall accumulation is {mx:.1f} mm with 1/2 rainfall from {time_half} to {time_max}')
axs[0].legend()
for a, title in zip(axs, ['Raw', '1h']):
    a.set_title(title + ' Radar Rainfall at Carmont')
    a.set_ylabel('mm/h')
    lab.plot(a)
    a.set_ylim(0.0, None)
    time = np.datetime64('2020-08-12 09:37')
    a.annotate('Derail',(time,0),ha ='center',
               xytext=(0,5),textcoords='offset fontsize',
             arrowprops=dict(facecolor='black', shrink=0.01))
fig.show()
commonLib.saveFig(fig)
