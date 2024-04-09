# plot radar for carmont.
import matplotlib.pyplot as plt

import CPM_rainlib
import CPMlib
import commonLib

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
for radar, label, color in zip([radar_1km, radar_5km], ['1km/5min', '5km/15min'], ['red', 'blue']):
    r = radar.sel(time=slice('2020-08-12T02:45', '2020-08-12T10:15'))
    r.plot(ax=axs[0], drawstyle='steps-mid', label=label, color=color)
    r.resample(time='1h').mean().plot(ax=axs[1], drawstyle='steps-mid', label=label, color=color)

axs[0].legend()
for a, title in zip(axs, ['Raw', '1h']):
    a.set_title(title + ' Radar Rainfall at Carmont')
    a.set_ylabel('mm/h')
    lab.plot(a)
    a.set_ylim(0.0, None)
fig.show()
commonLib.saveFig(fig)
