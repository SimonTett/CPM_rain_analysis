# Does qualitative comparison of CPM for 2008-2023 vs 2065-2080  via kde plots of common time,
# note no bias correction (in terms of CET done)
import itertools

import cftime
import matplotlib.pyplot as plt
import seaborn as sns
import xarray
from matplotlib.ticker import MaxNLocator, LogLocator

import CPM_rainlib
import CPMlib
import commonLib
import numpy as np

my_logger = CPM_rainlib.logger
commonLib.init_log(my_logger, level='DEBUG')

dataset = xarray.open_dataset(CPMlib.CPM_filt_dir / "CPM_filter_all_events.nc", chunks={})  # load the processed events
raw_dataset = xarray.open_dataset(CPMlib.CPM_dir / "CPM_all_events.nc", chunks={})  # load the processed events

datasets = {"Filtered": dataset, "Raw": raw_dataset}
datasets_2008_2023 = dict()
datasets_2065_2080 = dict()
for key, ds in datasets.items():
    time = ds.t.load()
    area = (ds.count_cells * 4.4 ** 2).rename('Area')
    hour = ds.t.dt.hour.rename('Hour')
    height = ds.height.rename('Height')
    accum = (ds.max_precip * ds['rolling']).rename("Accum")
    cet = ds.CET
    L1 = ((cftime.datetime(2008, 1, 1, calendar='360_day') <= time) &
          (time <= cftime.datetime(2023, 12, 30, calendar='360_day')) &
          (time.dt.season == 'JJA')
          )

    L2 = ((cftime.datetime(2065, 1, 1, calendar='360_day') <= time) &
          (time <= cftime.datetime(2080, 12, 30, calendar='360_day')) &
          (time.dt.season == 'JJA')
          )
    lst_2008_2023=[]
    lst_2065_2080=[]
    for var in [area,cet]:
        lst_2008_2023 += [var.where(L1.isel(quantv=0), drop=True)]
        lst_2065_2080 += [var.where(L2.isel(quantv=0), drop=True)]

    for v in [hour, accum, height]:
        lst_2008_2023 += [v.where(L1, drop=True)]
        lst_2065_2080 += [v.where(L2, drop=True)]
    datasets_2008_2023[key] = xarray.merge(lst_2008_2023)
    datasets_2065_2080[key] = xarray.merge(lst_2065_2080)

my_logger.debug(f"Opened datasets and masked to 2008-2023 and 2065-2080")
# %%


## plot the data


#fig, axs = plt.subplots(nrows=2, ncols=4, figsize=[8, 6], clear=True,
#                        num='kde_smooth_events_2065_2080', sharex='col', sharey='col',layout='constrained'
#                        )


rx=['Hour', 'Area', 'Height', 'Accum']
layout = [[f'Rx1h {r}' for r in rx],[f'Rx4h {r}' for r in rx],['Median']*len(rx)]
fig, axs = plt.subplot_mosaic(layout, figsize=[8, 7], clear=True, height_ratios=[2,2,1],
                        num='kde_smooth_events_2065_2080', layout='constrained')
fig.get_layout_engine().set(rect=(0.05, 0.0, 0.95, 1.0)) # slightly wider margins

labels = commonLib.plotLabel()
plot_col_titles = True
now_color = 'red'
future_color = 'blue'



for (q, rolling) in itertools.product([0.5], [1, 4]):
    sel = dict(quantv=q, rolling=rolling, method='nearest')  # what we want!
    ds_now = {key: ds.sel(**sel).stack(idx=['EventTime', 'ensemble_member']).dropna('idx').load() for key, ds in
              datasets_2008_2023.items()}
    ds_future = {key: ds.sel(**sel).stack(idx=['EventTime', 'ensemble_member']).dropna('idx').load() for key, ds in
                 datasets_2065_2080.items()}
    # plot RxnH/Q on axis
    key = f'Rx{rolling:1d}h Hour'
    pos = axs[key].get_position()
    y = (pos.ymax + pos.ymin) / 2
    x = 0.02
    fig.text(x, y, f'Rx{rolling:d}h Q:{q:3.1f}',
             ha='left', va='center', rotation=90, fontsize=10
             )

    for var, unit, kdeplot_args in zip(
                                             rx,
                                             ['', 'km$^2$', 'm', 'mm'],
                                             [dict(gridsize=24, clip=(0, 24)),  # hours
                                              dict(gridsize=50, log_scale=(10, None)),  # area
                                              dict(gridsize=50, clip=(0, 1000), log_scale=(10, None)),  # ht
                                              dict(gridsize=50, log_scale=(10, None), clip=(0.1, 150.))  # accum
                                              ]
                                             ):
        xlabel = var
        if unit != '':
            xlabel+=f'({unit})'
        key = f'Rx{rolling:1d}h {var}'
        ax = axs[key]
        name = 'Filtered'
        linestyle='solid'
        if var in ['Area','Accum']:
            change = ds_future['Filtered'][var].mean('idx')/ds_now[name][var].mean('idx') 
            change = (float(change) -1)*100.
            title = f'{change:2.0f}%'

        elif var in ['Hour']:
            change = ds_future[name][var].mean('idx')-ds_now[name][var].mean('idx') 
            change = change*60
            title= f'{change:2.0f} mins'
        else:
            change = ds_future[name][var].mean('idx')-ds_now[name][var].mean('idx') 

            title=f'{change:2.0f} {unit}'
        #if rolling == 1:
        title = var+' '+title
        ax.set_title(title)
        kdeplot_args.update(linewidth=2, cut=0,linestyle=linestyle)
        sns.kdeplot(ds_now[name][var], ax=ax, label='2008-23', color=now_color, **kdeplot_args)
        sns.kdeplot(ds_future[name][var], ax=ax, label='2065-80', color=future_color, **kdeplot_args)
        qvalues = [0.10, 0.5, 0.9]
        now_median = ds_now[name][var].quantile(qvalues).values
        fut_median = ds_future[name][var].quantile(qvalues).values
        rat = fut_median / now_median
        for m in now_median:
            ax.axvline(m, color=now_color, linestyle='dashed')
        for m in fut_median:
            ax.axvline(m, color=future_color, linestyle='dashed')

        print(f'{name} {var} Rx{rolling:d}h Q:{q:2.1f} 2008-23 : {now_median[1]:3.1f}'
              f' 2065-2080: {fut_median[1]:3.1f} %increase: {100 * (rat[1] - 1):3.0f} %'
              )
        locator = LogLocator(numticks=5)  # 5 ticks
        ax.set_xlabel(unit)#, fontsize='small')
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        labels.plot(ax)
        if ax.get_xscale() == 'log':
            ax.xaxis.set_major_locator(locator)
            #ax.set_xlim(10,None)
        else:
            ax.set_xlim(0, None)
        if ax.get_yscale() == 'log':
            ax.yaxis.set_major_locator(locator)
            # truncate at 10^-3
            ylim = ax.get_ylim()
            ax.set_ylim(np.max([1e-3, ylim[0]]), None)



axs['Rx1h Height'].legend(fontsize='small', loc='upper center')


## Explore chanegs in rain and area.
datas=[v.stack(idx=['EventTime', 'ensemble_member']).load()
    for v in [datasets_2008_2023['Filtered'],datasets_2065_2080['Filtered']]]

area = [ds.Area for ds in datas]
wt = (datas[0].Accum.quantv.shift(quantv=-1, fill_value=1) -
      datas[0].Accum.quantv.shift(quantv=1, fill_value=0))/2.0
rain = [ds.Accum.weighted(wt).mean('quantv') for ds,a in zip(datas,area)]
rain_0_9 = [ds.Accum.sel(quantv=0.9) for ds,a in zip(datas,area)]
rain_mx = [ds.Accum.sel(quantv=1) for ds,a in zip(datas,area)]
cet = [ds.CET.weighted(a.fillna(0.0)).mean('idx') for ds,a in zip(datas,area)]
#cet = [ds.CET.mean('idx') for ds in datas]

cet_delta = cet[1]-cet[0]
area_ratio = (area[1].mean('idx')/area[0].mean('idx'))
rain_ratio = (rain[1]/rain[0])
rain_ratio_0_9 = (rain_0_9[1] / rain_0_9[0])
with (np.printoptions(precision=2)):
    print('Cet Delta',cet_delta.values)
for name,var in zip(['Mn Rain','Rain (0.9)','mxRain','Area'],
                    [rain,rain_0_9,rain_mx,area]):
    if name == 'Area':
        mn = [v.mean('idx') for v in var]


    else:
        mn = [v.weighted(a.fillna(0.0)).mean('idx') for v,a in zip(var,area)]
    ratio = mn[1]/mn[0]
    values = ((ratio-1)/cet_delta)*100
    with (np.printoptions(precision=2)):
        print('ratio ', name,values.values)
        print('2008-23',name,mn[0].values)
        



## Produce plot showing  change between two periods as a function
# of quantile. Will only consider median extreme rather than trying to fit GEV's
# simple enough that no complex explanation needed. 
ax = axs['Median']
mn_coords = ['EventTime','ensemble_member']

median_extremes_2065_2080 = {k:ds.Accum.median(mn_coords) for k,ds in datasets_2065_2080.items()}
median_extremes_2008_2023 = {k:ds.Accum.median(mn_coords) for k,ds in datasets_2008_2023.items()}
ratio_med_extremes = {k:(median_extremes_2065_2080[k]*100/median_extremes_2008_2023[k]-100.).load() for k in median_extremes_2008_2023.keys()}
# compute area change in each period.

area_change = {k: (datasets_2065_2080[k].Area.median(mn_coords)/datasets_2008_2023[k].Area.median(mn_coords)).load() for k in datasets_2008_2023.keys()}
area_change = {k:(area-1)*100 for k,area in area_change.items()}# % increase

# now plot the ratios for filltered  data & 1 & 4 hour accum.


for rolling,linestyle in zip([1,4],['solid','dashed','dotted']):
    ratio_med_extremes['Filtered'].sel(rolling=rolling).plot(ax=ax,label=f'Rx{rolling:d}h',color='k',linestyle=linestyle)
ax.legend(loc='lower right',ncols=3)
ax.set_ylabel('% Change')
ax.set_xlabel('Quantile')
ax.set_title('Median Seasonal Extreme Increase (%)')
ax.set_yticks([5,10,15,20,25])
labels.plot(ax)
fig.show()
commonLib.saveFig(fig,figtype='pdf')
        
        
        





