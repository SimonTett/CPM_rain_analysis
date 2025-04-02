# Does qualitative comparison of CPM and radar events via kde plots of common time,
# note no bias correction (in terms of CET done)
import itertools
import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np
import xarray

import CPM_rainlib
import CPMlib
import seaborn as sns
from matplotlib.ticker import MaxNLocator

import commonLib

import cftime
my_logger = CPM_rainlib.logger
commonLib.init_log(my_logger,level='DEBUG')
topog_5km = xarray.load_dataarray(CPMlib.radar_dir/'topography'/'topog_5km.nc').sel(**CPMlib.carmont_rgn_OSGB)
dataset = xarray.open_dataset(CPMlib.CPM_filt_dir/"CPM_filter_all_events.nc",chunks={}) # load the processed events
raw_dataset = xarray.open_dataset(CPMlib.CPM_dir/"CPM_all_events.nc",chunks={}) # load the processed events
radar_dataset = xarray.load_dataset(CPMlib.radar_dir/"radar_events/events_2008_5km.nc") # load the processed radar
radar_dataset_c4 = xarray.load_dataset(CPMlib.radar_dir/"radar_events/events_2008_1km_c4.nc") # load the processed radar
radar_dataset_c5 = xarray.load_dataset(CPMlib.radar_dir/"radar_events/events_2008_1km_c5.nc") # load the processed radar
my_logger.debug(f"Loaded datasets")


fig, axs = plt.subplots(nrows=2, ncols=4, figsize=[8, 6],clear=True,
                        num='kde_smooth_events',sharex='col', sharey='col',layout='constrained')

fig.get_layout_engine().set(rect=[0.05,0.0,0.95,1.0])#.execute(fig)
colors = CPMlib.radar_cols.copy()
colors.update({'CPM':'red','Raw':'brown','Fut':'black'})
linewidths = dict(CPM=1.5,Raw=1.5,Fut=2)
labels = commonLib.plotLabel()
plot_col_titles=True
for (q,rolling),axis in zip(itertools.product([0.5],[1,4]),axs):
    sel = dict(quantv=q,rolling=rolling,method='nearest') # waht we want!
    quant = dataset.sel(**sel).stack(idx=['EventTime','ensemble_member']).dropna('idx') # select the quantile and rolling period
    raw_quant = raw_dataset.sel(**sel).stack(idx=['EventTime', 'ensemble_member']).dropna('idx'
                                                                                                              )  # select the quantile and rolling period
    my_logger.debug(f"Loaded model data for quantile {q} rolling {rolling}")
    #fit = sm.OLS(quant['CET'].values, sm.add_constant(np.log10(quant['count_cells'].values))).fit()
    #print(rolling,fit.summary())

    # Extract  for radar period
    time = quant.t.load()
    L= ((cftime.datetime(2008,1,1,calendar='360_day') <= time) &
        (time <= cftime.datetime(2023,12,30,calendar='360_day')) )
    L2= ((cftime.datetime(2065,1,1,calendar='360_day') <= time) &
        (time <= cftime.datetime(2080,12,30,calendar='360_day')) )
    quant_2065_2080 = quant.where(L2,drop=True).dropna('idx').load()
    quant= quant.where(L,drop=True).dropna('idx').load()

    time = raw_quant.t.load()
    L= ((cftime.datetime(2008,1,1,calendar='360_day') <= time) &
        (time <= cftime.datetime(2023,12,30,calendar='360_day')) )
    raw_quant = raw_quant.where(L, drop=True).dropna('idx').load()
    my_logger.debug(f"Extracted to 2008-2023")
    radar_quant = radar_dataset.sel(**sel).rename(EventTime='idx').dropna('idx')
    radar_c4_quant = radar_dataset_c4.sel(**sel).rename(EventTime='idx').dropna('idx')
    radar_c5_quant = radar_dataset_c5.sel(**sel).rename(EventTime='idx').dropna('idx')
    my_logger.debug(f"Loaded radar")
    for ds,area in zip([radar_quant, radar_c4_quant, radar_c5_quant, quant, raw_quant,quant_2065_2080],
                       [25.0, 16.0, 25, 4.4 * 4.4, 4.4 * 4.4,4.4*4.4]):
        L = (ds.t.dt.season == 'JJA' )
        ds['Area'] = ds.count_cells*area
        ds['Hour'] = ds.t.dt.hour
        ds['Accum'] = ds.max_precip*rolling
        try:
            ds['indx_2020_08_12'] = ds.idx.where(ds.t.dt.strftime('%Y-%m-%d') == '2020-08-12',drop=True).values[0]
        except IndexError:
            pass
    my_logger.debug(f"Computed area, hour, accum")
    pos = axis[0].get_position()
    y=(pos.ymax+pos.ymin)/2
    x=0.02
    fig.text(x,y,f'Rx{rolling:d}h',
             ha='left', va='center', rotation=90,fontsize=10)



    for ax,var,xlabel,kdeplot_args in zip(axis.flatten(),
                        ['Hour','Area','height','Accum'],
                        ['Hour','Area (km$^2$)','Height (m)','Accum precip (mm)'],
                        [ dict(gridsize=24,clip=(0,24)), # hours
                          dict(gridsize=50,log_scale=(10,None)), # area
                          dict(gridsize=50,clip=(0,1000),log_scale=(10,None)), # ht
                          dict(gridsize=50,log_scale=(10,None),clip=(0.1,150.)) # accum
                          ]):

        for ds,name in zip([quant, raw_quant, radar_quant, radar_c4_quant, radar_c5_quant],
                                 ['CPM','Raw', '5km','1km-c4','1km-c5']):
            color = colors[name]
            kdeplot_args.update(label=name.replace('1km-',''),color=color,linewidth=linewidths.get(name,1.0),cut=0)
            sns.kdeplot(ds[var],ax=ax,common_norm=True,**kdeplot_args)

            if var == 'height': # plot the heights from the topog.
                kwargs = kdeplot_args.copy()
                kwargs.update(color='purple',linewidth=1,linestyle='dashed',label='_Topog 5km')
                sns.kdeplot(topog_5km.values.flatten(),common_norm=True,ax=ax, **kwargs)
            if name == '5km': # 5km radar data.
                ax.axvline(ds[var].sel(idx=ds.indx_2020_08_12),color=color,linestyle='dashed')
        ax.set_xlabel(xlabel,fontsize='small')
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        labels.plot(ax)
from matplotlib.ticker import LogLocator

locator = LogLocator(numticks=5)  # 5 ticks
for ax in axs.flat:
    if ax.get_xscale() == 'log':
        ax.xaxis.set_major_locator(locator)
        #ax.set_xlim(10,None)
    else:
        ax.set_xlim(0,None)
    if ax.get_yscale() == 'log':
        ax.yaxis.set_major_locator(locator)
        # truncate at 10^-3
        ylim = ax.get_ylim()
        ax.set_ylim(np.max([1e-3,ylim[0]]),None)
axs[-1][-1].set_xlim(1,200)
axs[1][2].legend(fontsize='x-small',ncols=2,loc='upper left',columnspacing=0.1,borderaxespad=0,borderpad=0.0)
fig.show()
commonLib.saveFig(fig,figtype='pdf')

