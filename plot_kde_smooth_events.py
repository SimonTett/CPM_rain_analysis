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
                        num='kde_smooth_events',sharex='col', layout='constrained')

fig.get_layout_engine().set(rect=[0.05,0.0,0.95,1.0])#.execute(fig)

labels = commonLib.plotLabel()
plot_col_titles=True
for (q,rolling),axis in zip(itertools.product([0.95],[1,4]),axs):
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
    for ds,area in zip([radar_quant, radar_c4_quant, radar_c5_quant, quant, raw_quant], [25.0, 16.0, 25, 4.4 * 4.4, 4.4 * 4.4]):
        L = (ds.t.dt.season == 'JJA' ) & (ds.t.dt.year >=2008 )  &  (ds.t.dt.year <= 2023)
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
                        [ dict(gridsize=24),dict(gridsize=30,log_scale=(10.,None)),dict(gridsize=30),dict(gridsize=30)]):

        for ds,name in zip([quant, raw_quant, radar_quant, radar_c4_quant, radar_c5_quant],
                                 ['CPM','CPM-Raw','5km','1km-c4','1km-c5']):
            color = CPMlib.radar_cols.get(name.split(" ")[-1].replace('-','_'),'orange')
            if 'Raw' in name:
               color='brown'
            kdeplot_args.update(label=f'{name} ',color=color,linewidth=2,cut=0)
            sns.kdeplot(ds[var],ax=ax,**kdeplot_args)

            if var == 'height': # plot the heights from the topog.
                kwargs = kdeplot_args.copy()
                kwargs.update(color='k',linestyle='dashed',label='Topog 5km')
                sns.kdeplot(topog_5km.values.flatten(),ax=ax, **kwargs)
            if name == '5km': # 5km radar data.
                ax.axvline(ds[var].sel(idx=ds.indx_2020_08_12),color=color,linestyle='dashed')
        ax.set_xlabel(xlabel,fontsize='small')
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        labels.plot(ax)

for ax in axs:
    ax[1].set_xscale('log')
axs[-1][-1].set_xlim(0,80)
axs[0][1].legend(fontsize='small',loc='lower left')
fig.show()
commonLib.saveFig(fig)

