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
filter = True
dataset = xarray.open_dataset(CPMlib.CPM_filt_dir/"CPM_filter_all_events.nc",chunks={}) # load the processed events
radar_dataset = xarray.load_dataset(CPMlib.radar_dir/"radar_events/events_2008_5km.nc") # load the processed radar
radar_dataset_c4 = xarray.load_dataset(CPMlib.radar_dir/"radar_events/events_2008_1km_c4.nc") # load the processed radar
radar_dataset_c5 = xarray.load_dataset(CPMlib.radar_dir/"radar_events/events_2008_1km_c5.nc") # load the processed radar
my_logger.debug(f"Loaded datasets")
fig, axs = plt.subplots(nrows=2, ncols=4, figsize=[8, 6],clear=True,
                        num='kde_smooth_events',sharex='col', layout='constrained')

fig.get_layout_engine().set(rect=[0.05,0.0,0.95,1.0])#.execute(fig)

labels = commonLib.plotLabel()
plot_col_titles=True
for (q,rolling),axis in zip(itertools.product([0.5],[1,4]),axs):

    quant = dataset.sel(quantv=q,rolling=rolling).stack(idx=['EventTime','ensemble_member']).dropna('idx') # select the quantile and rolling period
    my_logger.debug(f"Loaded model data for quantile {q} rolling {rolling}")
    #fit = sm.OLS(quant['CET'].values, sm.add_constant(np.log10(quant['count_cells'].values))).fit()
    #print(rolling,fit.summary())

    # Extract  for radar period
    time = quant.t.load()
    L= ((cftime.datetime(2008,1,1,calendar='360_day') <= time) &
        (time <= cftime.datetime(2023,12,30,calendar='360_day')) )
    quant = quant.where(L,drop=True).dropna('idx').load()
    my_logger.debug(f"Extracted to 2008-2023")
    radar_quant = radar_dataset.sel(quantv=q,rolling=rolling).rename(EventTime='idx').dropna('idx')
    radar_c4_quant = radar_dataset_c4.sel(quantv=q,rolling=rolling).rename(EventTime='idx').dropna('idx')
    radar_c5_quant = radar_dataset_c5.sel(quantv=q,rolling=rolling).rename(EventTime='idx').dropna('idx')
    my_logger.debug(f"Loaded radar")
    for ds,area in zip([radar_quant,radar_c4_quant,radar_c5_quant,quant],[25.0,16.0,25,4.4*4.4]):
        ds['log10 area'] = np.log10(ds.count_cells*area)
        ds['Hour'] = ds.t.dt.hour
        ds['Accum'] = ds.max_precip*rolling
    my_logger.debug(f"Computed log10_area, hour, accum")
    pos = axis[0].get_position()
    y=(pos.ymax+pos.ymin)/2
    x=0.02
    fig.text(x,y,f'Rx{rolling:d}h',
             ha='left', va='center', rotation=90,fontsize=10)



    for ax,var,bins,xlabel in zip(axis.flatten(),
                        ['Hour','log10 area','height','Accum'],
                        [24,20,20,20], # bin sizes
                        ['Hour',r'$\log_{10}$ Area','Height (m)','Accum precip (mm)']     ):

        for ds,name in zip([quant,radar_quant,radar_c4_quant,radar_c5_quant],
                                 ['CPM','RADAR 5km','RADAR 1km-c4','RADAR 1km-c5']):
            color = CPMlib.radar_cols.get(name.split(" ")[-1].replace('-','_'),'orange')
            sns.kdeplot(ds[var],ax=ax,label=f'{name} ',
                        color=color,linewidth=2,cut=0)



        ax.set_xlabel(xlabel,fontsize='small')
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        labels.plot(ax)


axs[-1][-1].set_xlim(0,50)
axs[0][1].legend(fontsize='small',loc='lower left')
fig.show()
commonLib.saveFig(fig)

