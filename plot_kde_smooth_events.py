# Does qualitative comparison of CPM and radar events via kde plots of common time,
# note no bias correction (in terms of CET done)
import itertools
import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np
import xarray
import CPMlib
import seaborn as sns
import statsmodels.api as sm

import commonLib
import  pandas as pd
import cftime

filter = True
dataset = xarray.load_dataset(CPMlib.CPM_filt_dir/"CPM_filter_all_events.nc") # load the processed events
radar_dataset = xarray.load_dataset(CPMlib.radar_dir/"radar_events/events_2008_5km.nc") # load the processed radar
rolling= 1
fig, axs = plt.subplots(nrows=2, ncols=2, figsize=[8, 6],clear=True, num='CPM_radar_comp',
                                                                         layout='constrained')
for (q,rolling),linestyle in zip(itertools.product([0.5],[1,4]),['solid','dashed']):

    quant = dataset.sel(quantv=q,rolling=rolling).stack(idx=['EventTime','ensemble_member']).dropna('idx') # select the quantile and rolling period
    fit = sm.OLS(quant['CET'].values, sm.add_constant(np.log10(quant['count_cells'].values))).fit()
    print(rolling,fit.summary())
    # Extract  for radar period
    L= ((cftime.datetime(2008,1,1,calendar='360_day') <= quant.t) &
        (quant.t <= cftime.datetime(2023,12,30,calendar='360_day')) )
    quant=quant.where(L,drop=True).dropna('idx')
    area = (quant.count_cells*(4.4)**2).rename("Area")
    log10_area = np.log10(area).rename("log10 Area")

    radar_quant = radar_dataset.sel(quantv=q,rolling=rolling).rename(EventTime='idx').dropna('idx')
    radar_area = (radar_quant.count_cells*(5.0)**2).rename("Rdr Area")
    radar_log10_area = np.log10(radar_area).rename("Rdr log10 Area")

    labels = commonLib.plotLabel()
    for ax,var,obs_var,bins,xlabel in zip(axs.flatten(),
                        [quant.t.dt.hour,log10_area,quant.height,quant.max_precip*rolling],
                        [radar_quant.t.dt.hour,radar_log10_area,radar_quant.height,
                         radar_quant.max_precip*rolling],
                        [24,20,20,20], # bin sizes
                        ['Hour','Log10(Area (km^2))','Height (m)','Max Total precip (mm)']     ):

        #edges = np.histogram_bin_edges(var, bins)
        sns.kdeplot(var,ax=ax,linestyle=linestyle,label=f'CPM Rx{rolling:d}h',
                    color='orange',linewidth=2,cut=0)
        sns.kdeplot(obs_var,ax=ax,linestyle=linestyle,label=f'RADAR Rx{rolling:d}h',
                    color='green',linewidth=2,cut=0)



        # sns.histplot(var,bins=edges,kde=True,ax=ax,stat='density',line_kws=dict(linewidth=4,label='CPM'),
        #              alpha=0.7,color='orange')
        # sns.histplot(obs_var,bins=edges,kde=True,ax=ax,color='green',alpha=0.7,stat='density',
        #              line_kws=dict(linewidth=4,label='RADAR'))
        ax.set_xlabel(xlabel)
        labels.plot(ax)

axs[-1][-1].legend()
axs[-1][-1].set_xlim(0,40)
fig.show()
commonLib.saveFig(fig)

# # plto combined distributions -- largely to show no systematic change
# #Will do one plot per figure as using jointplot
# plt.close('all') # close all exisiting figures
# if filter:
#     fname='_filter'
# else:
#     fname = ''
# g=sns.jointplot(x=quant.CET,y=quant.t.dt.hour,kind='kde',fill=True,height=3,xlim=[12.5,23.5],ylim=[0,23],cut=0)
# g.fig.savefig(f'figures/cet_hour{fname}.png')
# g=sns.jointplot(x=quant.CET,y=log10_area,kind='kde',fill=True,height=3,xlim=[12.5,23.5],ylim=[1.2,4.],cut=5)
# g.fig.savefig(f'figures/cet_log10_area{fname}.png')
# g=sns.jointplot(x=quant.CET,y=quant.height,kind='kde',fill=True,height=3,xlim=[12.5,23.5],ylim=[0,300.],cut=0)
# g.fig.savefig(f'figures/cet_height{fname}.png')
