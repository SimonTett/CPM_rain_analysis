# Plot the following:
# For some  quantiles KDE  plots of:
# CPM vs Radar for
# hour of extreme
# area of extreme -- on log plot
# max precipitation distribution.
# Then entirely for CPM to see if there is any dependence on CET:
# hour vs CET
# log10 area vs CET
# Height vs CET

import matplotlib.pyplot as plt
import numpy as np
import xarray
import CPMlib
import seaborn as sns

import commonLib
import  pandas as pd
import cftime

dataset = xarray.load_dataset(CPMlib.CPM_dir/"CPM_all_events.nc") # load the processed events
radar_dataset = xarray.load_dataset(CPMlib.datadir/"radar_events.nc") # load the processed radar
dataset_flat= dataset.stack(idx=['ensemble_member','EventTime']).dropna('idx')
for q in [0.1,0.5,0.9]:

    quant = dataset_flat.sel(quantv=q)
    area = (quant.count_cells*(4.4)**2).rename("Area")
    log10_area = np.log10(area).rename("log10 Area")


    radar_quant = radar_dataset.sel(quantv=q)
    radar_area = (radar_quant.count_cells*(5.0)**2).rename("Rdr Area")
    radar_log10_area = np.log10(radar_area).rename("Rdr log10 Area")


    # plot KDEs for radar period
    L= ((cftime.datetime(2005,1,1,calendar='360_day') <= quant.t) &
        (quant.t <= cftime.datetime(2023,12,30,calendar='360_day')) )
    fig,axes = plt.subplots(nrows=2,ncols=2,figsize=[11,7],clear=True,num=f"Obs Period q={q} ")
    crit_area=(2.2**2)*25.
    L2 = L & (area > crit_area)
    L_rad = radar_area > crit_area
    for ax,var,obs_var,bins,xlabel in zip(axes.flatten(),
                        [quant.t.dt.hour,area,quant.height,quant.max_precip],
                        [radar_quant.t.dt.hour,radar_area,radar_quant.height,radar_quant.max_precip],
                        [24,100,50,50], # bin sizes
                        ['Hour','Area (km^2)','Height (m)','Max precip (mm/hr)']     ):
        if 'Area' in var.name : # only apply time mask
            vv = var[L]
            vv_radar = obs_var
            log_scale=[True,True]
            ax.axvline(crit_area,linestyle='dashed')
            edges = np.histogram_bin_edges(np.log10(vv), bins)

        else: # restrict to areas > crit_area
            #vv=var[L2]
            #vv_radar = obs_var[L_rad]
            vv = var[L]
            vv_radar = obs_var
            log_scale=False
            edges = np.histogram_bin_edges(vv, bins)


        sns.histplot(vv,bins=edges,kde=True,ax=ax,stat='density',line_kws=dict(linewidth=4,label='CPM'),
                     alpha=0.7,log_scale=log_scale,color='orange')
        sns.histplot(vv_radar,bins=edges,kde=True,ax=ax,color='green',alpha=0.7,stat='density',
                     line_kws=dict(linewidth=4,label='RADAR'),log_scale=log_scale)
        ax.set_xlabel(xlabel)

    axes[0][0].legend()
    fig.tight_layout()
    fig.show()
    commonLib.saveFig(fig)

# plto combined distributions -- largely to show no systematic change
#Will do one plot per figure as using jointplot
plt.close('all') # close all exisiting figures
g=sns.jointplot(x=quant.CET,y=quant.t.dt.hour,kind='kde',fill=True,height=3,xlim=[12.5,23.5],ylim=[0,23],cut=0)
g.fig.savefig('figures/cet_hour.png')
g=sns.jointplot(x=quant.CET,y=log10_area,kind='kde',fill=True,height=3,xlim=[12.5,23.5],ylim=[1.2,4.],cut=5)
g.fig.savefig('figures/cet_log10_area.png')
g=sns.jointplot(x=quant.CET,y=quant.height,kind='kde',fill=True,height=3,xlim=[12.5,23.5],ylim=[0,300.],cut=0)
g.fig.savefig('figures/cet_height.png')
