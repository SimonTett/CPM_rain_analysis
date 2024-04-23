# compare summary stats from radar and two CPM datasets.
# For CPM data will look at 2006-2022 data.

import CPMlib
import xarray
import logging
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

import commonLib

logging.basicConfig(level='INFO',force=True)
events=dict()
quant=0.5 # want to look at 50%
#quant=0.1
# get in the radar data.
for file,name,cell_area in zip([CPMlib.radar_dir/'radar_events_summary_5km_1hr_scotland.nc',
                      CPMlib.radar_dir/'radar_events_1km_c5.nc',
                      CPMlib.radar_dir/'radar_events_1km_c4.nc'],
                     ['radar_5km','radar_1km_c5','radar_1km_c4'],
                               [25.,25.,16]):
    ds=xarray.load_dataset(file).sel(quantv=quant)
    L=ds.t.dt.season == 'JJA' # extract JJA (which might be all the data)
    ds=ds.sel(EventTime=L) # select out the events we want.
    ds['area'] = ds['count_cells']*cell_area
    events[name]=ds
    nyear = np.unique(ds.t.dt.year).size
    med_area = float(ds.area.median())
    median_maxp = float(ds.max_precip.median())
    print(f"{name} has {events[name].max_precip.size/nyear:3.0f} events per year with med area {med_area:4.0f} km^2 and median max precip {median_maxp:3.1f} mm/hr")
    logging.info(f"Loaded {file} into {name}")

time_prd = slice(2005,2022)
# get in the CPM data selecting out the time period and quant
cell_area = 4.4**2
for file,name in zip([CPMlib.CPM_dir/'CPM_all_events.nc',
                      CPMlib.CPM_filt_dir/'CPM_filter_all_events.nc'],
                    ['Raw CPM','Filt. CPM']):
    ds=xarray.load_dataset(file).sel(quantv=quant).stack(idx=['ensemble_member','EventTime'])
    L = (ds.t.dt.year >= time_prd.start) & (ds.t.dt.year <= time_prd.stop) & (ds.t.dt.season=='JJA')
    ds=ds.sel(idx=L).dropna(dim='idx')
    # convert count_cells to area (and rename)
    ds['area'] = ds['count_cells'] * cell_area
    events[name]=ds
    nyear = np.unique(ds.t.dt.year).size
    med_area = float(ds.area.median())
    median_maxp= float(ds.max_precip.median())
    print(f"{name} has {events[name].max_precip.size/(12*nyear):3.0f} events per year with med area {med_area:4.0f} km^2 and median max precip {median_maxp:3.1f} mm/hr")
    logging.info(f"Loaded {file} pinto {name}")

## plot
plot_styles={'Raw CPM':dict(color='royalblue',marker='x'),
             'Filt. CPM':dict(color='darkblue',marker='x'),
             'radar_5km':dict(color='orange',marker='o'),
             'radar_1km_c5':dict(color='brown',marker='+'),
             'radar_1km_c4':dict(color='red',marker='+')} # put this in CPMlib?
fig,axes = plt.subplots(nrows=1,ncols=3,clear=True,figsize=[8,5],layout='constrained',num='rad_cpm_kde')
for k,event_da in events.items():
    plot_style = plot_styles[k].copy()
    plot_style.update(markevery=5,linewidth=2,gridsize=50,common_grid=True,common_norm=True)
    sns.kdeplot(event_da.t.dt.hour,label=k,ax=axes[0],clip=[0,23],cut=0,**plot_style)
    sns.kdeplot(event_da.height,cut=0,ax=axes[1],**plot_style)
    sns.kdeplot(np.log10(event_da.area),cut=0,ax=axes[2],**plot_style)

axes[0].legend(loc='center left')
# add on labels & title
label = commonLib.plotLabel()
for ax,title in zip(axes,['Hour','Height (m)','Log$_{10}$ Area']):
    ax.set_xlabel(title)
    ax.set_title(f"KDE of {title}")
    label.plot(ax)
fig.show()
commonLib.saveFig(fig)