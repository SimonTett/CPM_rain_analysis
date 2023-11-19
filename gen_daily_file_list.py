# code to generate a list of daily files. Rather hacky code but as not running lots of time nto worht spending time making
# super clean
import xarray
import CPMlib
import  pandas as pd
# read in the events

cpm_events = xarray.open_dataset(CPMlib.CPM_dir/"CPM_all_events.nc").sel(ensemble_member=1).dropna('EventTime')
time_fmt = cpm_events.t.sel(quantv=0.5).dt.strftime("%Y%m%d")

periods=dict( # periods and year ranges. Each TS#n corresponds to a differnt simulation (which differs per ensmeble)
TS1= ("19800101","19991230"),
TS4= ("20000101","20191230"), #2000 – 2020
TS2 =("20200101","20391230"),#2020 – 2040
TS5= ("20400101","20591230"), #2040 – 2060
TS3= ("20600101","20801231") #2060 - 2080
)
id = time_fmt.ensemble_member_id.str.split("split","-").values[2].astype('str').strip() # get out the ensemble ID
if id != "r001i1p00000": # and check it as expecte
    raise ValueError("Only set up for ens r001i1p00000")
#r001i1p00000
experiments = dict(# simulations for ensmble r001i1p00000
TS1="mi-bb171",
TS2= "mi-bb188",
TS3= "mi-bb189",
TS4= "mi-bc005",
TS5= "mi-bb991"
)
# make the files.
files=[]
for tf in time_fmt:
    # work out the range.
    file = None
    for tp,range in periods.items():
        if (tf <= range[1]) and (tf >= range[0]): # using string comparison here.
            file=experiments[tp]
            continue # found it so skip rest of loop.
    if file is None: # got here with no file. Something went wrong.
        raise ValueError("Failed to find time")
    filename = file.split("-")[1]+"a.pb"+str(tf.values)+".pp" # construct filename
    files.append(filename) # add to list
files=pd.Series(files,index=time_fmt,name='Ens#1 files',dtype='string') # make into a pandas series.
files.index.name='date' # name the index
files.to_csv(CPMlib.CPM_dir/'files_to_retrieve_ens1.csv')


