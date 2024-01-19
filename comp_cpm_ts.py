
"""
Compute seasonal-mean CET, stonehaven region & CPM average  from
  monthly-mean temperatures and precipitation. (No CET for precip!)

"""
import CPMlib
import CPM_rainlib
import matplotlib.pyplot as plt
import matplotlib.patches
import cartopy.crs as ccrs
import xarray

wts = dict(
    Rothampsted=1.0/3,
    Malvern=1.0/3.,
    Squires_Gate=1.0/6,
    Ringway=1.0/6
)
coords = dict(
    Stonehaven=(-2.211,56.964),
    Rothampsted=(-0.35, 51.8),
    Malvern=(-2.317, 52.117),
    Squires_Gate=(-3.033, 53.767),
    Ringway=(-2.267, 53.35)
)







projRot=ccrs.RotatedPole(pole_longitude=177.5,pole_latitude=37.5)
proj=ccrs.PlateCarree()
# compute rotated co-ords
rotated_coords={k:projRot.transform_point(*c,proj) for k,c in coords.items()}
# add 360 to the x coord to match co-ords of grid
rotated_coords={k:[c[0]+360,c[1]] for k,c in rotated_coords.items()}



fig=plt.figure(num='test_cet',clear=True)
ax=fig.add_subplot(111,projection=proj)
ax.set_extent([-5,2, 50,58],crs=proj)
# plot
for key,coords in coords.items():
    ax.plot(coords[0],coords[1],marker='x',color='red')
    crot = rotated_coords[key]
    ax.plot(*crot,marker='+',color='blue',transform=projRot)

ax.coastlines()


## now to read the ensemble data.

stonehaven=rotated_coords['Stonehaven']
offset=0.75 # degress offset. 1 degree is ~ 100 km
region = dict(zip(['grid_longitude','grid_latitude'],
                  [slice(v-offset,v+offset) for v in stonehaven]))

# now to plot the region -- which is in rotated coords
xy=(region['grid_longitude'].start,region['grid_latitude'].start)
w=region['grid_longitude'].stop-xy[0]
h=region['grid_latitude'].stop-xy[1]

rect = matplotlib.patches.Rectangle(xy,width=w,height=h,alpha=0.1,transform=projRot)
ax.add_patch(rect)
fig.show()
chunks=dict(grid_longitude=10,grid_latitude=10)
for var in ['tas','pr']:
    cpm_list=[]
    ed_list=[]
    cet_list=[]
    for p in CPM_rainlib.cpmDir.glob('[0-9][0-9]'):
        pth=p/var/'mon/latest'
        pth_rain = p/'pr/mon/latest'
        ncfiles=list(pth.glob('*.nc'))
        da=xarray.open_mfdataset(ncfiles,
                                 chunks=chunks,parallel=True,
                                 concat_dim='time',combine='nested',
                                 data_vars='minimal',coords='minimal',
                                 compat='override')[var]
        # not too bad performance! The extra args come from the xarray doc.
        cet=0.0
        for key,wt in wts.items():
            coords=rotated_coords[key]
            ts = da.sel(method='nearest',tolerance=0.1,
                             grid_longitude=coords[0],
                             grid_latitude=coords[1]).load()
            cet += (ts*wt)
        
        
        cet_list.append(cet)
        stonehaven_ts=da.sel(**region).mean(CPM_rainlib.cpm_horizontal_coords).load()
        stonehaven_list.append(stonehaven_ts)
        cpm_ts=da.mean(CPM_rainlib.cpm_horizontal_coords).load()
        cpm_list.append(cpm_ts)
        print(f"Done with {p} for {var}") # end loop over ensemble members
        




    cet=xarray.concat(cet_list,dim='ensemble_member')
    stonehaven_data=xarray.concat(stonehaven_list,dim='ensemble_member')
    cpm=xarray.concat(cpm_list,dim='ensemble_member')
    cet.to_netcdf(f'cet_{var}.nc')
    stonehaven_data.to_netcdf(f'stonehaven_reg_{var}.nc')
    cpm.to_netcdf(f'cpm_reg_{var}.nc')
    print(f"Done with {var}")
    # end loop over variables
    
    
                             
                    
    
