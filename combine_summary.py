#!/bin/env python
"""
Combine summary files to produce 1 file 
"""
import pathlib
import xarray
import numpy as np
import pandas as pd
import argparse
import dask

parser=argparse.ArgumentParser(description="combine processed radar data")
parser.add_argument("dir",type=str,help='Dir to process')
parser.add_argument("--pattern",help='glob pattern for files',default='radar_rain*.nc')
parser.add_argument("output",type=str,help='Name of Output file')
parser.add_argument("--nocompress","-n",action='store_true',
                    help="Do not compress output data")
parser.add_argument("--verbose","-v",action='store_true',help='Be verbose')
parser.add_argument('--region', nargs=4, type=float, help='Region to extract (x0, x1,y0,y1)')


args=parser.parse_args()
if args.verbose:
    print(args)

region = None
if args.region is not None:
    region = dict(
        projection_x_coordinate=slice(args.region[0], args.region[1]),
        projection_y_coordinate=slice(args.region[2], args.region[3])
    )

dask.config.set({"array.slicing.split_large_chunks": True})
chunks=dict(time=12,projection_y_coordinate=100,projection_x_coordinate=100)
files=list(pathlib.Path(args.dir).glob(args.pattern))
files=sorted(files)

ds=xarray.open_mfdataset(files,chunks=chunks,parallel=True)
if region:
    ds=ds.sel(**region)

encoding=dict() # make it an empty dict

if not args.nocompress:
    # compress the ouput... useful because quite a lot of the data is missing
    comp = dict(zlib=True)
    for v in ds.data_vars:
        encoding[v]=comp

ds.to_netcdf(args.output,encoding=encoding)





