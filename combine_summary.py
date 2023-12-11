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
chunks=dict(time=24,projection_y_coordinate=100,projection_x_coordinate=100)
files=pathlib.Path(args.dir).glob("*monthly*.nc")
# remove all the 2004 files
files_wanted = sorted([ file for file in files if '2004' not in file.name])
if args.verbose:
    print("Combining ",files_wanted)

ds=xarray.open_mfdataset(files_wanted,chunks=chunks,combine='nested')
if region:
    ds=ds.sel(**region)
encoding=dict()

if not args.nocompress:
    # compress the ouput... useful because quite a lot of the data is missing
    comp = dict(zlib=True)
    for v in ds.data_vars:
        encoding[v]=comp

ds.to_netcdf(args.output,encoding=encoding)





