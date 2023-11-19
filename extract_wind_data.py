#!/bin/env python
import argparse
import cartopy.crs as ccrs
import iris
import pathlib
import xarray
stonehaven=dict(grid_longitude=360.16, grid_latitude=4.46) # co-ords on rotated grid

rgns={k: (lambda cell: v-1.0 < cell < v+1.0) for k,v in stonehaven.items()}
constraint = iris.Constraint(**rgns) # make constraint
pth=pathlib.Path(r'c:\users\stett2\Downloads\bb171a.pb19840723.pp')
cubes=iris.load(pth)

cubes2 =[c.extract(constraint) for c in cubes]