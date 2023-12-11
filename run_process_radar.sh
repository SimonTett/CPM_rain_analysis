#!/bin/bash

# run processing jobs for 1km radar data.
# Takes about 3 hours per year when running interactively. Will give 12 hours!
. ./set_env_jasmin.sh # set up the environment
mkdir -p output # where output is going
years="2004 2005 2006 2007 2008 2009 2010 2011 2012 2013 2014 2015 2016 2017 2018 2019 2020 2021 2022"
#years=2023
TIME="1:00:00"
RESOLN="5-km"
Q='short-serial'
# 1km analysis
TIME="12:00:00"
RESOLN="1km"
Q='high-mem --mem=100000' # high mem q with a 100 GB of RAM
# for 1km data takes about 4 hours to process a year and 100 GB might be overkill.
for year in $years
do
	export TMPDIR=scratch/$year
	mkdir -p $TMPDIR
	echo "Made $TMPDIR submitting jobs"
	cmd="./process_radar_data.py $year --resolution 1km --verbose --log_level DEBUG  --coarsen 1 2 4 5 8 --region 52500 497500 502500 1497500"
	slurm_cmd="sbatch -p $Q --time=$TIME -o output/process_radar_${RESOLN}_${coarsen}_${year}.out ${cmd}"
	#
	echo "Running: $slurm_cmd" 
	echo $($slurm_cmd)
done


