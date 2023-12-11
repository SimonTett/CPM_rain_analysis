#!/bin/bash
# run combination jobs.  Before running make sure you have done 
# . ~/CPM_rain_analysis/set_env_jasmin.sh
. ./set_env_jasmin.sh
dirs=/gws/nopw/j04/edin_cesd/stett2/summary_1km/1km
#dirs="summary_5km_15Min"
Q='high-mem --mem=100000' # high mem q with a 100 GB of RAM
time="12:00:00" # need lots of time for 1km case.
for dir in $dirs
do
    dir_name=$(basename $dir)
    cmd="combine_summary.py --verbose ${dir}  ${dir}_summary.nc --region 287000 488000 685000 886000"
    cmd="sbatch --export=ALL -p $Q --time=$time -o output_summary/${dir_name}.out ${cmd}"
    echo "$cmd"
    output=$($cmd)
    echo "Status: $? Output: $output"
done
