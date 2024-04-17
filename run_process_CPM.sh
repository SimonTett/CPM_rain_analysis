#!/bin/bash
. ./set_env_jasmin.sh # setup output
out_logdir=output_reprocess  # where output is going
mkdir -p $out_logdir
# run processing jobs for CPM ensemble. Filters data -- which is expensive.
# Takes about 40 minutes to do a decade worth of summers. So expected ~ 3 hours
# to do a decade of all seasons. 
# Needs about 2-4 min/season to filter so about an hour 45 mins  to filter decade. 
# processing after that fairly quick! (becasue data loaded).
rgn="357.5 361 1.5 7.5" # scottish region
TIME='6:00:00' # time job will be in the Q -- takes ages to process when filtering as filtering is slow...
Q='high-mem --mem=100000' # high mem q with a 100 GB of RAM -- needs lots of memory esp when writeout happens. Needed for processing a decade
#Q='short-serial --mem=63000' # standard q with 63 GB of RAM -- works if processing one season
# jobs for one ensemble are submitted dependant on the previous job.
# that avoids I/O contention on the disk space and painfully slow processing
# multiple ensemnles being processed at once might cause problems...
sleep_time=120 # sleep for two minutes between ensembles
sleep_time=1
for dir in /badc/ukcp18/data/land-cpm/uk/2.2km/rcp85/[0-9][0-9]/pr/1hr/latest 
do
    echo "Processing $dir"
    job_id='' # reset job_id to empty. Means ensembles run in parallel.
    ens=$(echo $dir | sed -E  s'!^.*rcp85/([0-9][0-9]*)/.*!\1!')
    outdir="/gws/nopw/j04/edin_cesd/stett2/Scotland_extremes/CPM_scotland_filter/CPM"$ens
    outdir="/gws/nopw/j04/edin_cesd/stett2/Scotland_extremes/CPM_scotland/CPM"$ens
    mkdir -p $outdir
    decs=$(ls $dir| cut -f8 -d"_" | cut -c1-3|grep 19[8,9]|sort|uniq) # the grep 198 restricts to 1980 -1990 data. 
    for dec in $decs
    do
	decp=$(($dec+1))
	range=$dec"0-12-01 "
	range=$range$decp"0-11-30"
	#range=$range$dec"1-11-30" # testing
	r1=$dec"0"
	# filtered
	cmd="./ens_seas_max.py  $dir/pr_*_1hr_$dec[0-9]*.nc $dir/pr_*_1hr_$decp[0]*.nc --region $rgn -o $outdir/CPM_pr_seas_max_ens$ens --monitor --verbose  --coarsen 2 --range $range --filter_bad --no_overwrite --rolling 2 4 8"
	outfile=output/"ens_seas_max_"$ens"_"$r1".out"
	# non filtered!
	cmd="./ens_seas_max.py  $dir/pr_*_1hr_$dec[0-9]*.nc $dir/pr_*_1hr_$decp[0]*.nc --region $rgn -o $outdir/CPM_pr_seas_max_ens$ens --monitor --verbose  --coarsen 2 --range $range --no_overwrite --rolling 2 4 8"
	outfile=${out_logdir}/"nofilt_ens_seas_max_"$ens"_"$r1".out"
	echo "cmd is $cmd"

	# interactive
	#run_cmd="${cmd} |& tee  ${outfile}" 
	#echo $run_cmd
	#val $run_cmd
	slurm_cmd="sbatch  --parsable -p $Q -t $TIME -o  $outfile "
	if  [[ -n $job_id ]]
	then
	    slurm_cmd=${slurm_cmd}" --dependency=afterany:"${job_id}
	fi
	slurm_cmd=${slurm_cmd}" "${cmd}
	echo "Running: $slurm_cmd"
	job_id=$($slurm_cmd)
	echo "job_id is $job_id"
    done
    echo "Sleeping for $sleep_time secs"
    sleep $sleep_time # try and avoid having all jobs at the same time

done
