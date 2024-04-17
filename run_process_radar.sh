#!/bin/bash

# run processing jobs for 1km radar data.
# Takes about 3 hours per year when running interactively. Will give 12 hours!
. ./set_env_jasmin.sh # set up the environment
mkdir -p output # where output is going
years="2004 2005 2006 2007 2008 2009 2010 2011 2012 2013 2014 2015 2016 2017 2018 2019 2020 2021 2022 2023"
#years="2007 2011 2012 2013 2015 2016 2017 2020 2021 2023" # failed first time
#years="2015 2016 2017 2021" # 5km failed 2nd time.
region="52500 497500 502500 1497500" # region to process
log_level="DEBUG" # debug info
rolling="1 2 4 8" # rolling levels

RESOLN="1km"
OUTDIR="/gws/nopw/j04/edin_cesd/stett2/Scotland_extremes/radar/${RESOLN}_summary"
if [[ "$RESOLN" == '5km' ]] ; 	then
    TIME="4:00:00" # plenty of leeway. Should take 30 minutes.
    Q='short-serial'
    cmd="./process_radar_data.py --resolution $RESOLN --log_level ${log_level} --rolling ${rolling} --minsample 3 --region ${region} --outdir ${OUTDIR}"
elif [[ "$RESOLN" == '1km' ]] ; then
    TIME="36:00:00" # lots more time than we need!
    Q='high-mem --mem=100000' # high mem q with a 100 GB of RAM
    #Q='long-serial' # memory use low enoguh can get away with long Q.
    cmd="./process_radar_data.py --resolution $RESOLN --log_level ${log_level}  --coarsen 1 2 4 5 8 --rolling ${rolling} --minsample 9 --region ${region} --outdir ${OUTDIR}"
else
    echo "Only deal with 1km or 5km data"
    exit 1
fi

no_jobs=2 # how many jobs will be running at once.  
# Writing out data can run too slowly if there is I/O problems.
# though I think bottleneck is reading gzipped radar data
# this value needs to be worked out empirically.
# generate array of job names. 
# These are used with the --dependency=singleton in slurm
job_names=()
for jn in $(seq $no_jobs)
do
    name="rdr_${RESOLN}${jn}"
    job_names+=("$name")
done
jobCount=0 # index for job name
for year in $years
do
	export TMPDIR=scratch/$year
	mkdir -p $TMPDIR
	echo "Made $TMPDIR submitting jobs"

	job_name=${job_names[$jobCount]} # get jobname 
	jobCount=$(( (${jobCount}+1)%${no_jobs} )) # increment count mod no of jobs
	slurm_cmd="sbatch --dependency=singleton --job-name=${job_name} --parsable -p $Q --time=$TIME -o output/process_radar_${RESOLN}_${year}.out ${cmd} ${year}"
	#
	echo "Running: $slurm_cmd" 
	echo $($slurm_cmd)
done


