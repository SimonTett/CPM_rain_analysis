#!/bin/bash
# combine processed monthly data
module load jasmin-sci
RADAR_DIR="/gws/nopw/j04/edin_cesd/stett2/Scotland_extremes/radar"
SUMMARY_DIR=$RADAR_DIR/summary
for pattern in 5km 1km 1km_c2 1km_c4 1km_c5 1km_c8; do
    files=$(echo "${RADAR_DIR}/*/radar_rain_200[89]-*${pattern}.nc" "${RADAR_DIR}/*/radar_rain_20[1-9][0-9]-*${pattern}.nc")
    mkdir -p ${SUMMARY_DIR}
    output="${SUMMARY_DIR}/summary_2008_${pattern}.nc"
    cmd="ncrcat -O ${files} ${output}"
    echo "$(echo $files | wc -w) files will be processed to ${output}"
    #echo $cmd
echo $($cmd)
done
