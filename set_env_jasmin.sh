# setup env for running rpy2 and xarray on jasmin
# run before import gev_r or anything else that uses R
if  [[ -z "$SET_ENV_JASMIN" ]]
then
    module load jasr # load the jasmin r env.
    conda activate /home/users/tetts/CPMpy # my python + R env
    CPM_DIR=~tetts/CPM_rain_analysis
    export PATH=$CPM_DIR:$PATH:~/pycharm-community-2023.2.4/bin/ # give me pycharm
    export PYTHONPATH=$CPM_DIR:$PYTHONPATH
    SET_ENV_JASMIN="true"
else
    echo "Already set environment. Not setting again"
fi
