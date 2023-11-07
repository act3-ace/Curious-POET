#!/bin/bash
if [ -z "$1" ]
then
    echo "Missing an experiment id"
    exit 1
fi

# setup ipp(?) and log directory names
if (($# == 2));
then
  # if there are two arguments then the dirs are both arg1/arg2
  # the idea is: ./run_peot_local_large_icm.sh output_path run_name
  ippDir=$1/$2
  logDir=$1/$2
else
  # setup ipp(?) and log directory names
  # in this case, with a single argument, the output_path is assumed
  ippDir=~/ipp/poet_$1
  logDir=~/logs/poet_$1
fi

# make directories
mkdir -p $ippDir
mkdir -p $logDir

export PYTHONHASHSEED=0

config_file=test_params_large_icm.yml
cp $config_file $logDir/$config_file

python -u cpoet/master.py \
  --log_file $logDir \
  --config $config_file \
  2>&1 | tee $ippDir/run.log
