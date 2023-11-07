#!/bin/bash
if [ -z "$1" ]
then
    echo "Missing an experiment id"
    exit 1
fi

# setup ipp(?) and log directory names
if (($# == 3));
then
  # if there are two arguments then the dirs are both arg1/arg2
  # the idea is: ./run_local_short.sh output_path run_name
  ippDir=$1/$2
  logDir=$1/$2
  echo $ippDir
else
  # setup ipp(?) and log directory names
  # in this case, with a single argument, the output_path is assumed
  ippDir=~/ipp/poet_$1
  logDir=~/logs/poet_$1
fi


export PYTHONHASHSEED=0

python -u cpoet/master.py \
  --log_file $ippDir \
  --start_from $3 \
  --logtag "" \
  --n_iterations 20 \
  2>&1 | tee $ippDir/$3.resume_run.log
