#!/bin/bash
if [ -z "$1" ]
then
    echo "Missing a run name"
    exit 1
fi

run_name=$1

export KUBERNETES_SERVICE_HOST=''
export RUN_NAME=$run_name
# export OUTPUT_DIR='/opt/data/curious_poet/curious_poet_paper_dataset/centralized_ICM'
export OUTPUT_DIR=. #'/opt/home/data/test'

# export SKIP_WANDB=''

mkdir -p $OUTPUT_DIR/$RUN_NAME/logs

if [ -z "$2" ]
    then
        echo $OUTPUT_DIR/$RUN_NAME/logs/train_icm_stdout_$2.log
        python -u cpoet/train_icm.py 2>&1 | tee $OUTPUT_DIR/$RUN_NAME/logs/train_icm_stdout_$2.log
    else
        python -u cpoet/train_icm.py 2>&1 | tee $OUTPUT_DIR/$RUN_NAME/logs/train_icm_stdout.log
fi