#!/bin/bash

source "shell_jobs/_experiment_configuration.sh"


#bias_types=("race")
#bias_types=("religion")

#seeds=(2 1 0)

glue_tasks_=(
    #"wnli"
    #"stsb"
    #"sst2"
    #"cola"
    "qqp"
    #"rte"
    #"mrpc"
    #"mnli"
    #"qnli"
)

seed=$1
echo "Using seed ${seed}"
seeds=(${seed})


set -- "gpt2" "GPT2" "gpt2"
source "shell_jobs/glue.sh"
set -- "distilbert-base-cased" "Distilbert" "distilbert-base-cased"
source "shell_jobs/glue.sh"

set -- "bert-base-cased" "Bert" "bert-base-cased"
source "shell_jobs/glue.sh"
set -- "bert-large-cased" "BertLarge" "bert-large-cased"
source "shell_jobs/glue.sh"




set -- "roberta-large" "Roberta" "roberta-large"
source "shell_jobs/glue.sh"

exit










exit


# compute the bootstrap scores for GLUE
python experiments/glue_bootstrap_evaluation.py
