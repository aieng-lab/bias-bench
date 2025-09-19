#!/bin/bash

source "shell_jobs/_experiment_configuration.sh"


bias_types=("race" "religion" "gender")

seed=$1
echo "Using seed ${seed}"
seeds=(${seed})

set -- "roberta-large" "Roberta" "roberta-large"
source "shell_jobs/super_glue.sh"

set -- "gpt2" "GPT2" "gpt2"
source "shell_jobs/super_glue.sh"

set -- "bert-base-cased" "Bert" "bert-base-cased"
source "shell_jobs/super_glue.sh"

set -- "distilbert-base-cased" "Distilbert" "distilbert-base-cased"
source "shell_jobs/super_glue.sh"

set -- "bert-large-cased" "BertLarge" "bert-large-cased"
source "shell_jobs/super_glue.sh"


# compute the bootstrap scores for SuperGLUE
python experiments/glue_bootstrap_evaluation.py