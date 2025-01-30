#!/bin/bash

#export CUDA_VISIBLE_DEVICES=0
source "shell_jobs/_experiment_configuration.sh"

set -- "bert-large-cased" "BertLarge" "bert-large-cased"
source "shell_jobs/glue.sh"
set -- "${gradiend_dir}/results/changed_models/bert-large-cased-N" "BertLarge" "bert-large-cased"
source "shell_jobs/glue.sh"
set -- "${gradiend_dir}/results/changed_models/bert-large-cased-F" "BertLarge" "bert-large-cased"
source "shell_jobs/glue.sh"
set -- "${gradiend_dir}/results/changed_models/bert-large-cased-M" "BertLarge" "bert-large-cased"
source "shell_jobs/glue.sh"

set -- "distilbert-base-cased" "Distilbert" "distilbert-base-cased"
source "shell_jobs/glue.sh"
set -- "${gradiend_dir}/results/changed_models/distilbert-base-cased-F" "Distilbert" "distilbert-base-cased"
source "shell_jobs/glue.sh"
set -- "${gradiend_dir}/results/changed_models/distilbert-base-cased-N" "Distilbert" "distilbert-base-cased"
source "shell_jobs/glue.sh"
set -- "${gradiend_dir}/results/changed_models/distilbert-base-cased-M" "Distilbert" "distilbert-base-cased"
source "shell_jobs/glue.sh"

set -- "roberta-large" "Roberta" "roberta-large"
source "shell_jobs/glue.sh"
set -- "${gradiend_dir}/results/changed_models/roberta-large-N" "Roberta" "roberta-large"
source "shell_jobs/glue.sh"
set -- "${gradiend_dir}/results/changed_models/roberta-large-M" "Roberta" "roberta-large"
source "shell_jobs/glue.sh"
set -- "${gradiend_dir}/results/changed_models/roberta-large-F" "Roberta" "roberta-large"
source "shell_jobs/glue.sh"

set -- "bert-base-cased" "Bert" "bert-base-cased"
source "shell_jobs/glue.sh"
set -- "${gradiend_dir}/results/changed_models/bert-base-cased-N" "Bert" "bert-base-cased"
source "shell_jobs/glue.sh"
set -- "${gradiend_dir}/results/changed_models/bert-base-cased-F" "Bert" "bert-base-cased"
source "shell_jobs/glue.sh"
set -- "${gradiend_dir}/results/changed_models/bert-base-cased-M" "Bert" "bert-base-cased"
source "shell_jobs/glue.sh"

# compute the bootstrap scores for GLUE
python experiments/glue_bootstrap_evaluation.py