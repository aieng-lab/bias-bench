#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
source "shell_jobs/_experiment_configuration.sh"

set -- "bert-base-cased" "Bert" "bert-base-cased"
source "shell_jobs/glue.sh"

set -- "bert-large-cased" "BertLarge" "bert-large-cased"
source "shell_jobs/glue.sh"

set -- "distilbert-base-cased" "Distilbert" "distilbert-base-cased"
source "shell_jobs/glue.sh"

set -- "roberta-large" "Roberta" "roberta-large"
source "shell_jobs/glue.sh"

set -- "gpt2" "GPT2" "gpt2"
source "shell_jobs/glue.sh"

# compute the bootstrap scores for GLUE
python experiments/glue_bootstrap_evaluation.py