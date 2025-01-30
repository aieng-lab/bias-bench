#!/bin/bash

export CUDA_VISIBLE_DEVICES=1



set -- "/srv/data/drechsel/Git/gradient/results/changed_models/roberta-large-N" "Roberta" "roberta-large"
source "shell_jobs/glue.sh"
set -- "/srv/data/drechsel/Git/gradient/results/changed_models/roberta-large-M" "Roberta" "roberta-large"
source "shell_jobs/glue.sh"
set -- "/srv/data/drechsel/Git/gradient/results/changed_models/roberta-large-F" "Roberta" "roberta-large"
source "shell_jobs/glue.sh"


set -- "/srv/data/drechsel/Git/gradient/results/changed_models/bert-base-cased-N" "Bert" "bert-base-cased"
source "shell_jobs/glue.sh"
set -- "bert-base-cased" "Bert" "bert-base-cased"
source "shell_jobs/glue.sh"
set -- "/srv/data/drechsel/Git/gradient/results/changed_models/bert-base-cased-F" "Bert" "bert-base-cased"
source "shell_jobs/glue.sh"
set -- "/srv/data/drechsel/Git/gradient/results/changed_models/bert-base-cased-M" "Bert" "bert-base-cased"
source "shell_jobs/glue.sh"

set -- "roberta-large" "Roberta" "roberta-large"
source "shell_jobs/glue.sh"

