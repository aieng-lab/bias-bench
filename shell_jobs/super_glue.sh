#!/bin/bash

# Inputs
model=$1
size_model_type=$2
base_model=$3
echo "Running SuperGLUE for model: ${model}, size_model_type: ${size_model_type}, base_model: ${base_model}"


glue_tasks=(
    "boolq"
    "cb"
    "copa"
    "multirc"
   # "record"
    "rte"
    "wic"
    "wsc.fixed"
    #"axb"
    #"axg"
)



#glue_tasks=("record")


# call glue.sh with parameters
set -- $1 $2 $3 "super_glue"
source "shell_jobs/glue.sh"