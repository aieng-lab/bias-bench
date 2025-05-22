#!/bin/bash

source "shell_jobs/_experiment_configuration.sh"
#export CUDA_VISIBLE_DEVICES=0

echo "Running SEAT experiments"
for model in ${models[@]}; do
    model_name=${model_to_model_name_or_path[${model}]}
    model_id=$(basename "$model_name")
    experiment_id="seat_m-${model}_c-${model_id}"
    if [ ! -f "${persistent_dir}/results/seat/${experiment_id}.json" ]; then
        echo ${experiment_id}
        python experiments/seat.py \
            --tests ${seat_tests} \
            --model ${model} \
            --model_name_or_path ${model_name} \
            --persistent_dir "${persistent_dir}"
    else
        echo "${experiment_id} already computed"
    fi
done



for model in ${models[@]}; do
    debiased_models=(${model_to_debiased_models[$model]})
    echo ${debiased_models[@]}
    for model_id in ${debiased_models[@]}; do
        base_model_id=$(basename "$model_id")
        experiment_id="seat_m-${model}_c-${base_model_id}"
        if [ ! -f "${persistent_dir}/results/seat/${experiment_id}.json" ]; then
            echo ${experiment_id}
            python experiments/seat.py \
                --tests ${seat_tests} \
                --model ${model} \
                --model_name_or_path ${model_id} \
                --persistent_dir "${persistent_dir}"
        else
            echo "${experiment_id} already computed"
        fi
    done
done


model_name="bert-large-cased"
model="BertModel"
experiment_id="seat_m-${model}_c-${model_name}"
if [ ! -f "${persistent_dir}/results/seat/${experiment_id}.json" ]; then
    echo ${experiment_id}
    python experiments/seat.py \
        --tests ${seat_tests} \
        --model ${model} \
        --model_name_or_path ${model_name} \
        --persistent_dir "${persistent_dir}"
else
    echo "${experiment_id} already computed"
fi


