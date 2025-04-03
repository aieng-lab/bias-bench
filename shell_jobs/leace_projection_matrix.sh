#!/bin/bash

source "shell_jobs/_experiment_configuration.sh"
#export CUDA_VISIBLE_DEVICES=0

bert_base_models=("bert-base-cased" "bert-large-cased")
bias_types=("gender")
model="BertModel"
for base_model in ${bert_base_models[@]}; do
    for bias_type in ${bias_types[@]}; do
        experiment_id="leace_projection_m-${model}_c-${base_model}_t-${bias_type}_s-0"
        if [ ! -f "${persistent_dir}/results/projection_matrix/${experiment_id}.pt" ]; then
            echo ${experiment_id}
            python experiments/leace_projection_matrix.py \
                --model ${model} \
                --model_name_or_path ${base_model} \
                --bias_type ${bias_type} \
                --seed 0 \
                --persistent_dir ${persistent_dir}
        else
            echo "${experiment_id} already computed"
        fi
    done
done

other_models=("RobertaModel" "DistilbertModel" "GPT2Model")
for model in ${other_models[@]}; do
    base_model=${model_to_model_name_or_path[${model}]}
    model_id=$(echo $base_model | sed 's/\//-/g')
    for bias_type in ${bias_types[@]}; do

        experiment_id="leace_projection_m-${model}_c-${model_id}_t-${bias_type}_s-0"
        if [ ! -f "${persistent_dir}/results/projection_matrix/${experiment_id}.pt" ]; then
            echo ${experiment_id}
            python experiments/leace_projection_matrix.py \
                --model ${model} \
                --model_name_or_path ${base_model} \
                --bias_type ${bias_type} \
                --seed 0 \
                --persistent_dir ${persistent_dir}
        else
            echo "${experiment_id} already computed"
        fi
    done
done

# run leace for debiased models
for model in ${models[@]}; do
    debiased_models=(${model_to_debiased_models[$model]})  # Split string into array
    for model_name in ${debiased_models[@]}; do
        # remove prefix persistent_dir
        echo ${model_name}
        model_id=${model_name#$changed_models_dir"/"}
        model_id=${model_id#$checkpoint_dir"/"}
        base_model_id=$(echo "$model_id" | tr '/' '-')

        for bias_type in ${bias_types[@]}; do
            experiment_id="leace_projection_m-${model}_c-${base_model_id}_t-${bias_type}_s-0"
            if [ ! -f "${persistent_dir}/results/projection_matrix/${experiment_id}.pt" ]; then
                echo ${experiment_id}
                python experiments/leace_projection_matrix.py \
                    --model ${model} \
                    --model_name_or_path ${model_name} \
                    --bias_type ${bias_type} \
                    --seed 0 \
                    --persistent_dir ${persistent_dir}
            else
                echo "${experiment_id} already computed"
            fi
        done
    done
done
