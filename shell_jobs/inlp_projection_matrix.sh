#!/bin/bash

source "shell_jobs/_experiment_configuration.sh"

other_models=("RobertaModel" "DistilbertModel" "GPT2Model" "LlamaModel" "LlamaInstructModel")
for model in ${other_models[@]}; do
    base_model=${model_to_model_name_or_path[${model}]}
    model_id=$(basename ${base_model})
    for bias_type in ${bias_types[@]}; do

        experiment_id="projection_m-${model}_c-${model_id}_t-${bias_type}_s-0"
        if [ ! -f "${persistent_dir}/results/projection_matrix/${experiment_id}.pt" ]; then
            echo ${experiment_id}
            python experiments/inlp_projection_matrix.py \
                --model ${model} \
                --model_name_or_path ${base_model} \
                --bias_type ${bias_type} \
                --n_classifiers ${model_to_n_classifiers[${model}]} \
                --seed 0 \
                --persistent_dir ${persistent_dir}
        else
            echo "${experiment_id} already computed"
        fi
    done
done


bert_base_models=("bert-base-cased" "bert-large-cased")
bias_types=("gender")
model="BertModel"
for base_model in ${bert_base_models[@]}; do
    for bias_type in ${bias_types[@]}; do
        experiment_id="projection_m-${model}_c-${base_model}_t-${bias_type}_s-0"
        if [ ! -f "${persistent_dir}/results/projection_matrix/${experiment_id}.pt" ]; then
            echo ${experiment_id}
            python experiments/inlp_projection_matrix.py \
                --model ${model} \
                --model_name_or_path ${base_model} \
                --bias_type ${bias_type} \
                --n_classifiers ${model_to_n_classifiers[${model}]} \
                --seed 0 \
                --persistent_dir ${persistent_dir}
        else
            echo "${experiment_id} already computed"
        fi
    done
done


# run inlp for debiased models
for model in ${models[@]}; do
    debiased_models=(${model_to_debiased_models[$model]})  # Split string into array

    for model_name in ${debiased_models[@]}; do
        # only continue for -N models
        if [[ $model_name != *"-N" ]]; then
            continue
        fi

        # remove prefix persistent_dir
        echo ${model_name}
        model_id=${model_name#$changed_models_dir"/"}
        model_id=${model_id#$checkpoint_dir"/"}
        base_model_id=$(basename "$model_id")

        for bias_type in ${bias_types[@]}; do
            experiment_id="projection_m-${model}_c-${base_model_id}_t-${bias_type}_s-0"
            if [ ! -f "${persistent_dir}/results/projection_matrix/${experiment_id}.pt" ]; then
                echo ${experiment_id}
                python experiments/inlp_projection_matrix.py \
                    --model ${model} \
                    --model_name_or_path ${model_name} \
                    --bias_type ${bias_type} \
                    --n_classifiers ${model_to_n_classifiers[${model}]} \
                    --seed 0 \
                    --persistent_dir ${persistent_dir}
            else
                echo "${experiment_id} already computed"
            fi
        done
    done
done