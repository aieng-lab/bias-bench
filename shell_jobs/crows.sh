#!/bin/bash

source "shell_jobs/_experiment_configuration.sh"


for model in ${causal_lm_models[@]}; do
    for bias_type in ${bias_types[@]}; do
        model_name=${model_to_model_name_or_path[${model}]}
        model_id=$(basename "${model_name}")
        experiment_id="crows_m-${model}_c-${model_id}_t-${bias_type}"
        echo ${experiment_id}
        file="${persistent_dir}/results/crows/${experiment_id}.json"
        echo "${file}"
        # print if it is file
        if [ -f "${file}" ]; then
            echo "${experiment_id} already computed"

        fi
        if [ ! -f "${persistent_dir}/results/crows/${experiment_id}.json" ]; then
            echo ${experiment_id}
            python experiments/crows.py \
                --model ${model} \
                --model_name_or_path ${model_name} \
                --bias_type ${bias_type} \
                --persistent_dir "${persistent_dir}"
        else
            echo "${experiment_id} already computed"
        fi
    done


    if [[ $model == *"LMHeadModel"* ]]; then
        base_model="${model%LMHeadModel}Model"
    else
        base_model="${model%ForCausalLM}Model"
    fi

    debiased_models=(${model_to_debiased_models[$base_model]})
    for model_id in ${debiased_models[@]}; do
        base_model_id=$(basename "$model_id")

        for bias_type in ${bias_types[@]}; do
            experiment_id="crows_m-${model}_c-${base_model_id}_t-${bias_type}"
            if [ ! -f "${persistent_dir}/results/crows/${experiment_id}.json" ]; then
                echo ${experiment_id}
                python experiments/crows.py \
                    --model ${model} \
                    --model_name_or_path ${model_id} \
                    --bias_type ${bias_type} \
                    --persistent_dir "${persistent_dir}"
            else
                echo "${experiment_id} already computed"
            fi
        done
    done
done


for model in ${masked_lm_models[@]}; do
    for bias_type in ${bias_types[@]}; do
        experiment_id="crows_m-${model}_c-${model_to_model_name_or_path[${model}]}_t-${bias_type}"
        if [ ! -f "${persistent_dir}/results/crows/${experiment_id}.json" ]; then
            echo ${experiment_id}
            python experiments/crows.py \
                --model ${model} \
                --model_name_or_path ${model_to_model_name_or_path[${model}]} \
                --bias_type ${bias_type} \
                --persistent_dir "${persistent_dir}"
        else
            echo "${experiment_id} already computed"
        fi
    done

    base_model="${model%ForMaskedLM}Model"
    debiased_models=(${model_to_debiased_models[$base_model]})
    for model_id in ${debiased_models[@]}; do
        base_model_id=$(basename "$model_id")

        for bias_type in ${bias_types[@]}; do
            experiment_id="crows_m-${model}_c-${base_model_id}_t-${bias_type}"
            if [ ! -f "${persistent_dir}/results/crows/${experiment_id}.json" ]; then
                echo ${experiment_id}
                python experiments/crows.py \
                    --model ${model} \
                    --model_name_or_path ${model_id} \
                    --bias_type ${bias_type} \
                    --persistent_dir "${persistent_dir}"
            else
                echo "${experiment_id} already computed"
            fi
        done
    done
done

