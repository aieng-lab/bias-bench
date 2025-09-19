#!/bin/bash

source "shell_jobs/_experiment_configuration.sh"

base_models=("bert-base-cased" "bert-large-cased")
model="BertModel"
bias_types=("gender" "race" "religion")

for base_model in ${base_models[@]}; do
    for bias_type in ${bias_types[@]}; do
        experiment_id="subspace_m-${model}_c-${base_model}_t-${bias_type}"
        if [ ! -f "${persistent_dir}/results/subspace/${experiment_id}.pt" ]; then
            echo ${experiment_id}
            python experiments/sentence_debias_subspace.py \
                --model ${model} \
                --model_name_or_path ${base_model} \
                --bias_type ${bias_type} \
                --persistent_dir "${persistent_dir}"
        else
            echo "${experiment_id} already computed"
        fi
    done
done


base_models=("roberta-large")
model="RobertaModel"
for base_model in ${base_models[@]}; do
    for bias_type in ${bias_types[@]}; do
        experiment_id="subspace_m-${model}_c-${base_model}_t-${bias_type}"
        if [ ! -f "${persistent_dir}/results/subspace/${experiment_id}.pt" ]; then
            echo ${experiment_id}
            python experiments/sentence_debias_subspace.py \
                --model ${model} \
                --model_name_or_path ${base_model} \
                --bias_type ${bias_type} \
                --persistent_dir "${persistent_dir}"
        else
            echo "${experiment_id} already computed"
        fi
    done
done

base_models=("distilbert-base-cased")
model="DistilbertModel"
for base_model in ${base_models[@]}; do
    for bias_type in ${bias_types[@]}; do
        experiment_id="subspace_m-${model}_c-${base_model}_t-${bias_type}"
        if [ ! -f "${persistent_dir}/results/subspace/${experiment_id}.pt" ]; then
            echo ${experiment_id}
            python experiments/sentence_debias_subspace.py \
                --model ${model} \
                --model_name_or_path ${base_model} \
                --bias_type ${bias_type} \
                --persistent_dir "${persistent_dir}"
        else
            echo "${experiment_id} already computed"
        fi
    done
done

base_models=("gpt2")
model="GPT2Model"
for base_model in ${base_models[@]}; do
    for bias_type in ${bias_types[@]}; do
        experiment_id="subspace_m-${model}_c-${base_model}_t-${bias_type}"
        if [ ! -f "${persistent_dir}/results/subspace/${experiment_id}.pt" ]; then
            echo ${experiment_id}
            python experiments/sentence_debias_subspace.py \
                --model ${model} \
                --model_name_or_path ${base_model} \
                --bias_type ${bias_type} \
                --persistent_dir "${persistent_dir}"
        else
            echo "${experiment_id} already computed"
        fi
    done
done


base_models=(${llama_instruct_model})
model="LlamaInstructModel"
for base_model in ${base_models[@]}; do
    for bias_type in ${bias_types[@]}; do
        base_model_id=$(basename "$base_model")
        experiment_id="subspace_m-${model}_c-${base_model_id}_t-${bias_type}"
        if [ ! -f "${persistent_dir}/results/subspace/${experiment_id}.pt" ]; then
            echo ${experiment_id}
            python experiments/sentence_debias_subspace.py \
                --model ${model} \
                --model_name_or_path ${base_model} \
                --bias_type ${bias_type} \
                --persistent_dir "${persistent_dir}"
        else
            echo "${experiment_id} already computed"
        fi
    done
done

base_models=(${llama_model})
model="LlamaModel"
for base_model in ${base_models[@]}; do
    for bias_type in ${bias_types[@]}; do
        base_model_id=$(basename "$base_model")
        experiment_id="subspace_m-${model}_c-${base_model_id}_t-${bias_type}"
        if [ ! -f "${persistent_dir}/results/subspace/${experiment_id}.pt" ]; then
            echo ${experiment_id}
            python experiments/sentence_debias_subspace.py \
                --model ${model} \
                --model_name_or_path ${base_model} \
                --bias_type ${bias_type} \
                --persistent_dir "${persistent_dir}"
        else
            echo "${experiment_id} already computed"
        fi
    done
done

# run sentence debias for debiased models
for model in ${models[@]}; do
    debiased_models=(${model_to_debiased_models["${model}_${bias_type}"]})  # Split string into array
    for model_name in ${debiased_models[@]}; do
        # if model_name ends with -M or F, skip
        if [[ $model_name == *"-M" ]] || [[ $model_name == *"-F" ]]; then
            continue
        fi

        echo ${model_name}
        model_id=${model_name#$changed_models_dir"/"}
        model_id=${model_id#$checkpoint_dir"/"}
        base_model_id=$(echo "$model_id" | tr '/' '-')

        for bias_type in ${bias_types[@]}; do

            if [[ $model_name == *"cda"* ]] || [[ $model_name == *"dropout"* ]]; then
                if [[ $model_name != *"${bias_type}"* ]]; then
                    echo "Skipping model ${model_name} for bias type ${bias_type}"
                    continue
                fi
            fi

            experiment_id="subspace_m-${model}_c-${base_model_id}_t-${bias_type}"
            if [ ! -f "${persistent_dir}/results/subspace/${experiment_id}.pt" ]; then
                echo ${experiment_id}
                python experiments/sentence_debias_subspace.py \
                    --model ${model} \
                    --model_name_or_path ${model_name} \
                    --bias_type ${bias_type} \
                    --persistent_dir "${persistent_dir}"
            else
                echo "${experiment_id} already computed"
            fi
        done
    done
done
