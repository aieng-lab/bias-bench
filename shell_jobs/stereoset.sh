#!/bin/bash

source "shell_jobs/_experiment_configuration.sh"
#export CUDA_VISIBLE_DEVICES=0

tested_masked_lm_models=(
  male_model,
  femal_model,
  unbiased_model
)

for model in ${masked_lm_models[@]}; do
    model_name=${model_to_model_name_or_path[${model}]}
    model_id=$(echo "$model_name" | tr '/' '-')
    experiment_id="stereoset_m-${model}_c-${model_id}"
    if [ ! -f "${persistent_dir}/results/stereoset/${experiment_id}.json" ]; then
        echo ${experiment_id}
        python experiments/stereoset.py \
            --model ${model} \
            --model_name_or_path ${model_name} \
            --persistent_dir ${persistent_dir}
    else
      echo "${experiment_id} already computed"
    fi
done

for model in ${causal_lm_models[@]}; do
    experiment_id="stereoset_m-${model}_c-${model_to_model_name_or_path[${model}]}"
    if [ ! -f "${persistent_dir}/results/stereoset/${experiment_id}.json" ]; then
        echo ${experiment_id}
        python experiments/stereoset.py \
            --model ${model} \
            --model_name_or_path ${model_to_model_name_or_path[${model}]} \
            --persistent_dir ${persistent_dir}
    else
        echo "${experiment_id} already computed"
    fi
done


for model in ${debiased_distilbert_models[@]}; do
    base_name=$(basename "$model")
    experiment_id="stereoset_m-DistilbertForMaskedLM_c-${base_name}"
    if [ ! -f "${persistent_dir}/results/stereoset/${experiment_id}.json" ]; then
        echo ${experiment_id}
        python experiments/stereoset.py \
            --model "DistilbertForMaskedLM" \
            --model_name_or_path ${model} \
            --persistent_dir ${persistent_dir}
    else
        echo "${experiment_id} already computed"
    fi
done

for model in ${debiased_bert_models[@]}; do
    base_name=$(basename "$model")
    experiment_id="stereoset_m-BertForMaskedLM_c-${base_name}"
    if [ ! -f "${persistent_dir}/results/stereoset/${experiment_id}.json" ]; then
        echo ${experiment_id}
        python experiments/stereoset.py \
            --model "BertForMaskedLM" \
            --model_name_or_path ${model} \
            --persistent_dir ${persistent_dir}
    else
        echo "${experiment_id} already computed"
    fi
done


for model in ${debiased_roberta_models[@]}; do
    base_name=$(basename "$model")
    experiment_id="stereoset_m-RobertaForMaskedLM_c-${base_name}"
    if [ ! -f "${persistent_dir}/results/stereoset/${experiment_id}.json" ]; then
        echo ${experiment_id}
        python experiments/stereoset.py \
            --model "RobertaForMaskedLM" \
            --model_name_or_path ${model} \
            --persistent_dir ${persistent_dir}
    else
        echo "${experiment_id} already computed"
    fi
done


for model in ${debiased_gpt2_models[@]}; do
    base_name=$(basename "$model")
    experiment_id="stereoset_m-GPT2LMHeadModel_c-${base_name}"
    if [ ! -f "${persistent_dir}/results/stereoset/${experiment_id}.json" ]; then
        echo ${experiment_id}
        python experiments/stereoset.py \
            --model "GPT2LMHeadModel" \
            --model_name_or_path ${model} \
            --persistent_dir ${persistent_dir}
    else
        echo "${experiment_id} already computed"
    fi
done
