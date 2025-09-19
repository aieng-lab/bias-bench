#!/bin/bash

source "shell_jobs/_experiment_configuration.sh"


declare -A model_type_suffixes=(
    ["Bert"]="ForMaskedLM"
    ["BertLarge"]="ForMaskedLM"
    ["Roberta"]="ForMaskedLM"
    ["Distilbert"]="ForMaskedLM"
    ["GPT2"]="LMHeadModel"
    ["Llama"]="ForCausalLM"
    ["LlamaInstruct"]="ForCausalLM"
)

for model_type in ${models[@]}; do
    echo "Model type: ${model_type}"

    for bias_type in ${bias_types[@]}; do
        gradiend_debiased_models=(${model_to_debiased_gradiend_models["${model_type}_${bias_type}"]})
        for model in ${gradiend_debiased_models[@]}; do
            base_name=$(basename "$model")
            model_type_without_model=${model_type//Model/}
            model_type_suffix=${model_type_suffixes[${model_type_without_model}]}
            experiment_id="stereoset_m-${model_type_without_model}${model_type_suffix}_c-${base_name}"
            if [[ ! -f "${persistent_dir}/results/stereoset/${experiment_id}.json" ]]; then

                echo ${experiment_id}
                python experiments/stereoset.py \
                    --model "${model_type_without_model}${model_type_suffix}" \
                    --model_name_or_path ${model} \
                    --persistent_dir ${persistent_dir}
            else
                echo "${experiment_id} already computed"
            fi
        done
    done
done



for model in ${masked_lm_models[@]}; do
    model_name=${model_to_model_name_or_path[${model}]}
    #model_id=$(echo "$model_name" | tr '/' '-')
    model_id=$(basename "$model_name")
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
    model_name=${model_to_model_name_or_path[${model}]}
    base_name=$(basename "$model_name")
    experiment_id="stereoset_m-${model}_c-${base_name}"
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

exit


for model in ${debiased_llama_instruct_models[@]}; do
    base_name=$(basename "$model")
    experiment_id="stereoset_m-LlamaInstructForCausalLM_c-${base_name}"
    if [ ! -f "${persistent_dir}/results/stereoset/${experiment_id}.json" ]; then
        echo ${experiment_id}
        python experiments/stereoset.py \
            --model "LlamaInstructForCausalLM" \
            --model_name_or_path ${model} \
            --persistent_dir ${persistent_dir}
    else
        echo "${experiment_id} already computed"
    fi
done

for model in ${debiased_llama_models[@]}; do
    base_name=$(basename "$model")
    experiment_id="stereoset_m-LlamaForCausalLM_c-${base_name}"
    if [ ! -f "${persistent_dir}/results/stereoset/${experiment_id}.json" ]; then
        echo ${experiment_id}
        python experiments/stereoset.py \
            --model "LlamaForCausalLM" \
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


exit

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




