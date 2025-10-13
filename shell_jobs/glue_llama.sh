#!/bin/bash
# this script requires the lm-evalaution-harness fork belonging to this study to be installed (see the ReadMe)

source "shell_jobs/_experiment_configuration.sh"

eval "$(conda shell.bash hook)"
conda activate lm-evaluation-harness

num_fewshot=0
tasks=("glue" "super-glue-lm-eval-v1")
seeds=(0 1 2)

bias_types=("race" "religion" "gender" )

for bias_type in ${bias_types[@]}; do
    echo "Bias type: ${bias_type}"
    echo ${model_to_debiased_models[@]}
    key="LlamaModel_${bias_type}"
    echo "Key: $key"
    debiased_llama_models=(${model_to_debiased_models["${key}"]})
    echo "Debiased LLaMA models: ${debiased_llama_models[@]}"
    models=(${llama_model} ${debiased_llama_models[@]})
    echo "Models: ${models[@]}"
    for task in "${tasks[@]}"; do
        result_dir="results/${task}_zero_shot"
        echo "Running task: ${task} (${result_dir})"

        for seed in ${seeds[@]}; do
            echo "Running seed ${seed}"

            for model in ${models[@]}; do
                echo "Running model ${model}"
                model_id=$(basename $model)
                if [[ $model_id == *"-F" || $model_id == *"-M" || $model_id == *"-N" ]]; then
                    $model_id="${model_id}-gender"
                fi
                output_path="${result_dir}/${model_id}/${seed}"

                if [[ ! -d $output_path ]]; then
                    echo $output_path
                    lm_eval \
                      --model hf \
                      --model_args pretrained=$model \
                      --tasks $task \
                      --num_fewshot $num_fewshot \
                      --batch_size 32 \
                      --output_path "${output_path}" \
                      --seed $seed \
                      --log_samples
                else
                    echo "${output_path} already exists"
                fi
                # continue the loop if the model is -F or -M
                if [[ $model_id == *"-F" || $model_id == *"-M" ]]; then
                    echo "Skipping model ${model_id}"
                    continue
                fi

                if [[ $bias_type != "gender" ]]; then
                    continue
                fi

                for projection_prefix in "${projection_matrix_prefixes[@]}"; do
                    projection_prefix_str=$projection_prefix

                    if [[ $projection_prefix != "" ]] && [[ $bias_type != "gender" ]]; then
                       continue
                    fi

                    if [[ ! -n "$projection_prefix" ]]; then
                        projection_prefix_str="inlp"
                    fi

                    output_path="${result_dir}/${model_id}-${projection_prefix_str}-${bias_type}/${seed}"
                    echo "Running model ${model} with projection prefix ${projection_prefix_str} at ${output_path}"

                    if [[ ! -d $output_path ]]; then
                        projection_matrix="${persistent_dir}/results/projection_matrix/${projection_prefix}projection_m-LlamaModel_c-${model_id}_t-${bias_type}_s-0.pt"
                        lm_eval \
                          --model hf \
                          --model_args pretrained=$model \
                          --tasks $task \
                          --num_fewshot $num_fewshot \
                          --batch_size 32 \
                          --output_path $output_path \
                          --seed $seed \
                          --log_samples \
                          --gender_debias "${projection_matrix}"
                    else
                        echo "${output_path} already exists"
                    fi
                done

                output_path="${result_dir}/${model_id}-sentence_debias-${bias_type}/${seed}"

                if [[ ! -d $output_path ]]; then
                    echo "Subspace: $subspace"
                    echo "Running model ${model} with subspace"
                    echo $output_path
                    subspace="${persistent_dir}/results/subspace/subspace_m-LlamaModel_c-${model_id}_t-${bias_type}.pt"

                    lm_eval \
                      --model hf \
                      --model_args pretrained=$model \
                      --tasks $task \
                      --num_fewshot $num_fewshot \
                      --batch_size 32 \
                      --output_path $output_path \
                      --seed $seed \
                      --log_samples \
                      --gender_debias $subspace

                else
                    echo "${output_path} already exists"
                fi
            done
        done

        key="LlamaInstructModel_${bias_type}"
        debiased_llama_instruct_models=(${model_to_debiased_models["${key}"]})
        models=(${llama_instruct_model} ${debiased_llama_instruct_models[@]})
        for seed in ${seeds[@]}; do
            echo "Running seed ${seed}"

            for model in ${models[@]}; do
                echo "Running model ${model}"
                model_id=$(basename $model)
                output_path="${result_dir}/${model_id}/${seed}"

                if [[ ! -d $output_path ]]; then
                    echo $output_path
                    lm_eval \
                      --model hf \
                      --model_args pretrained=$model \
                      --tasks $task \
                      --num_fewshot $num_fewshot \
                      --batch_size 32 \
                      --output_path $output_path \
                      --seed $seed \
                      --log_samples \
                      --apply_chat_template
                else
                    echo "${output_path} already exists"
                fi

                # continue the loop if the model is -F or -M
                if [[ $model_id == *"-F" || $model_id == *"-M" ]]; then
                    echo "Skipping model ${model_id}"
                    continue
                fi

                for projection_prefix in "${projection_matrix_prefixes[@]}"; do
                    projection_prefix_str=$projection_prefix
                    if [[ $projection_prefix != "" ]] && [[ $bias_type != "gender" ]]; then
                       continue
                    fi

                    if [[ ! -n "$projection_prefix" ]]; then
                        projection_prefix_str="inlp"
                    fi

                    echo "Running model ${model} with projection prefix ${projection_prefix_str}"

                    output_path="${result_dir}/${model_id}-${projection_prefix_str}-${bias_type}/${seed}"

                    if [[ ! -d $output_path ]]; then
                        projection_matrix="${persistent_dir}/results/projection_matrix/${projection_prefix}projection_m-LlamaInstructModel_c-${model_id}_t-${bias_type}_s-0.pt"
                        echo "Projection matrix: $projection_matrix"
                        lm_eval \
                          --model hf \
                          --model_args pretrained=$model \
                          --tasks $task \
                          --num_fewshot $num_fewshot \
                          --batch_size 32 \
                          --output_path $output_path \
                          --seed $seed \
                          --apply_chat_template \
                          --log_samples \
                          --gender_debias $projection_matrix
                    else
                        echo "${output_path} already exists"
                    fi
                done

                output_path="${result_dir}/${model_id}-sentence_debias-${bias_type}/${seed}"
                if [[ ! -d $output_path ]]; then
                    subspace="${persistent_dir}/results/subspace/subspace_m-LlamaInstructModel_c-${model_id}_t-${bias_type}.pt"
                    echo "Subspace: $subspace"
                    lm_eval \
                      --model hf \
                      --model_args pretrained=$model \
                      --tasks $task \
                      --num_fewshot $num_fewshot \
                      --batch_size 32 \
                      --output_path $output_path \
                      --seed $seed \
                      --log_samples \
                      --gender_debias $subspace \
                      --apply_chat_template
                else
                    echo "${output_path} already exists"
                fi
            done
        done
    done
done