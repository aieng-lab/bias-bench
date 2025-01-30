#!/bin/bash

source "shell_jobs/_experiment_configuration.sh"
#export CUDA_VISIBLE_DEVICES=0

bias_types=("gender")

for model in ${sentence_debias_models[@]}; do
    for bias_type in ${bias_types[@]}; do
        experiment_id="seat_m-${model}_c-${model_to_model_name_or_path[${model}]}_t-${bias_type}"
        if [ ! -f "${persistent_dir}/results/seat/${experiment_id}.json" ]; then
            echo ${experiment_id}
            python experiments/seat_debias.py \
                --tests ${seat_tests} \
                --model ${model} \
                --model_name_or_path ${model_to_model_name_or_path[${model}]} \
                --bias_direction "${persistent_dir}/results/subspace/subspace_m-${debiased_model_to_base_model[${model}]}_c-${model_to_model_name_or_path[${model}]}_t-${bias_type}.pt" \
                --bias_type ${bias_type} \
                --persistent_dir "${persistent_dir}"
        else
            echo "${experiment_id} already computed"
        fi
    done


    base_model=${debiased_model_to_base_model[${model}]}
    debiased_models=(${model_to_debiased_models[$base_model]})
    for debiased_model in ${debiased_models[@]}; do
        if [[ $debiased_model == *"-N" ]] || [[ $debiased_model == *"-F" ]]; then
            echo "Model ends with -N or -F"
            echo "Model: $model"
            model_id=$(basename "$debiased_model")

            for bias_type in ${bias_types[@]}; do
                experiment_id="seat_m-${model}_c-${model_id}_t-${bias_type}"
                if [ ! -f "${persistent_dir}/results/seat/${experiment_id}.json" ]; then
                    echo ${experiment_id}
                    python experiments/seat_debias.py \
                        --tests ${seat_tests} \
                        --model ${model} \
                        --model_name_or_path ${debiased_model} \
                        --bias_direction "${persistent_dir}/results/subspace/subspace_m-${base_model}_c-${model_id}_t-${bias_type}.pt" \
                        --bias_type ${bias_type} \
                        --persistent_dir "${persistent_dir}"
                else
                    echo "${experiment_id} already computed"
                fi
            done
        fi
    done
done



for model in ${inlp_models[@]}; do
    model_name=${model_to_model_name_or_path[${model}]}
    model_id=$(echo "$model_name" | tr '/' '-')
    for bias_type in ${bias_types[@]}; do
        experiment_id="seat_m-${model}_c-${model_id}_t-${bias_type}"
        if [ ! -f "${persistent_dir}/results/seat/${experiment_id}.json" ]; then
            echo ${experiment_id}
            python experiments/seat_debias.py \
                --tests ${seat_tests} \
                --model ${model} \
                --model_name_or_path ${model_to_model_name_or_path[${model}]} \
                --projection_matrix "${persistent_dir}/results/projection_matrix/projection_m-${debiased_model_to_base_model[${model}]}_c-${model_id}_t-${bias_type}_s-0.pt" \
                --bias_type ${bias_type} \
                --persistent_dir "${persistent_dir}"
        else
            echo "${experiment_id} already computed"
        fi
    done
done

for model in ${inlp_models[@]}; do
    base_model=${debiased_model_to_base_model[${model}]}
    debiased_models=(${model_to_debiased_models[$base_model]})
    for debiased_model in ${debiased_models[@]}; do

        if [[ $debiased_model == *"-N" ]]; then
            echo "Model ends with -N"
            echo "Model: $model"

            model_id=$(basename "$debiased_model")

            for bias_type in ${bias_types[@]}; do
                experiment_id="seat_m-${model}_c-${model_id}_t-${bias_type}"
                if [ ! -f "${persistent_dir}/results/seat/${experiment_id}.json" ]; then
                    echo ${experiment_id}
                    python experiments/seat_debias.py \
                        --tests ${seat_tests} \
                        --model ${model} \
                        --model_name_or_path ${debiased_model} \
                        --projection_matrix "${persistent_dir}/results/projection_matrix/projection_m-${base_model}_c-${model_id}_t-${bias_type}_s-0.pt" \
                        --bias_type ${bias_type} \
                        --persistent_dir "${persistent_dir}"
                else
                    echo "${experiment_id} already computed"
                fi
            done
        fi
    done
done



for model in ${cda_models[@]}; do
    for bias_type in ${bias_types[@]}; do
        experiment_id="seat_m-${model}_c-${model_to_model_name_or_path[${model}]}_t-${bias_type}"
        if [ ! -f "${persistent_dir}/results/seat/${experiment_id}.json" ]; then
            echo ${experiment_id}
            python experiments/seat_debias.py \
                --tests ${seat_tests} \
                --model ${model} \
                --model_name_or_path ${model_to_model_name_or_path[${model}]} \
                --load_path "${persistent_dir}/results/checkpoints/cda_c-${model_to_model_name_or_path[${model}]}_t-${bias_type}_s-0" \
                --bias_type ${bias_type} \
                --persistent_dir "${persistent_dir}"
            else
              echo "${experiment_id} already computed"
        fi
    done
done


for model in ${dropout_models[@]}; do
    experiment_id="seat_m-${model}_c-${model_to_model_name_or_path[${model}]}"
    if [ ! -f "${persistent_dir}/results/seat/${experiment_id}.json" ]; then
        echo ${experiment_id}
        python experiments/seat_debias.py \
            --tests ${seat_tests} \
            --model ${model} \
            --model_name_or_path ${model_to_model_name_or_path[${model}]} \
            --load_path "${persistent_dir}/results/checkpoints/dropout_c-${model_to_model_name_or_path[${model}]}_s-0" \
            --persistent_dir "${persistent_dir}"
        else
            echo "${experiment_id} already computed"
        fi
done
