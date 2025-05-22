#!/bin/bash

source "shell_jobs/_experiment_configuration.sh"
#export CUDA_VISIBLE_DEVICES=0

bias_types=("gender")

for model in ${inlp_models[@]}; do
    for projection_prefix in "${projection_matrix_prefixes[@]}"; do
        echo "Projection prefix: $projection_prefix"
        model_name=${model_to_model_name_or_path[${model}]}
        model_id=$(basename "${model_name}")
        for bias_type in ${bias_types[@]}; do
            experiment_id="seat_m-${projection_prefix}${model}_c-${model_id}_t-${bias_type}"

            if [ ! -f "${persistent_dir}/results/seat/${experiment_id}.json" ]; then
                echo ${experiment_id}
                cmd=(python experiments/seat_debias.py \
                    --tests ${seat_tests} \
                    --model ${model} \
                    --model_name_or_path ${model_to_model_name_or_path[${model}]} \
                    --projection_matrix "${persistent_dir}/results/projection_matrix/${projection_prefix}projection_m-${debiased_model_to_base_model[${model}]}_c-${model_id}_t-${bias_type}_s-0.pt" \
                    --bias_type ${bias_type} \
                    --persistent_dir "${persistent_dir}")

                if [[ -n "$projection_prefix" ]]; then
                    cmd+=(--projection_prefix "$projection_prefix")
                fi
                "${cmd[@]}"
            else
                echo "${experiment_id} already computed"
            fi
        done

        base_model=${debiased_model_to_base_model[${model}]}
        debiased_models=(${model_to_debiased_models[$base_model]})
        for debiased_model in ${debiased_models[@]}; do

            if [[ $debiased_model != *"-M" ]] && [[ $debiased_model != *"-F" ]]; then
                echo "Model: $model"
                model_id=$(basename "$debiased_model")

                for bias_type in ${bias_types[@]}; do
                    experiment_id="seat_m-${projection_prefix}${model}_c-${model_id}_t-${bias_type}"
                    if [ ! -f "${persistent_dir}/results/seat/${experiment_id}.json" ]; then
                        echo ${experiment_id}
                        cmd=(python experiments/seat_debias.py \
                            --tests ${seat_tests} \
                            --model ${model} \
                            --model_name_or_path ${debiased_model} \
                            --projection_matrix "${persistent_dir}/results/projection_matrix/${projection_prefix}projection_m-${base_model}_c-${model_id}_t-${bias_type}_s-0.pt" \
                            --bias_type ${bias_type} \
                            --persistent_dir "${persistent_dir}")

                        if [[ -n "$projection_prefix" ]]; then
                            cmd+=(--projection_prefix "$projection_prefix")
                        fi
                        "${cmd[@]}"
                    else
                        echo "${experiment_id} already computed"
                    fi
                done
            fi
        done
    done
done

for model in ${sentence_debias_models[@]}; do
    for bias_type in ${bias_types[@]}; do
        experiment_id="seat_m-${model}_c-${model_to_model_name_or_path[${model}]}_t-${bias_type}"
        if [ ! -f "${persistent_dir}/results/seat/${experiment_id}.json" ]; then
            echo ${experiment_id}
            model_id=$(basename ${model_to_model_name_or_path[${model}]})
            python experiments/seat_debias.py \
                --tests ${seat_tests} \
                --model ${model} \
                --model_name_or_path ${model_to_model_name_or_path[${model}]} \
                --bias_direction "${persistent_dir}/results/subspace/subspace_m-${debiased_model_to_base_model[${model}]}_c-${model_id}_t-${bias_type}.pt" \
                --bias_type ${bias_type} \
                --persistent_dir "${persistent_dir}"
        else
            echo "${experiment_id} already computed"
        fi
    done

    base_model=${debiased_model_to_base_model[${model}]}
    debiased_models=(${model_to_debiased_models[$base_model]})
    for debiased_model in ${debiased_models[@]}; do
        if [[ $debiased_model != *"-M" ]] && [[ $debiased_model != *"-F" ]]; then
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
