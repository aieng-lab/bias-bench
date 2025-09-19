#!/bin/bash

source "shell_jobs/_experiment_configuration.sh"


for bias_type in ${bias_types[@]}; do

    for model in ${cda_masked_lm_models[@]}; do
        model_name=${model_to_model_name_or_path[${model}]}
        model_id=$(basename "${model_name}")
        experiment_id="crows_m-${model}_c-${model_id}_t-${bias_type}"
        if [ ! -f "${persistent_dir}/results/crows/${experiment_id}.json" ]; then
            echo ${experiment_id}
            python experiments/crows_debias.py \
                --model ${model} \
                --model_name_or_path ${model_name} \
                --load_path "${persistent_dir}/results/checkpoints/cda_c-${model_id}_t-${bias_type}_s-0" \
                --bias_type ${bias_type} \
                --persistent_dir "${persistent_dir}"
        else
            echo "${experiment_id} already computed"
        fi
    done


    for model in ${dropout_masked_lm_models[@]}; do
        experiment_id="crows_m-${model}_c-${model_to_model_name_or_path[${model}]}_t-${bias_type}"
        if [ ! -f "${persistent_dir}/results/crows/${experiment_id}.json" ]; then
            echo ${experiment_id}
            python experiments/crows_debias.py \
                --model ${model} \
                --model_name_or_path ${model_to_model_name_or_path[${model}]} \
                --load_path "${persistent_dir}/results/checkpoints/dropout_c-${model_to_model_name_or_path[${model}]}_s-0" \
                --bias_type ${bias_type} \
                --persistent_dir "${persistent_dir}"
        else
            echo "${experiment_id} already computed"
        fi
    done


    for model in ${inlp_masked_lm_models[@]}; do
        for projection_prefix in "${projection_matrix_prefixes[@]}"; do
            experiment_id="crows_m-${projection_prefix}${model}_c-${model_to_model_name_or_path[${model}]}_t-${bias_type}"
            if [ ! -f "${persistent_dir}/results/crows/${experiment_id}.json" ]; then
                echo ${experiment_id}
                cmd=(python experiments/crows_debias.py \
                     --model ${model} \
                     --model_name_or_path ${model_to_model_name_or_path[${model}]} \
                     --projection_matrix "${persistent_dir}/results/projection_matrix/${projection_prefix}projection_m-${debiased_model_to_base_model[${model}]}_c-${model_to_model_name_or_path[${model}]}_t-${bias_type}_s-0.pt" \
                     --bias_type ${bias_type} \
                      --persistent_dir "${persistent_dir}")
                    if [[ -n "$projection_prefix" ]]; then
                        cmd+=(--projection_prefix "$projection_prefix")
                    fi
                    "${cmd[@]}"
            else
                echo "${experiment_id} already computed"
            fi

            # also compute CrowS for combined model GRADIEND + INLP
            base_model=${debiased_model_to_base_model[${model}]}
            debiased_models=(${model_to_debiased_models["${base_model}_${bias_type}"]})
            for debiased_model in ${debiased_models[@]}; do
                # skip if not *-N
                if [[ "$debiased_model" != *-N* ]]; then
                    continue
                fi

                # skip if projection_prefix is not empty
                if [[ -n "$projection_prefix" ]]; then
                    echo "Projection prefix: $projection_prefix"
                    continue
                fi

                model_id=$(basename "$debiased_model")
                experiment_id="crows_m-${projection_prefix}${model}_c-${model_id}_t-${bias_type}"
                if [ ! -f "${persistent_dir}/results/crows/${experiment_id}.json" ]; then
                    echo ${experiment_id}
                    cmd=(python experiments/crows_debias.py \
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
        done
    done


    for model in ${sentence_debias_masked_lm_models[@]}; do
        experiment_id="crows_m-${model}_c-${model_to_model_name_or_path[${model}]}_t-${bias_type}"
        if [ ! -f "${persistent_dir}/results/crows/${experiment_id}.json" ]; then
            echo ${experiment_id}
            python experiments/crows_debias.py \
                --model ${model} \
                --model_name_or_path ${model_to_model_name_or_path[${model}]} \
                --bias_direction "${persistent_dir}/results/subspace/subspace_m-${debiased_model_to_base_model[${model}]}_c-${model_to_model_name_or_path[${model}]}_t-${bias_type}.pt" \
                --bias_type ${bias_type} \
                --persistent_dir "${persistent_dir}"
        else
            echo "${experiment_id} already computed"
        fi

        # also compute CrowS for combined model GRADIEND + SentDebias
        base_model=${debiased_model_to_base_model[${model}]}
        debiased_models=(${model_to_debiased_models["${base_model}_${bias_type}"]})
        for debiased_model in ${debiased_models[@]}; do
            # skip if not *-N
            if [[ "$debiased_model" != *-N* ]]; then
                continue
            fi

            model_id=$(basename "$debiased_model")
            experiment_id="crows_m-${model}_c-${model_id}_t-${bias_type}"
            if [ ! -f "${persistent_dir}/results/crows/${experiment_id}.json" ]; then
                echo ${experiment_id}
                python experiments/crows_debias.py \
                     --model ${model} \
                     --model_name_or_path ${debiased_model} \
                     --bias_direction "${persistent_dir}/results/subspace/subspace_m-${base_model}_c-${model_id}_t-${bias_type}.pt" \
                     --bias_type ${bias_type} \
                     --persistent_dir "${persistent_dir}"
            else
                echo "${experiment_id} already computed"
            fi
        done
    done


    for model in ${self_debias_masked_lm_models[@]}; do
        experiment_id="crows_m-${model}_c-${model_to_model_name_or_path[${model}]}_t-${bias_type}"
        if [ ! -f "${persistent_dir}/results/crows/${experiment_id}.json" ]; then
            echo ${experiment_id}
            python experiments/crows_debias.py \
                --model ${model} \
                --model_name_or_path ${model_to_model_name_or_path[${model}]} \
                --bias_type ${bias_type} \
                --persistent_dir "${persistent_dir}"
        else
            echo "${experiment_id} already computed"
        fi
    done



    for model in ${cda_causal_lm_models[@]}; do
        model_name=${model_to_model_name_or_path[${model}]}
        model_id=$(basename "${model_name}")
        experiment_id="crows_m-${model}_c-${model_id}_t-${bias_type}"
        if [ ! -f "${persistent_dir}/results/crows/${experiment_id}.json" ]; then
            echo ${experiment_id}
            python experiments/crows_debias.py \
                --model ${model} \
                --model_name_or_path ${model_name} \
                --load_path "${persistent_dir}/results/checkpoints/cda_c-${model_id}_t-${bias_type}_s-0" \
                --bias_type ${bias_type} \
                --persistent_dir "${persistent_dir}"
        else
            echo "${experiment_id} already computed"
        fi
    done


    for model in ${dropout_causal_lm_models[@]}; do
        model_name=${model_to_model_name_or_path[${model}]}
        model_id=$(basename "${model_name}")
        experiment_id="crows_m-${model}_c-${model_id}_t-${bias_type}"
        if [ ! -f "${persistent_dir}/results/crows/${experiment_id}.json" ]; then
            echo ${experiment_id}
            python experiments/crows_debias.py \
                --model ${model} \
                --model_name_or_path ${model_name} \
                --load_path "${persistent_dir}/results/checkpoints/dropout_c-${model_id}_s-0" \
                --bias_type ${bias_type} \
                --persistent_dir "${persistent_dir}"
        else
            echo "${experiment_id} already computed"
        fi
    done


    for model in ${sentence_debias_causal_lm_models[@]}; do
        model_name=${model_to_model_name_or_path[${model}]}
        model_id=$(basename "${model_name}")
        experiment_id="crows_m-${model}_c-${model_id}_t-${bias_type}"
        if [ ! -f "${persistent_dir}/results/crows/${experiment_id}.json" ]; then
            echo ${experiment_id}
            python experiments/crows_debias.py \
                --model ${model} \
                --model_name_or_path ${model_name} \
                --bias_direction "${persistent_dir}/results/subspace/subspace_m-${debiased_model_to_base_model[${model}]}_c-${model_id}_t-${bias_type}.pt" \
                --bias_type ${bias_type} \
                --persistent_dir "${persistent_dir}"
        else
            echo "${experiment_id} already computed"
        fi

        # also compute CrowS for combined model GRADIEND + SentDebias
        base_model=${debiased_model_to_base_model[${model}]}
        debiased_models=(${model_to_debiased_models["${base_model}_${bias_type}"]})

        for debiased_model in ${debiased_models[@]}; do
            # skip if not *-N
            if [[ "$debiased_model" != *-N* ]]; then
                continue
            fi

            model_id=$(basename "$debiased_model")
            experiment_id="crows_m-${model}_c-${model_id}_t-${bias_type}"
            if [ ! -f "${persistent_dir}/results/crows/${experiment_id}.json" ]; then
                echo ${experiment_id}
                python experiments/crows_debias.py \
                     --model ${model} \
                     --model_name_or_path ${debiased_model} \
                     --bias_direction "${persistent_dir}/results/subspace/subspace_m-${base_model}_c-${model_id}_t-${bias_type}.pt" \
                     --bias_type ${bias_type} \
                     --persistent_dir "${persistent_dir}"
            else
                echo "${experiment_id} already computed"
            fi
        done
    done


    for model in ${inlp_causal_lm_models[@]}; do
        model_name=${model_to_model_name_or_path[${model}]}
        model_id=$(basename "${model_name}")
        for projection_prefix in "${projection_matrix_prefixes[@]}"; do

            # skip if projection_prefix is not empty
            echo "Projection prefix: $projection_prefix"
            experiment_id="crows_m-${projection_prefix}${model}_c-${model_id}_t-${bias_type}"
            if [ ! -f "${persistent_dir}/results/crows/${experiment_id}.json" ]; then
                echo ${experiment_id}
                cmd=(python experiments/crows_debias.py \
                     --model ${model} \
                     --model_name_or_path ${model_name} \
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

            if [[ ! -n "${projection_prefix}" ]]; then
                echo "Continue because of Projection prefix: $projection_prefix"
                continue
            fi

            # also compute CrowS for combined model GRADIEND + INLP
            base_model=${debiased_model_to_base_model[${model}]}
            base_model_id=$(basename "$base_model")
            debiased_models=(${model_to_debiased_models["${base_model}_${bias_type}"]})
            for debiased_model in ${debiased_models[@]}; do
              echo "Model: $debiased_model"
                # skip if not *-N
                if [[ "$debiased_model" != *-N* ]]; then
                    continue
                fi

                model_id=$(basename "$debiased_model")
                experiment_id="crows_m-${projection_prefix}${model}_c-${model_id}_t-${bias_type}"
                if [ ! -f "${persistent_dir}/results/crows/${experiment_id}.json" ]; then
                    echo ${experiment_id}
                    cmd=(python experiments/crows_debias.py \
                         --model ${model} \
                         --model_name_or_path ${debiased_model} \
                         --projection_matrix "${persistent_dir}/results/projection_matrix/${projection_prefix}projection_m-${base_model_id}_c-${model_id}_t-${bias_type}_s-0.pt" \
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
        done
    done


    for model in ${self_debias_causal_lm_models[@]}; do
        model_name=${model_to_model_name_or_path[${model}]}
        model_id=$(basename "${model_name}")
        experiment_id="crows_m-${model}_c-${model_id}_t-${bias_type}"
        if [ ! -f "${persistent_dir}/results/crows/${experiment_id}.json" ]; then
            echo ${experiment_id}
            python experiments/crows_debias.py \
                --model ${model} \
                --model_name_or_path ${model_name} \
                --bias_type ${bias_type} \
                --persistent_dir "${persistent_dir}"
        else
            echo "${experiment_id} already computed"
        fi
    done

done
