#!/bin/bash

source "shell_jobs/_experiment_configuration.sh"


for bias_type in ${bias_types[@]}; do
  echo "Using bias type: ${bias_type}"

    for model in ${cda_masked_lm_models[@]}; do
        model_name=${model_to_model_name_or_path[${model}]}
        model_without_cda=${model//CDA/}
        model_id=$(basename "${model_name}")
        for seed in 0; do
            experiment_id="stereoset_m-${model_without_cda}_c-cda_c-${model_id}_t-${bias_type}_s-${seed}"
            if [ ! -f "${persistent_dir}/results/stereoset/${experiment_id}.json" ]; then
                echo ${experiment_id}
                python experiments/stereoset_debias.py \
                    --model ${model} \
                    --model_name_or_path ${model_name} \
                    --load_path "${persistent_dir}/results/checkpoints/cda_c-${model_id}_t-${bias_type}_s-${seed}" \
                    --bias_type ${bias_type} \
                    --seed ${seed} \
                    --persistent_dir "${persistent_dir}"
            else
                echo "${experiment_id} already computed"
            fi
        done
    done

    for model in ${cda_causal_lm_models[@]}; do
        model_name=${model_to_model_name_or_path[${model}]}
        model_without_cda=${model//CDA/}
        model_id=$(basename "${model_name}")
        for seed in 0; do
            experiment_id="stereoset_m-${model_without_cda}_c-cda_c-${model_id}_t-${bias_type}_s-${seed}"
            if [ ! -f "${persistent_dir}/results/stereoset/${experiment_id}.json" ]; then
                echo ${experiment_id}
                python experiments/stereoset_debias.py \
                    --model ${model} \
                    --model_name_or_path ${model_name} \
                    --load_path "${persistent_dir}/results/checkpoints/cda_c-${model_id}_t-${bias_type}_s-${seed}" \
                    --bias_type ${bias_type} \
                    --seed ${seed} \
                    --persistent_dir "${persistent_dir}"
            else
                echo "${experiment_id} already computed"
            fi
        done
    done



    for model in ${self_debias_causal_lm_models[@]}; do
        model_name=${model_to_model_name_or_path[${model}]}
        model_id=$(basename "${model_name}")
        experiment_id="stereoset_m-${model}_c-${model_id}_t-${bias_type}"
        if [ ! -f "${persistent_dir}/results/stereoset/${experiment_id}.json" ]; then
            echo ${experiment_id}
            python experiments/stereoset_debias.py \
                --model ${model} \
                --model_name_or_path ${model_name} \
                --bias_type ${bias_type} \
                --persistent_dir "${persistent_dir}"
        else
            echo "${experiment_id} already computed"
        fi
    done


    for model in ${inlp_causal_lm_models[@]}; do
        model_name=${model_to_model_name_or_path[${model}]}
        model_id=$(basename "${model_name}")
    
        for projection_prefix in "${projection_matrix_prefixes[@]}"; do
            if [[ $bias_type != "gender" ]] && [[ $projection_prefix != "" ]]; then
                echo "Skipping projection prefix ${projection_prefix} for bias type ${bias_type}"
                continue
            fi

            echo "Projection prefix: ${projection_prefix}"
            experiment_id="stereoset_m-${projection_prefix}${model}_c-${model_id}_t-${bias_type}"
            if [ ! -f "${persistent_dir}/results/stereoset/${experiment_id}.json" ]; then
                echo ${experiment_id}
                cmd=(python experiments/stereoset_debias.py \
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

            if [[ $bias_type != "gender" ]]; then
                echo "Skipping debiased models for bias type ${bias_type}"
                continue
            fi

            base_model=${debiased_model_to_base_model[${model}]}
            debiased_models=(${model_to_debiased_models["${base_model}_${bias_type}"]})
            for debiased_model in ${debiased_models[@]}; do
                debiased_model_id=$(basename "$debiased_model")
                if [[ $model_id != *"-M" ]] && [[ $model_id != *"-F" ]]; then
                    experiment_id="stereoset_m-${projection_prefix}${model}_c-${debiased_model_id}_t-${bias_type}"
                    if [ ! -f "${persistent_dir}/results/stereoset/${experiment_id}.json" ]; then
                         echo ${experiment_id}
                         cmd=(python experiments/stereoset_debias.py \
                             --model ${model} \
                             --model_name_or_path ${debiased_model} \
                             --projection_matrix "${persistent_dir}/results/projection_matrix/${projection_prefix}projection_m-${base_model}_c-${debiased_model_id}_t-${bias_type}_s-0.pt" \
                             --bias_type ${bias_type} \
                             --persistent_dir "${persistent_dir}")
                         if [[ -n "$projection_prefix" ]]; then
                            cmd+=(--projection_prefix "$projection_prefix")
                         fi
                         "${cmd[@]}"
                    else
                        echo "${experiment_id} already computed"
                    fi
                fi
            done
        done
    done


    for model in "${inlp_masked_lm_models[@]}"; do
        model_name=${model_to_model_name_or_path[${model}]}
        model_id=$(basename "${model_name}")

        for projection_prefix in "${projection_matrix_prefixes[@]}"; do
            if [[ $bias_type != "gender" ]] && [[ $projection_prefix != "" ]]; then
                echo "Skipping projection prefix ${projection_prefix} for bias type ${bias_type}"
                continue
            fi

            experiment_id="stereoset_m-${projection_prefix}${model}_c-${model_id}_t-${bias_type}"
            if [ ! -f "${persistent_dir}/results/stereoset/${experiment_id}.json" ]; then
                 echo ${experiment_id}
                 cmd=(python experiments/stereoset_debias.py \
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


            if [[ $bias_type != "gender" ]]; then
                echo "Skipping debiased models for bias type ${bias_type}"
                continue
            fi

            base_model=${debiased_model_to_base_model[${model}]}
            debiased_models=(${model_to_debiased_models["${base_model}_${bias_type}"]})
            for debiased_model in ${debiased_models[@]}; do
                if [[ $debiased_model != *"-M" ]] && [[ $debiased_model != *"-F" ]]; then
                    model_id=$(basename "$debiased_model")
                    experiment_id="stereoset_m-${projection_prefix}${model}_c-${model_id}_t-${bias_type}"
                    if [ ! -f "${persistent_dir}/results/stereoset/${experiment_id}.json" ]; then
                         echo ${experiment_id}
                         cmd=(python experiments/stereoset_debias.py \
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
                fi
            done
        done
    done

    for model in ${sentence_debias_causal_lm_models[@]}; do
        model_name=${model_to_model_name_or_path[${model}]}
        model_id=$(basename "${model_name}")

        experiment_id="stereoset_m-${model}_c-${model_to_model_name_or_path[${model}]}_t-${bias_type}"
        if [ ! -f "${persistent_dir}/results/stereoset/${experiment_id}.json" ]; then
            echo ${experiment_id}
            python experiments/stereoset_debias.py \
                --model ${model} \
                --model_name_or_path ${model_to_model_name_or_path[${model}]} \
                --bias_direction "${persistent_dir}/results/subspace/subspace_m-${debiased_model_to_base_model[${model}]}_c-${model_id}_t-${bias_type}.pt" \
                --bias_type ${bias_type} \
                --persistent_dir "${persistent_dir}"
        else
            echo "${experiment_id} already computed"
        fi


        if [[ $bias_type != "gender" ]]; then
            echo "Skipping debiased models for bias type ${bias_type}"
            continue
        fi

        base_model=${debiased_model_to_base_model[${model}]}
        debiased_models=(${model_to_debiased_models["${base_model}_${bias_type}"]})
        for debiased_model in ${debiased_models[@]}; do
            model_id=$(basename "$debiased_model")
            if [[ $model_id != *"-M" ]] && [[ $model_id != *"-F" ]]; then
                experiment_id="stereoset_m-${model}_c-${model_id}_t-${bias_type}"
                if [ ! -f "${persistent_dir}/results/stereoset/${experiment_id}.json" ]; then
                     echo ${experiment_id}
                     python experiments/stereoset_debias.py \
                         --model ${model} \
                         --model_name_or_path ${debiased_model} \
                         --bias_direction "${persistent_dir}/results/subspace/subspace_m-${base_model}_c-${model_id}_t-${bias_type}.pt" \
                         --bias_type ${bias_type} \
                         --persistent_dir "${persistent_dir}"
                else
                    echo "${experiment_id} already computed"
                fi
            fi
        done
    done


    for model in ${sentence_debias_masked_lm_models[@]}; do
        model_name=${model_to_model_name_or_path[${model}]}
        model_id=$(basename "${model_name}")
        experiment_id="stereoset_m-${model}_c-${model_id}_t-${bias_type}"
        if [ ! -f "${persistent_dir}/results/stereoset/${experiment_id}.json" ]; then
            echo ${experiment_id}
            python experiments/stereoset_debias.py \
                --model ${model} \
                --model_name_or_path ${model_name} \
                --bias_direction "${persistent_dir}/results/subspace/subspace_m-${debiased_model_to_base_model[${model}]}_c-${model_id}_t-${bias_type}.pt" \
                --bias_type ${bias_type} \
                --persistent_dir "${persistent_dir}"
        else
            echo "${experiment_id} already computed"
        fi

        # check if model ends with '-N'
        if [[ $model_name != *"-M" ]] && [[ $model_name != *"-F" ]]; then
            base_model=${debiased_model_to_base_model[${model}]}
            debiased_models=(${model_to_debiased_models["${base_model}_${bias_type}"]})
            for debiased_model in ${debiased_models[@]}; do
                model_id=$(basename "$debiased_model")
                experiment_id="stereoset_m-${model}_c-${model_id}_t-${bias_type}"
                if [ ! -f "${persistent_dir}/results/stereoset/${experiment_id}.json" ]; then
                     echo ${experiment_id}
                     python experiments/stereoset_debias.py \
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

    for model in ${self_debias_masked_lm_models[@]}; do
        model_name=${model_to_model_name_or_path[${model}]}
        model_id=$(basename "${model_name}")
        for bias_type in ${bias_types[@]}; do
            experiment_id="stereoset_m-${model}_c-${model_id}_t-${bias_type}"
            if [ ! -f "${persistent_dir}/results/stereoset/${experiment_id}.json" ]; then
                echo ${experiment_id}
                python experiments/stereoset_debias.py \
                    --model ${model} \
                    --model_name_or_path ${model_name} \
                    --bias_type ${bias_type} \
                    --persistent_dir "${persistent_dir}"
            else
                echo "${experiment_id} already computed"
            fi
        done
    done

    for model in ${dropout_masked_lm_models[@]}; do
        model_name=${model_to_model_name_or_path[${model}]}
        model_without_dropout=${model_name//Dropout/}
        model_id=$(basename "${model_name}")
        for seed in ${seeds[@]}; do
            experiment_id="stereoset_m-${model_without_dropout}_c-dropout_c-${model_id}_s-${seed}"
            if [ ! -f "${persistent_dir}/results/stereoset/${experiment_id}.json" ]; then
                echo ${experiment_id}
                python experiments/stereoset_debias.py \
                    --model ${model} \
                    --model_name_or_path ${model_name} \
                    --load_path "${persistent_dir}/results/checkpoints/dropout_c-${model_id}_s-${seed}" \
                    --seed ${seed} \
                    --persistent_dir "${persistent_dir}"
            else
                echo "${experiment_id} already computed"
            fi
        done
    done
done

for model in ${dropout_causal_lm_models[@]}; do
    model_name=${model_to_model_name_or_path[${model}]}
    model_without_dropout=${model_name//Dropout/}
    model_id=$(basename "${model_name}")
    for seed in 0; do
        experiment_id="stereoset_m-${model_without_dropout}_c-dropout_c-${model_id}_s-${seed}"
        if [ ! -f "${persistent_dir}/results/stereoset/${experiment_id}.json" ]; then
            echo ${experiment_id}
            python experiments/stereoset_debias.py \
              --model ${model} \
              --model_name_or_path ${model_to_model_name_or_path[${model}]} \
              --load_path "${persistent_dir}/results/checkpoints/dropout_c-${model_id}_s-${seed}" \
              --seed ${seed} \
              --persistent_dir "${persistent_dir}"
        else
            echo "${experiment_id} already computed"
        fi
    done
done
