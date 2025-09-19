#!/bin/bash

source "shell_jobs/_experiment_configuration.sh"


#models=("BertModel" "DistilbertModel")
#models=("BertModel")
#models=("DistilbertModel" "GPT2Model")
bias_types=("race" "religion" )
#bias_types=("race")

echo "Bias types: ${bias_types[@]}"
echo "Models: ${models[@]}"



for bias_type in ${bias_types[@]}; do
    if [[ $bias_type == "gender" ]]; then
        seat_tests=${seat_gender_tests}
    else if [[ $bias_type == "race" ]]; then
        seat_tests=${seat_race_tests}
    else if [[ $bias_type == "religion" ]]; then
        seat_tests=${seat_religion_tests}
    else
        echo "Unknown bias type: $bias_type"
        exit 1
    fi
    fi
    fi

    echo "Using bias type: $bias_type"


    for model in ${models[@]}; do
        echo "Running debiased models for base model: $model"
        echo "${model}_${bias_type}"
        debiased_models=(${model_to_debiased_gradiend_models["${model}_${bias_type}"]})
        echo ${debiased_models[@]}
        for model_id in ${debiased_models[@]}; do
            base_model_id=$(basename "$model_id")
            experiment_id="seat_m-${model}_c-${base_model_id}"
            if [ ! -f "${persistent_dir}/results/seat/${bias_type}/${experiment_id}.json" ]; then
                echo ${experiment_id}
                python experiments/seat.py \
                    --tests ${seat_tests} \
                    --model ${model} \
                    --model_name_or_path ${model_id} \
                    --persistent_dir "${persistent_dir}" \
                    --bias_type ${bias_type}
            else
                echo "${experiment_id} already computed"
            fi
        done
    done

    echo "Running SEAT experiments"
    for model in ${models[@]}; do
        model_name=${model_to_model_name_or_path[${model}]}
        model_id=$(basename "$model_name")
        experiment_id="seat_m-${model}_c-${model_id}"
        if [ ! -f "${persistent_dir}/results/seat/${bias_type}/${experiment_id}.json" ]; then
            echo ${experiment_id}
            python experiments/seat.py \
                --tests ${seat_tests} \
                --model ${model} \
                --model_name_or_path ${model_name} \
                --persistent_dir "${persistent_dir}" \
                --bias_type ${bias_type}
        else
            echo "${experiment_id} already computed"
        fi
    done
    
    model_name="bert-large-cased"
    model="BertModel"
    experiment_id="seat_m-${model}_c-${model_name}"
    if [ ! -f "${persistent_dir}/results/seat/${bias_type}/${experiment_id}.json" ]; then
        echo ${experiment_id}
        python experiments/seat.py \
            --tests ${seat_tests} \
            --model ${model} \
            --model_name_or_path ${model_name} \
            --persistent_dir "${persistent_dir}" \
            --bias_type ${bias_type}
    else
        echo "${experiment_id} already computed"
    fi

done
