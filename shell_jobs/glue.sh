#!/bin/bash

# Inputs
model=$1
size_model_type=$2
base_model=$3
benchmark_name=$4
echo "Running GLUE for model: ${model}, size_model_type: ${size_model_type}, base_model: ${base_model}, benchmark_name: ${benchmark_name}"
# default benchmark task is "glue"
if [ -z "$benchmark_name" ]; then
    echo "Benchmark task not specified, using default: glue"
    benchmark_name="glue"
fi

# Remove suffix "Large" from size_model_type
model_type=${size_model_type%"Large"}
model_type_model=${size_model_type}"Model"

# Get the base name of the model (e.g., from a path)
base_model_name=$(basename "$model")

# Output results (for debugging or further processing)
echo "Model: $model"
echo "Size/Model Type: $size_model_type"
echo "Base Model Name: $base_model_name"
echo "Base Model: $base_model"

glue_dir="${persistent_dir}/results/${benchmark_name}"
per_device_train_batch_size=32


# Compute GLUE for weight-debiased models (CDA, Dropout, GRADIEND_BPI/FPI/MPI) and certain combinations of debiasing approaches
# debiased GRADIEND_BPI models (-N) with INLP/RLACE/LEACE/SENTDEBIAS and CDA/DROPOUT+INLP/SENTDEBIAS
for bias_type in ${bias_types[@]}; do

    experiment_id="${benchmark_name}_m-SentenceDebias${model_type}ForSequenceClassification_c-${base_model}_t-${bias_type}"
    debiased_base_model=${debiased_model_to_base_model["SentenceDebias${model_type}ForSequenceClassification"]}
    model_name_or_path=${model_to_model_name_or_path["SentenceDebias${size_model_type}ForSequenceClassification"]}
    for seed in ${seeds[@]}; do
        for task in ${glue_tasks[@]}; do
           if [ ! -f "${glue_dir}/${experiment_id}/${seed}/${task}/README.md" ]; then
              echo "${experiment_id}_g-${task}_s-${seed}"
              python experiments/run_glue.py \
                      --model "SentenceDebias${model_type}ForSequenceClassification" \
                      --model_name_or_path ${base_model} \
                      --task_name ${task} \
                      --do_train \
                      --do_eval \
                      --do_predict \
                      --max_seq_length 128 \
                      --per_device_train_batch_size ${per_device_train_batch_size} \
                      --learning_rate 2e-5 \
                      --num_train_epochs 3 \
                      --seed ${seed} \
                      --bias_direction "${persistent_dir}/results/subspace/subspace_m-${debiased_base_model}_c-${model_name_or_path}_t-${bias_type}.pt" \
                      --output_dir "${glue_dir}/${experiment_id}/${seed}/${task}" \
                      --persistent_dir ${persistent_dir} \
                      --benchmark_name ${benchmark_name}
            else
                echo "${experiment_id}_g-${task}_s-${seed} already computed"
            fi
        done
    done

    for projection_prefix in "${projection_matrix_prefixes[@]}"; do

        if [[ $projection_prefix != "" ]] && [[ $bias_type != "gender" ]]; then
            continue
        fi

        experiment_id="${benchmark_name}_m-${projection_prefix}INLP${model_type}ForSequenceClassification_c-${base_model}_t-${bias_type}"
        for seed in ${seeds[@]}; do
            for task in ${glue_tasks[@]}; do
                if [ ! -f "${glue_dir}/${experiment_id}/${seed}/${task}/README.md" ]; then
                    echo "${experiment_id}_g-${task}_s-${seed}"
                    python experiments/run_glue.py \
                      --model "INLP${model_type}ForSequenceClassification" \
                      --model_name_or_path ${base_model} \
                      --task_name ${task} \
                      --do_train \
                      --do_eval \
                      --do_predict \
                      --max_seq_length 128 \
                      --per_device_train_batch_size ${per_device_train_batch_size} \
                      --learning_rate 2e-5 \
                      --num_train_epochs 3 \
                      --seed ${seed} \
                      --projection_matrix "${persistent_dir}/results/projection_matrix/${projection_prefix}projection_m-${debiased_model_to_base_model["INLP${model_type}ForSequenceClassification"]}_c-${base_model_name}_t-${bias_type}_s-0.pt" \
                      --output_dir "${glue_dir}/${experiment_id}/${seed}/${task}" \
                      --persistent_dir ${persistent_dir} \
                      --benchmark_name ${benchmark_name}
                else
                    echo "${experiment_id}_g-${task}_s-${seed} already computed"
                fi
            done
        done
    done

    debiased_models=(${model_to_debiased_models["${model_type_model}_${bias_type}"]})
    echo "Debiased models for ${model_type_model} and bias type ${bias_type}: ${debiased_models[@]}"

    for model in "${debiased_models[@]}"; do
        model_id=$(basename "$model")
        echo "Model: $model"
        if [[ $model == *"dropout"* || $model == *"cda"* || $model == *"-N" || $model == *"-F" || $model == *"-M" ]]; then
            # Dropout is the same model for all types of biases
            # CDA has the type of bias already in its name
            echo "Model: $model"
            experiment_id="${benchmark_name}_m-${model_type}ForSequenceClassification_c-${model_id}"
        else
            experiment_id="${benchmark_name}_m-${model_type}ForSequenceClassification_c-${model_id}_t-${bias_type}"
        fi

        for seed in ${seeds[@]}; do
            for task in ${glue_tasks[@]}; do
               if [ ! -f "${glue_dir}/${experiment_id}/${seed}/${task}/README.md" ]; then
                  echo "${experiment_id}_g-${task}_s-${seed}"

                  python experiments/run_glue.py \
                          --model "${model_type}ForSequenceClassification" \
                          --model_name_or_path ${model} \
                          --task_name ${task} \
                          --do_train \
                          --do_eval \
                          --do_predict \
                          --max_seq_length 128 \
                          --per_device_train_batch_size ${per_device_train_batch_size} \
                          --learning_rate 2e-5 \
                          --num_train_epochs 3 \
                          --seed ${seed} \
                          --output_dir "${glue_dir}/${experiment_id}/${seed}/${task}" \
                          --persistent_dir ${persistent_dir} \
                          --benchmark_name ${benchmark_name}
                else
                    echo "${experiment_id}_g-${task}_s-${seed} already computed"
                fi
            done
        done

        # perform the next experiment only if the model ends not with "-M" or "-F"
        if [[ $bias_type != "gender" ]]; then
           continue
        fi
        if [[ $model != *"-M" ]] && [[ $model != *"-F" ]]; then
            echo "Model: $model"

            for projection_prefix in "${projection_matrix_prefixes[@]}"; do
                if [[ $projection_prefix != "" ]]; then
                   continue # skip non-INLP projection prefixes for debiased models
                fi


                # compute GLUE for any combination with BPI model (-N), and only for INLP with all other debiased models
                if [[ $model == *"-N" ]] || [[ $projection_prefix == "" ]]; then
                    if [[ $model == *"cda"* ]]; then
                        # CDA has the type of bias already in its name
                        echo "Model: $model"
                        experiment_id="${benchmark_name}_m-${projection_prefix}INLP${model_type}ForSequenceClassification_c-${model_id}"
                        #continue # Dropout and CDA already computed above
                    else
                        experiment_id="${benchmark_name}_m-${projection_prefix}INLP${model_type}ForSequenceClassification_c-${model_id}_t-${bias_type}"
                    fi
                    proj_matrix="${persistent_dir}/results/projection_matrix/${projection_prefix}projection_m-${model_type}Model_c-${model_id}_t-${bias_type}_s-0.pt"

                    for seed in ${seeds[@]}; do
                        for task in ${glue_tasks[@]}; do
                           if [ ! -f "${glue_dir}/${experiment_id}/${seed}/${task}/README.md" ]; then
                              echo "${experiment_id}_g-${task}_s-${seed}"
                              python experiments/run_glue.py \
                                  --model "INLP${model_type}ForSequenceClassification" \
                                  --model_name_or_path ${model} \
                                  --task_name ${task} \
                                  --do_train \
                                  --do_eval \
                                  --do_predict \
                                  --max_seq_length 128 \
                                  --per_device_train_batch_size ${per_device_train_batch_size} \
                                  --learning_rate 2e-5 \
                                  --num_train_epochs 3 \
                                  --seed ${seed} \
                                  --projection_matrix ${proj_matrix} \
                                  --output_dir "${glue_dir}/${experiment_id}/${seed}/${task}" \
                                  --persistent_dir ${persistent_dir} \
                                  --benchmark_name ${benchmark_name}
                            else
                                echo "${experiment_id}_g-${task}_s-${seed} already computed"
                            fi
                        done
                    done
                fi
            done


            if [[ $model == *"cda"* ]]; then
                # Dropout is the same model for all types of biases
                # CDA has the type of bias already in its name
                experiment_id="${benchmark_name}_m-SentenceDebias${model_type}ForSequenceClassification_c-${model_id}"
            else
                experiment_id="${benchmark_name}_m-SentenceDebias${model_type}ForSequenceClassification_c-${model_id}_t-${bias_type}"
            fi
            for seed in ${seeds[@]}; do
                for task in ${glue_tasks[@]}; do
                   if [ ! -f "${glue_dir}/${experiment_id}/${seed}/${task}/README.md" ]; then
                      echo "${experiment_id}_g-${task}_s-${seed}"
                      python experiments/run_glue.py \
                              --model "SentenceDebias${model_type}ForSequenceClassification" \
                              --model_name_or_path ${model} \
                              --task_name ${task} \
                              --do_train \
                              --do_eval \
                              --do_predict \
                              --max_seq_length 128 \
                              --per_device_train_batch_size ${per_device_train_batch_size} \
                              --learning_rate 2e-5 \
                              --num_train_epochs 3 \
                              --seed ${seed} \
                              --bias_direction "${persistent_dir}/results/subspace/subspace_m-${debiased_model_to_base_model["SentenceDebias${model_type}ForSequenceClassification"]}_c-${model_id}_t-gender.pt" \
                              --output_dir "${glue_dir}/${experiment_id}/${seed}/${task}" \
                              --persistent_dir ${persistent_dir} \
                              --benchmark_name ${benchmark_name}
                    else
                        echo "${experiment_id}_g-${task}_s-${seed} already computed"
                    fi
                done
            done
        fi
    done
done

# Compute GLUE for the original model
experiment_id="${benchmark_name}_m-${model_type}ForSequenceClassification_c-${base_model_name}"
for seed in ${seeds[@]}; do
    for task in ${glue_tasks[@]}; do

        if [ ! -f "${glue_dir}/${experiment_id}/${seed}/${task}/README.md" ]; then
            echo "${experiment_id}_g-${task}_s-${seed}"
            python experiments/run_glue.py \
                --model "${model_type}ForSequenceClassification" \
                --model_name_or_path ${model} \
                --task_name ${task} \
                --do_train \
                --do_eval \
                --do_predict \
                --max_seq_length 128 \
                --per_device_train_batch_size ${per_device_train_batch_size} \
                --learning_rate 2e-5 \
                --num_train_epochs 3 \
                --seed ${seed} \
                --output_dir "${glue_dir}/${experiment_id}/${seed}/${task}" \
                --persistent_dir ${persistent_dir} \
                --benchmark_name ${benchmark_name}
        else
            echo "${experiment_id}_g-${task}_s-${seed} already computed"
        fi
    done
done








