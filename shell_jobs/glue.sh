#!/bin/bash

# Inputs
model=$1
size_model_type=$2
base_model=$3

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




# Compute GLUE for the original model
experiment_id="glue_m-${model_type}ForSequenceClassification_c-${base_model_name}"
for seed in ${seeds[@]}; do
    for task in ${glue_tasks[@]}; do
        if [ ! -f "${checkpoint_dir}/${experiment_id}/${seed}/${task}/eval_results.json" ]; then
            echo "${experiment_id}_g-${task}_s-${seed}"
            python experiments/run_glue.py \
                --model "${model_type}ForSequenceClassification" \
                --model_name_or_path ${model} \
                --task_name ${task} \
                --do_train \
                --do_eval \
                --do_predict \
                --max_seq_length 128 \
                --per_device_train_batch_size 32 \
                --learning_rate 2e-5 \
                --num_train_epochs 3 \
                --seed ${seed} \
                --output_dir "${checkpoint_dir}/${experiment_id}/${seed}/${task}" \
                --persistent_dir ${persistent_dir}
        else
            echo "${experiment_id}_g-${task}_s-${seed} already computed"
        fi
    done
done

# Compute GLUE for CDA
experiment_id="glue_m-CDA${model_type}ForSequenceClassification_c-${base_model}_t-gender"
for seed in ${seeds[@]}; do
    for task in ${glue_tasks[@]}; do
       if [ ! -f "${checkpoint_dir}/${experiment_id}/${seed}/${task}/eval_results.json" ]; then
          echo "${experiment_id}_g-${task}_s-${seed}"
          python experiments/run_glue.py \
              --model "CDA${model_type}ForSequenceClassification" \
              --model_name_or_path "${checkpoint_dir}/cda_c-${base_model}_t-gender_s-0" \
              --task_name ${task} \
              --do_train \
              --do_eval \
              --do_predict \
              --max_seq_length 128 \
              --per_device_train_batch_size 32 \
              --learning_rate 2e-5 \
              --num_train_epochs 3 \
              --seed ${seed} \
              --output_dir "${checkpoint_dir}/${experiment_id}/${seed}/${task}" \
              --persistent_dir ${persistent_dir}
        else
            echo "${experiment_id}_g-${task}_s-${seed} already computed"
        fi
    done
done

# Compute GLUE for Dropout
experiment_id="glue_m-Dropout${model_type}ForSequenceClassification_c-${base_model}"
for seed in ${seeds[@]}; do
    for task in ${glue_tasks[@]}; do
       if [ ! -f "${checkpoint_dir}/${experiment_id}/${seed}/${task}/eval_results.json" ]; then
          echo "${experiment_id}_g-${task}_s-${seed}"
          python experiments/run_glue.py \
              --model "Dropout${model_type}ForSequenceClassification" \
              --model_name_or_path "${checkpoint_dir}/dropout_c-${base_model}_s-0" \
              --task_name ${task} \
              --do_train \
              --do_eval \
              --do_predict \
              --max_seq_length 128 \
              --per_device_train_batch_size 32 \
              --learning_rate 2e-5 \
              --num_train_epochs 3 \
              --seed ${seed} \
              --output_dir "${checkpoint_dir}/${experiment_id}/${seed}/${task}" \
              --persistent_dir ${persistent_dir}
        else
            echo "${experiment_id}_g-${task}_s-${seed} already computed"
        fi
    done
done



# Compute GLUE for INLP
for projection_prefix in "${projection_matrix_prefixes[@]}"; do
    experiment_id="glue_m-${projection_prefix}INLP${model_type}ForSequenceClassification_c-${base_model}_t-gender"
    for seed in ${seeds[@]}; do
        for task in ${glue_tasks[@]}; do
            if [ ! -f "${checkpoint_dir}/${experiment_id}/${seed}/${task}/eval_results.json" ]; then
                echo "${experiment_id}_g-${task}_s-${seed}"
                python experiments/run_glue.py \
                  --model "INLP${model_type}ForSequenceClassification" \
                  --model_name_or_path ${base_model} \
                  --task_name ${task} \
                  --do_train \
                  --do_eval \
                  --do_predict \
                  --max_seq_length 128 \
                  --per_device_train_batch_size 32 \
                  --learning_rate 2e-5 \
                  --num_train_epochs 3 \
                  --seed ${seed} \
                  --projection_matrix "${persistent_dir}/results/projection_matrix/${projection_prefix}projection_m-${debiased_model_to_base_model["INLP${model_type}ForSequenceClassification"]}_c-${base_model_name}_t-gender_s-0.pt" \
                  --output_dir "${checkpoint_dir}/${experiment_id}/${seed}/${task}" \
                  --persistent_dir ${persistent_dir}
            else
                echo "${experiment_id}_g-${task}_s-${seed} already computed"
            fi
        done
    done
done


# Compute GLUE for SentenceDebias
experiment_id="glue_m-SentenceDebias${model_type}ForSequenceClassification_c-${base_model}_t-gender"
debiased_base_model=${debiased_model_to_base_model["SentenceDebias${model_type}ForSequenceClassification"]}
model_name_or_path=${model_to_model_name_or_path["SentenceDebias${size_model_type}ForSequenceClassification"]}
for seed in ${seeds[@]}; do
    for task in ${glue_tasks[@]}; do
       if [ ! -f "${checkpoint_dir}/${experiment_id}/${seed}/${task}/eval_results.json" ]; then
          echo "${experiment_id}_g-${task}_s-${seed}"
          python experiments/run_glue.py \
                  --model "SentenceDebias${model_type}ForSequenceClassification" \
                  --model_name_or_path ${base_model} \
                  --task_name ${task} \
                  --do_train \
                  --do_eval \
                  --do_predict \
                  --max_seq_length 128 \
                  --per_device_train_batch_size 32 \
                  --learning_rate 2e-5 \
                  --num_train_epochs 3 \
                  --seed ${seed} \
                  --bias_direction "${persistent_dir}/results/subspace/subspace_m-${debiased_base_model}_c-${model_name_or_path}_t-gender.pt" \
                  --output_dir "${checkpoint_dir}/${experiment_id}/${seed}/${task}" \
                  --persistent_dir ${persistent_dir}
        else
            echo "${experiment_id}_g-${task}_s-${seed} already computed"
        fi
    done
done



# Compute GLUE for weight-debiased models (CDA, Dropout, GRADIEND_BPI/FPI/MPI) and certain combinations of debiasing approaches
# debiased GRADIEND_BPI models (-N) with INLP/RLACE/LEACE/SENTDEBIAS and CDA/DROPOUT+INLP/SENTDEBIAS
debiased_models=(${model_to_debiased_models[$model_type_model]})
for model in "${debiased_models[@]}"; do
    model_id=$(basename "$model")
    experiment_id="glue_m-${model_type}ForSequenceClassification_c-${model_id}_t-gender"
    for seed in ${seeds[@]}; do
        for task in ${glue_tasks[@]}; do
           if [ ! -f "${checkpoint_dir}/${experiment_id}/${seed}/${task}/eval_results.json" ]; then
              echo "${experiment_id}_g-${task}_s-${seed}"
              python experiments/run_glue.py \
                      --model "${model_type}ForSequenceClassification" \
                      --model_name_or_path ${model} \
                      --task_name ${task} \
                      --do_train \
                      --do_eval \
                      --do_predict \
                      --max_seq_length 128 \
                      --per_device_train_batch_size 32 \
                      --learning_rate 2e-5 \
                      --num_train_epochs 3 \
                      --seed ${seed} \
                      --output_dir "${checkpoint_dir}/${experiment_id}/${seed}/${task}" \
                      --persistent_dir ${persistent_dir}
            else
                echo "${experiment_id}_g-${task}_s-${seed} already computed"
            fi
        done
    done

    # perform the next experiment only if the model ends not with "-M" or "-F"
    if [[ $model != *"-M" ]] && [[ $model != *"-F" ]]; then
        echo "Model: $model"

        for projection_prefix in "${projection_matrix_prefixes[@]}"; do
            # compute GLUE for any combination with BPI model (-N), and only for INLP with all other debiased models
            if [[ $model == *"-N" ]] || [[ $projection_prefix == "" ]]; then
                experiment_id="glue_m-${projection_prefix}INLP${model_type}ForSequenceClassification_c-${model_id}_t-gender"
                proj_matrix="${persistent_dir}/results/projection_matrix/${projection_prefix}projection_m-${model_type}Model_c-${model_id}_t-gender_s-0.pt"

                for seed in ${seeds[@]}; do
                    for task in ${glue_tasks[@]}; do
                       if [ ! -f "${checkpoint_dir}/${experiment_id}/${seed}/${task}/eval_results.json" ]; then
                          echo "${experiment_id}_g-${task}_s-${seed}"
                          python experiments/run_glue.py \
                              --model "INLP${model_type}ForSequenceClassification" \
                              --model_name_or_path ${model} \
                              --task_name ${task} \
                              --do_train \
                              --do_eval \
                              --do_predict \
                              --max_seq_length 128 \
                              --per_device_train_batch_size 32 \
                              --learning_rate 2e-5 \
                              --num_train_epochs 3 \
                              --seed ${seed} \
                              --projection_matrix ${proj_matrix} \
                              --output_dir "${checkpoint_dir}/${experiment_id}/${seed}/${task}" \
                              --persistent_dir ${persistent_dir}
                        else
                            echo "${experiment_id}_g-${task}_s-${seed} already computed"
                        fi
                    done
                done
            fi
        done

        experiment_id="glue_m-SentenceDebias${model_type}ForSequenceClassification_c-${model_id}_t-gender"
        for seed in ${seeds[@]}; do
            for task in ${glue_tasks[@]}; do
               if [ ! -f "${checkpoint_dir}/${experiment_id}/${seed}/${task}/eval_results.json" ]; then
                  echo "${experiment_id}_g-${task}_s-${seed}"
                  python experiments/run_glue.py \
                          --model "SentenceDebias${model_type}ForSequenceClassification" \
                          --model_name_or_path ${model} \
                          --task_name ${task} \
                          --do_train \
                          --do_eval \
                          --do_predict \
                          --max_seq_length 128 \
                          --per_device_train_batch_size 32 \
                          --learning_rate 2e-5 \
                          --num_train_epochs 3 \
                          --seed ${seed} \
                          --bias_direction "${persistent_dir}/results/subspace/subspace_m-${debiased_model_to_base_model["SentenceDebias${model_type}ForSequenceClassification"]}_c-${model_id}_t-gender.pt" \
                          --output_dir "${checkpoint_dir}/${experiment_id}/${seed}/${task}" \
                          --persistent_dir ${persistent_dir}
                else
                    echo "${experiment_id}_g-${task}_s-${seed} already computed"
                fi
            done
        done
    fi
done