#!/bin/bash

source "shell_jobs/_experiment_configuration.sh"

#export CUDA_VISIBLE_DEVICES=0

# Inputs
model=$1
size_model_type=$2
base_model=$3
early_stopping="False"

# Remove suffix "Large" from size_model_type
model_type=${size_model_type%"Large"}

# Get the base name of the model (e.g., from a path)
base_model_name=$(basename "$model")

# Output results (for debugging or further processing)
echo "Model: $model"
echo "Size/Model Type: $size_model_type"
echo "Base Model Name: $base_model_name"
echo "Base Model: $base_model"


# perform the next experiment only if the model ends with "-N" or "-F"
if [[ $model == *"-N" ]]; then # || [[ $model == *"-F" ]]
    echo "Model ends with -N or -F"
    echo "Model: $model"

    model_id=$(basename "$model")
    experiment_id="glue_m-INLP${model_type}ForSequenceClassification_c-${model_id}_t-gender${early_stopping}"

    proj_matrix="${persistent_dir}/results/projection_matrix/projection_m-${model_type}Model_c-${model_id}_t-gender_s-0.pt"

    for seed in ${seeds[@]}; do
        for task in ${glue_tasks[@]}; do
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
                --output_dir "${persistent_dir}/results/checkpoints/${experiment_id}/${seed}/${task}" \
                --persistent_dir ${persistent_dir} \
                --early_stopping ${early_stopping}
        done
    done

    experiment_id="glue_m-SentenceDebias${model_type}ForSequenceClassification_c-${model_id}_t-gender${early_stopping}"
    for seed in ${seeds[@]}; do
        for task in ${glue_tasks[@]}; do
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
                    --output_dir "${persistent_dir}/results/checkpoints/${experiment_id}/${seed}/${task}" \
                    --persistent_dir ${persistent_dir} \
                    --early_stopping ${early_stopping}
        done
    done
fi


experiment_id="glue_m-CDA${model_type}ForSequenceClassification_c-${base_model}_t-gender${early_stopping}"
for seed in ${seeds[@]}; do
    for task in ${glue_tasks[@]}; do
        echo "${experiment_id}_g-${task}_s-${seed}"
        python experiments/run_glue.py \
            --model "CDA${model_type}ForSequenceClassification" \
            --model_name_or_path "${persistent_dir}/results/checkpoints/cda_c-${base_model}_t-gender_s-0" \
            --task_name ${task} \
            --do_train \
            --do_eval \
            --do_predict \
            --max_seq_length 128 \
            --per_device_train_batch_size 32 \
            --learning_rate 2e-5 \
            --num_train_epochs 3 \
            --seed ${seed} \
            --output_dir "${persistent_dir}/results/checkpoints/${experiment_id}/${seed}/${task}" \
            --persistent_dir ${persistent_dir} \
            --early_stopping ${early_stopping}
    done
done


experiment_id="glue_m-${model_type}ForSequenceClassification_c-${base_model_name}${early_stopping}"
for seed in ${seeds[@]}; do
    for task in ${glue_tasks[@]}; do
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
            --output_dir "${persistent_dir}/results/checkpoints/${experiment_id}/${seed}/${task}" \
            --persistent_dir ${persistent_dir} \
            --early_stopping ${early_stopping}
    done
done


experiment_id="glue_m-INLP${model_type}ForSequenceClassification_c-${base_model}_t-gender${early_stopping}"
for seed in ${seeds[@]}; do
    for task in ${glue_tasks[@]}; do
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
            --projection_matrix "${persistent_dir}/results/projection_matrix/projection_m-${debiased_model_to_base_model["INLP${model_type}ForSequenceClassification"]}_c-${base_model_name}_t-gender_s-0.pt" \
            --output_dir "${persistent_dir}/results/checkpoints/${experiment_id}/${seed}/${task}" \
            --persistent_dir ${persistent_dir} \
            --early_stopping ${early_stopping}
    done
done


experiment_id="glue_m-Dropout${model_type}ForSequenceClassification_c-${base_model}${early_stopping}"
for seed in ${seeds[@]}; do
    for task in ${glue_tasks[@]}; do
        echo "${experiment_id}_g-${task}_s-${seed}"
        python experiments/run_glue.py \
            --model "Dropout${model_type}ForSequenceClassification" \
            --model_name_or_path "${persistent_dir}/results/checkpoints/dropout_c-${base_model}_s-0" \
            --task_name ${task} \
            --do_train \
            --do_eval \
            --do_predict \
            --max_seq_length 128 \
            --per_device_train_batch_size 32 \
            --learning_rate 2e-5 \
            --num_train_epochs 3 \
            --seed ${seed} \
            --output_dir "${persistent_dir}/results/checkpoints/${experiment_id}/${seed}/${task}" \
            --persistent_dir ${persistent_dir} \
            --early_stopping ${early_stopping}
    done
done

experiment_id="glue_m-SentenceDebias${model_type}ForSequenceClassification_c-${base_model}_t-gender${early_stopping}"
debiased_base_model=${debiased_model_to_base_model["SentenceDebias${model_type}ForSequenceClassification"]}
model_name_or_path=${model_to_model_name_or_path["SentenceDebias${size_model_type}ForSequenceClassification"]}
for seed in ${seeds[@]}; do
    for task in ${glue_tasks[@]}; do
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
                --output_dir "${persistent_dir}/results/checkpoints/${experiment_id}/${seed}/${task}" \
                --persistent_dir ${persistent_dir} \
                --early_stopping ${early_stopping}
    done
done
