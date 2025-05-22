import json
import re

from datasets import load_dataset

from experiments import glue_bootstrap_evaluation


def get_jsonl_file(model_path, task):
    """
    Search for a .jsonl file in the given folder that matches the task name.
    Allows for dynamic timestamp suffix.

    Args:
        model_path (str): Path to the folder containing the jsonl files.
        task (str): Task name (e.g., 'mnli', 'mnli_mismatch', 'cola', etc.)

    Returns:
        str or None: Full path to the matching file, or None if not found.
    """
    pattern = re.compile(rf'^samples_{re.escape(task)}_\d{{4}}-\d{{2}}-\d{{2}}T\d{{2}}-\d{{2}}-\d{{2}}\.\d+\.jsonl$')

    for filename in os.listdir(model_path):
        if pattern.match(filename):
            return os.path.join(model_path, filename)

    return None


def read_jsonl(filepath):
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

tasks = [
    'wnli',
    #'stsb',
    'sst2',
    'rte',
    'qqp',
    'qnli',
    'mrpc',
    'mnli_matched',
    'mnli_mismatched',
    'cola',
]

dataset_cache = {}

def convert_model(input_path, seeds=(0, 1, 2), output_dir='results/checkpoints', model_type='Llama', output_suffix=''):
    model_id = input_path.split('/')[-1]
    for suffix in {'-inlp', '-rlace_', '-leace_', '-sentence_debias'}:
        model_id = model_id.replace(suffix, '')

    model_output_dir = f'{output_dir}/glue_m-{model_type}ForSequenceClassification_c-{model_id}{output_suffix}'

    for seed in seeds:
        seed_path = os.path.join(input_path, str(seed))
        # Check if the seed path exists
        if not os.path.exists(seed_path):
            print(f"Seed path {seed_path} does not exist.")
            continue

        # one folder should be contained in the seed path
        folders = [
            f for f in os.listdir(seed_path) if os.path.isdir(os.path.join(seed_path, f))
        ]

        if len(folders) == 0:
            print(f"No folders found in {seed_path}.")
            continue
        elif len(folders) != 1:
            print(f"Expected one folder in {seed_path}, found {len(folders)}, using the first one")

        folder = folders[0]
        model_path = os.path.join(seed_path, folder)
        model_output_dir_seed = f'{model_output_dir}/{seed}'

        for task in tasks:
            task_id = task
            task_name = task
            task_jsonl_id = task
            if task == 'mnli_mismatched':
                task_id = 'mnli'
                task_name = 'mnli-mm'
                task_jsonl_id = 'mnli_mismatch'
            elif task == 'mnli_matched':
                task_id = 'mnli'
                task_name = task_id
                task_jsonl_id = task_name

            output_file = f'{model_output_dir_seed}/{task_id}/eval_results_{task_name}.txt'

            if os.path.exists(output_file):
                print(f"Output file {output_file} already exists. Skipping conversion for task {task}.")
                #continue

            os.makedirs(os.path.dirname(output_file), exist_ok=True)

            jsonl_file = get_jsonl_file(model_path, task_jsonl_id)

            if jsonl_file is None:
                print(f"No jsonl file found for task {task} in {model_path}.")
                continue

            data = read_jsonl(jsonl_file)

            if task in dataset_cache:
                ds = dataset_cache[task]
            else:
                ds = load_dataset('glue', task, split='validation')
                dataset_cache[task] = ds
            labels = ds.features['label'].names

            converted_data = {'index': ['index'], 'prediction': ['prediction']}
            for d in data:
                index = d['doc']['idx']
                #logits = [float(x[0]) for x in data[0]['filtered_resps']]
                #most_likely_index = logits.index(max(logits))  # returns 0
                #prediction = labels[most_likely_index]
                prediction = labels[d['pred']]
                converted_data['index'].append(index)
                converted_data['prediction'].append(prediction)


            with open(output_file, 'w') as f:
                for index, prediction in zip(converted_data['index'], converted_data['prediction']):
                    f.write(f"{index}\t{prediction}\n")

            print(f"Converted data for task {task} saved to {output_file}")

            #all_results_file = f'{model_output_dir}/all_results.json'
            #all_results = {
            #    'eval_'
            #}

    eval_tasks = [
        'wnli',
        'sst2',
        'rte',
        'qqp',
        'qnli',
        'mrpc',
        'mnli',
        'mnli-mm',
        'cola',
    ]

    mean, margin_of_error = glue_bootstrap_evaluation.glue_bootstrap_scores(model_output_dir, tasks=eval_tasks)
    print(f"Mean: {mean}, Margin of Error: {margin_of_error}")


if __name__ == '__main__':
    # Convert all models in the directory
    import os
    import shutil

    result_dir = 'results/glue_zero_shot'

    convert_model(f'{result_dir}/Llama-3.2-3B', model_type='Llama')
    convert_model(f'{result_dir}/Llama-3.2-3B-N', model_type='Llama')
    convert_model(f'{result_dir}/Llama-3.2-3B-F', model_type='Llama')
    convert_model(f'{result_dir}/Llama-3.2-3B-M', model_type='Llama')
    convert_model(f'{result_dir}/Llama-3.2-3B-inlp', model_type='INLPLlama', output_suffix='_t-gender')
    convert_model(f'{result_dir}/Llama-3.2-3B-rlace_', model_type='rlace_INLPLlama', output_suffix='_t-gender')
    convert_model(f'{result_dir}/Llama-3.2-3B-leace_', model_type='leace_INLPLlama', output_suffix='_t-gender')
    convert_model(f'{result_dir}/Llama-3.2-3B-N-inlp', model_type='INLPLlama', output_suffix='_t-gender')
    convert_model(f'{result_dir}/Llama-3.2-3B-sentence_debias', model_type='SentenceDebiasLlama', output_suffix='_t-gender')
    convert_model(f'{result_dir}/Llama-3.2-3B-N-sentence_debias', model_type='SentenceDebiasLlama', output_suffix='_t-gender')


    convert_model(f'{result_dir}/Llama-3.2-3B-Instruct', model_type='LlamaInstruct')
    convert_model(f'{result_dir}/Llama-3.2-3B-Instruct-N', model_type='LlamaInstruct')
    convert_model(f'{result_dir}/Llama-3.2-3B-Instruct-F', model_type='LlamaInstruct')
    convert_model(f'{result_dir}/Llama-3.2-3B-Instruct-M', model_type='LlamaInstruct')
    convert_model(f'{result_dir}/Llama-3.2-3B-Instruct-inlp', model_type='INLPLlamaInstruct', output_suffix='_t-gender')
    convert_model(f'{result_dir}/Llama-3.2-3B-Instruct-rlace_', model_type='rlace_INLPLlamaInstruct', output_suffix='_t-gender')
    convert_model(f'{result_dir}/Llama-3.2-3B-Instruct-leace_', model_type='leace_INLPLlamaInstruct', output_suffix='_t-gender')
    convert_model(f'{result_dir}/Llama-3.2-3B-Instruct-N-inlp', model_type='INLPLlamaInstruct', output_suffix='_t-gender')
    convert_model(f'{result_dir}/Llama-3.2-3B-Instruct-sentence_debias', model_type='SentenceDebiasLlamaInstruct', output_suffix='_t-gender')
    convert_model(f'{result_dir}/Llama-3.2-3B-Instruct-N-sentence_debias', model_type='SentenceDebiasLlamaInstruct', output_suffix='_t-gender')
