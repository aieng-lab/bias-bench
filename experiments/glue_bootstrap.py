import json
import os

import pandas as pd
import numpy as np
import tqdm
from datasets import load_metric, load_dataset
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from scipy.stats import spearmanr, norm

from export.glue import metric_mapping


def bootstrap_glue_scores_v1(df: pd.DataFrame, n_samples: int = 1000, seed: int = 42) -> list:
    """
    Perform bootstrapping on a pandas DataFrame containing GLUE predicted and correct values.
    Different metrics are applied based on the task.

    Metrics:
    - Accuracy: Default for most tasks.
    - F1 Score: MRPC
    - Spearman Correlation: STS-B
    - Matthew's Correlation Coefficient: CoLA

    Parameters:
    - df (pd.DataFrame): DataFrame containing columns ['task', 'seed', 'prediction', 'label'].
    - n_samples (int): Number of bootstrap samples to generate.
    - seed (int): Random seed for reproducibility.

    Returns:
    - list: A list containing the bootstrapped GLUE scores.
    """
    np.random.seed(seed)  # Set random seed for reproducibility

    # Check for required columns
    if not {'task', 'seed', 'prediction', 'label'}.issubset(df.columns):
        raise ValueError("Input DataFrame must contain 'task', 'seed', 'prediction', and 'label' columns.")

    # Task-specific metrics
    def compute_metric(task, predicted, correct):
        metric = load_metric("glue", task)
        result = metric.compute(predictions=predicted, references=correct)
        from export.glue import metric_mapping
        metric_key = metric_mapping[task].removeprefix('eval_')
        return result[metric_key]


    # List to store bootstrapped GLUE scores
    bootstrap_scores = []

    seeds = df['seed'].unique()

    for sample_seed in tqdm.tqdm(range(n_samples)):
        task_scores = []  # To store scores for each task in this bootstrap iteration

        for task in df['task'].unique():
            # Subset DataFrame for the current task
            task_df = df[df['task'] == task]

            # Pivot the DataFrame to get seeds and idx as multi-level columns
            pivot_df = task_df.pivot_table(index='idx', columns='seed', values='prediction')

            # Sample at the seed level for each idx, keeping the idx's predictions together
            sampled_pivot_df = pivot_df.sample(n=len(pivot_df), replace=True, random_state=sample_seed)

            sampled_pivot_df = sampled_pivot_df.merge(task_df[['idx', 'label']].drop_duplicates(), on='idx')

            scores = []
            for seed in seeds:
                seed_score = compute_metric(task, sampled_pivot_df[seed], sampled_pivot_df['label'])
                scores.append(seed_score)

            mean_task_score = np.mean(scores)
            task_scores.append(mean_task_score)

        # Average across all tasks
        final_glue_score = np.mean(task_scores)
        bootstrap_scores.append(final_glue_score)

    return bootstrap_scores



def bootstrap_glue_scores(df: pd.DataFrame, n_samples: int = 1000, seed: int = 42) -> list:
    """
    Perform bootstrapping on a pandas DataFrame containing GLUE predicted and correct values.
    Different metrics are applied based on the task.

    Metrics:
    - Accuracy: Default for most tasks.
    - F1 Score: MRPC
    - Spearman Correlation: STS-B
    - Matthew's Correlation Coefficient: CoLA

    Parameters:
    - df (pd.DataFrame): DataFrame containing columns ['task', 'seed', 'prediction', 'label', 'idx'].
    - n_samples (int): Number of bootstrap samples to generate.
    - seed (int): Random seed for reproducibility.

    Returns:
    - list: A list containing the bootstrapped GLUE scores.
    """
    np.random.seed(seed)  # Set random seed for reproducibility

    # Precompute unique tasks and seeds
    unique_tasks = df['task'].unique()
    unique_seeds = df['seed'].unique()

    # Pre-load metrics to avoid repeated loading
    metrics = {task: load_metric("glue", task) for task in unique_tasks}

    # Function to compute metrics
    def compute_metric(task, correct, predicted):
        metric = metrics[task]
        result = metric.compute(predictions=predicted, references=correct)
        from export.glue import metric_mapping
        metric_key = metric_mapping[task].removeprefix('eval_')
        return result[metric_key]

    # Prepare the pivot table once for efficiency
    pivot_tables = {
        task: df[df['task'] == task]
        .pivot_table(index='idx', columns='seed', values='prediction')
        .merge(df[df['task'] == task][['idx', 'label']].drop_duplicates(), on='idx')
        for task in unique_tasks
    }

    # Predefine bootstrap scores list
    bootstrap_scores = np.zeros(n_samples)

    # Bootstrap loop
    for sample_idx in tqdm.tqdm(range(n_samples), desc="Bootstrapping"):
        task_scores = []

        for task in unique_tasks:
            task_pivot_df = pivot_tables[task]

            # Bootstrap sample
            sampled_pivot_df = task_pivot_df.sample(
                n=len(task_pivot_df), replace=True, random_state=seed + sample_idx
            )

            # Compute scores for each seed
            scores = [
                compute_metric(task, sampled_pivot_df['label'], sampled_pivot_df[seed])
                for seed in unique_seeds
            ]
            task_scores.append(np.mean(scores))

        # Average across all tasks
        bootstrap_scores[sample_idx] = np.mean(task_scores)

    return bootstrap_scores.tolist()


def read_glue_results(base_dir, seeds=(0, 1, 2), tasks=('wnli', 'stsb', 'sst2', 'rte', 'qqp', 'qnli', 'mrpc', 'cola')): # 'mnli',
    print('Read GLUE results for ', base_dir)

    df_data = []
    for task in tasks:
        print('Task:', task)

        raw_datasets = load_dataset("glue", task)
        if task == 'mnli':
            split = 'validation_matched'
        else:
            split = 'validation'

        validation_data = raw_datasets[split]
        label_type = validation_data.features['label']

        for seed in seeds:
            file = f'{base_dir}/{seed}/{task}/eval_results_{task}.txt'
            df = pd.read_csv(file, sep='\t')
            df['task'] = task
            df['seed'] = seed
            if hasattr(label_type, 'str2int'):
                label_mapping = label_type.str2int
                df['prediction'] = df['prediction'].apply(label_mapping)

            labels = validation_data['label']
            df['label'] = labels
            df['idx'] = df.index

            #metric = load_metric("glue", task, trust_remote_code=True)
            #score = metric.compute(predictions=predicted, references=labels)['accuracy']
            #print(score)

            df_data.append(df)

    df = pd.concat(df_data)

    return df


def calculate_symmetric_ci(bootstrap_results, confidence_level=0.95):
    """
    Calculate the symmetric confidence intervals for the bootstrap results.
    """
    mean_score = np.mean(bootstrap_results)
    std_error = np.std(bootstrap_results, ddof=1)
    margin_of_error = norm.ppf(1 - (1 - confidence_level) / 2) * std_error

    lower_bound = mean_score - margin_of_error
    upper_bound = mean_score + margin_of_error

    return mean_score, lower_bound, upper_bound, margin_of_error


def test_needed_samples(input):
    """
    Test how many bootstrap samples are needed to stabilize the confidence intervals.
    Compares confidence intervals for smaller subsets (e.g., n=10, 25, 50, etc.) of bootstrapped results.

    Parameters:
    - input: path to the GLUE results input data.
    - full_n_samples: number of bootstrap samples to use for the full bootstrap.
    - sample_sizes: list of sample sizes to test (e.g., [10, 25, 50]).
    """

    bootstrap_results = glue_bootstrap(input, n_samples=1000)
    full_n_samples = len(bootstrap_results)
    sample_sizes = [[10*10**i, 25*10**i, 50*10**i] for i in range(0, np.log10(full_n_samples).astype(int))]
    sample_sizes = [item for sublist in sample_sizes for item in sublist if item <= full_n_samples]
    if not full_n_samples in sample_sizes:
        sample_sizes.append(full_n_samples)

    # Calculate the confidence intervals for the full number of bootstrap samples
    full_mean, full_lower, full_upper, full_margin_of_error = calculate_symmetric_ci(bootstrap_results, 0.95)
    print(f"Full Bootstrap ({full_n_samples} samples):")
    print(f"  Mean: {full_mean:.4f}, Symmetric 95% CI: {full_lower:.2f} - {full_upper:.2f}, ± {full_margin_of_error:.4f}")

    # Now calculate for the smaller sample sizes
    for size in sample_sizes:
        if size <= len(bootstrap_results):
            sampled_results = bootstrap_results[:size]  # Take the first 'size' results
            mean, lower, upper, margin_of_error = calculate_symmetric_ci(sampled_results, 0.95)
            print(f"\nSample Size = {size}:")
            print(f"  Mean: {mean:.4f}, Symmetric 95% CI: {lower:.2f} - {upper:.2f}, ± {margin_of_error:.4f}")
        else:
            print(f"Skipping sample size {size} as it exceeds the total number of samples available ({len(bootstrap_results)}).")



def glue_bootstrap(input, n_samples=1000, confidence_level=0.95):
    output = f"{input}/bootstrap_results_{n_samples}_{confidence_level}.json"
    output_old = f"{input}/bootstrap_results_{n_samples}.json"

    if os.path.exists(output):
        with open(output, "r") as f:
            bootstrap_results = json.load(f)
    elif confidence_level == 0.95 and os.path.exists(output_old):
        print(f"Loading old bootstrap results from {output_old}")
        with open(output_old, "r") as f:
            bootstrap_results = json.load(f)

        # convert to new format
        with open(output, "w") as f:
            json.dump(bootstrap_results, f)

        # delete the old file
        os.remove(output_old)
        print(f"Old bootstrap results converted to new format and saved to {output}")
    else:

        df_output = f'{input}/glue_results.csv'
        if os.path.exists(df_output):
            df = pd.read_csv(df_output)
        else:
            df = read_glue_results(input)
            df.to_csv(df_output, index=False)

        # Run bootstrapping
        print("Running bootstrapping on GLUE scores...")
        bootstrap_results = bootstrap_glue_scores(df, n_samples=n_samples, seed=42)

        # Save bootstrapped scores to a JSON file
        with open(output, "w") as f:
            json.dump(bootstrap_results, f)

    print(f"Bootstrapped GLUE scores (first 10): {bootstrap_results[:10]}")
    print(f"Mean Bootstrapped GLUE score: {np.mean(bootstrap_results):.2f}")
    # 0.95 and 0.99 confidence intervals
    print(f"95% CI: {np.percentile(bootstrap_results, 2.5):.2f} - {np.percentile(bootstrap_results, 97.5):.2f}")
    print(f"99% CI: {np.percentile(bootstrap_results, 0.5):.2f} - {np.percentile(bootstrap_results, 99.5):.2f}")


    # Step 1: Compute the mean
    mean_score = np.mean(bootstrap_results)

    # Step 2: Compute the standard error (SE)
    std_error = np.std(bootstrap_results, ddof=1)  # Use ddof=1 for sample std dev
    margin_of_error = norm.ppf(1 - (1 - confidence_level) / 2) * std_error

    # Step 3: Calculate symmetric confidence intervals
    lower_bound = mean_score - margin_of_error
    upper_bound = mean_score + margin_of_error
    # Print results
    print(f"Mean Bootstrapped GLUE score: {mean_score:.4f}")
    print(f"Symmetric 95% CI: {lower_bound:.2f} - {upper_bound:.2f}")
    print(f"Mean Bootstrapped GLUE score: {mean_score:.4f} ± {margin_of_error:.4f}")

    return bootstrap_results


def calculate_bootstrap_for_all_models(base_path):
    # find all folders in base_path that start with 'glue_m'
    models = [f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f)) and f.startswith('glue_m')]
    for model in models:
        glue_bootstrap(f'{base_path}/{model}')



# Example usage
if __name__ == "__main__":
    calculate_bootstrap_for_all_models('results/checkpoints')
    #model = 'results/checkpoints/glue_m-BertForSequenceClassification_c-bert-base-cased'
    #glue_bootstrap(model)
    #test_needed_samples(model)