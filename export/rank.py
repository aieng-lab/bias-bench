from bootstrap_results import read_results, models, VARIANTS, BASE, SS, SEAT, CROWS

def print_mean_rank_table(models=list(models), metrics=[SS, SEAT, CROWS], variants=[BASE] + list(VARIANTS)):
    results = read_results(metrics=metrics, variants=variants, models=models)
    
    # Prepare: ranks[metric][variant] = list of ranks across models
    ranks = {metric.id: {variant: [] for variant in variants} for metric in metrics}

    for metric in metrics:
        for model in models:
            # Collect all variant scores for this model and metric
            variant_scores = []
            for variant in variants:
                score = results[model][variant].scores.get(metric.id, -1)
                if score == -1:
                    print(f"Missing score for {model} {variant} {metric.id}")
                    continue
                variant_scores.append((variant, score))
            # Sort: lower score = better rank (change if higher is better)
            variant_scores.sort(key=lambda x: x[1])  
            # Assign ranks
            for rank, (variant, _) in enumerate(reversed(variant_scores), start=1):
                ranks[metric.id][variant].append(rank)

    # Compute mean ranks
    mean_ranks = {
        variant: {metric.id: sum(ranks[metric.id][variant]) / len(ranks[metric.id][variant]) 
                  for metric in metrics if len(ranks[metric.id][variant]) > 0}
        for variant in variants
    }

    # Compute overall Mean Rank Score per variant
    mean_rank_score = {
        variant: sum(mean_ranks[variant].values()) / len(mean_ranks[variant])
        for variant in variants
    }

    # Sort variants by Mean Rank Score (ascending: lower is better)
    sorted_variants = sorted(variants, key=lambda v: mean_rank_score[v])

    # Generate LaTeX table
    print("\\begin{tabular}{l" + " c" * len(metrics) + " c}")
    print("\\toprule")

    # Print header
    header = ['Variant'] + [f"{metric.id}" for metric in metrics] + ["Mean Rank Score"]
    print(" & ".join(header) + " \\\\ ")
    print("\\midrule")

    # Print rows
    for variant in sorted_variants:
        row = [variant.pretty] + [f"{mean_ranks[variant][metric.id]:.2f}" if metric.id in mean_ranks[variant] else "-" for metric in metrics] + [
            f"{mean_rank_score[variant]:.2f}"]
        print(" & ".join(row) + " \\\\ ")

    print("\\bottomrule")
    print("\\end{tabular}")

if __name__ == '__main__':
    print_mean_rank_table()