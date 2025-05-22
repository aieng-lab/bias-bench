from math import isnan

from bootstrap_results import read_results, models, VARIANTS, BASE, SS, SEAT, CROWS, LMS, GLUE


def print_rank_table(models=list(models), metrics=[SS, SEAT, CROWS], performance_metrics=[LMS, GLUE], variants=[BASE] + list(VARIANTS), use_proportional=True):
    results = read_results(metrics=metrics, variants=variants, models=models)
    performance_results = read_results(metrics=performance_metrics, variants=variants, models=models)

    # Prepare: ranks[metric][variant] = list of ranks across models
    ranks = {metric.id: {variant: [] for variant in variants} for metric in metrics}

    for metric in metrics:
        for model in models:
            # Collect all variant scores for this model and metric
            variant_scores = []
            for variant in variants:
                score = results[model][variant].scores.get(metric.id, None)
                if score is None:
                    continue
                variant_scores.append((variant, score))

            n = len(variant_scores)
            if n == 0:
                continue

            # Sort: lower score = better rank
            variant_scores.sort(key=lambda x: x[1])

            if use_proportional:
                # Compute proportional ranks
                if n > 1:
                    for i, (variant, _) in enumerate(variant_scores):
                        prop_rank = i / (n - 1)
                        ranks[metric.id][variant].append(prop_rank)
                else:
                    variant, _ = variant_scores[0]
                    ranks[metric.id][variant].append(0.5)
            else:
                # Integer ranks (1-based)
                for i, (variant, _) in enumerate(reversed(variant_scores), start=1):
                    ranks[metric.id][variant].append(i)

    # Compute mean ranks
    mean_ranks = {
        variant: {
            metric.id: sum(ranks[metric.id][variant]) / len(ranks[metric.id][variant])
            for metric in metrics if ranks[metric.id][variant]
        }
        for variant in variants
    }

    # Compute overall Mean Rank Score per variant
    overall_score = {
        variant: (sum(mean_ranks[variant].values()) / len(mean_ranks[variant])
                  if mean_ranks[variant] else float('nan'))
        for variant in variants
    }

    best_fnc = max if use_proportional else min

    # Determine best (lowest) per metric
    best_per_metric = {
        metric.id: best_fnc(
            mean_ranks[variant].get(metric.id, float('inf'))
            for variant in variants
            if metric.id in mean_ranks[variant]
        )
        for metric in metrics
    }

    # Determine best overall
    best_overall = best_fnc(score for score in overall_score.values() if not isnan(score))

    # Sort variants by overall score ascending
    sort_sign = -1 if use_proportional else 1
    sorted_variants = sorted(variants, key=lambda v: sort_sign * overall_score.get(v, float('inf')))

    # Print LaTeX table
    header = ['Model', r'$\Delta W$', 'PP', 'Mean'] + [m.id for m in metrics] + [f"{m.id}" for m in performance_metrics]

    def map_bool(state, is_best=False):
        if state:
            return r'\textcolor{blue}{\cmark}' if is_best else r'\cmark'
        else:
            return r'\xmark'

    # Identify best weight-changing and best post-processing approaches
    best_weight_change = None
    best_post_process = None
    for variant in sorted_variants:
        if variant.changes_weights and (
                best_weight_change is None or overall_score.get(variant) > overall_score.get(best_weight_change)):
            best_weight_change = variant
        if variant.is_post_processing and (
                best_post_process is None or overall_score.get(variant) > overall_score.get(best_post_process)):
            best_post_process = variant

    print(' & '.join(header) + ' \\ \midrule')
    for variant in sorted_variants:
        row = [variant.pretty]

        # Add the Delta W and PP columns
        best_variant_type = best_overall or variant == best_post_process or variant == best_weight_change
        row.append(map_bool(variant.changes_weights, is_best=(variant.changes_weights == best_variant_type)))
        row.append(map_bool(variant.is_post_processing, is_best=(variant.is_post_processing == best_variant_type)))

        mean_val = overall_score.get(variant)
        if mean_val == best_overall:
            row.append(f"\\textbf{{{mean_val:.2f}}}")
        else:
            row.append(f"{mean_val:.2f}")

        for metric in metrics:
            val = mean_ranks[variant].get(metric.id)
            if val is None:
                row.append('--')
            elif val == best_per_metric[metric.id]:
                row.append(f"\\textbf{{{val:.2f}}}")
            else:
                row.append(f"{val:.2f}")

        # Add performance metrics: value + delta
        for perf_metric in performance_metrics:
            score = performance_results.get(models[0], {}).get(variant, {}).scores.get(perf_metric.id)
            base_score = performance_results.get(models[0], {}).get(BASE, {}).scores.get(perf_metric.id)
            if score is not None and base_score is not None:
                delta = score.score - base_score.score
                arrow = f"\\uagn{{{delta:.2f}}}" if delta > 0 else f"\\dabn{{{delta:.2f}}}" if delta < 0 else ""
                row.append(f"{arrow} {score.score:.2f}")
            else:
                row.append("--")


        print(' & '.join(row) + r' \\')
    print('\\bottomrule')

if __name__ == '__main__':
    print_rank_table(use_proportional=True)