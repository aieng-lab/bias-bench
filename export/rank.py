import numpy as np
from math import isnan

from bootstrap_results import read_results, models, VARIANTS, BASE, SS, SEAT, CROWS, LMS, GLUE, SUPER_GLUE, \
    GRADIEND_RACE1, GRADIEND_RACE2, GRADIEND_RACE3, SENTENCE_DEBIAS, SELF_DEBIAS, INLP, DROPOUT, CDA, \
    GRADIEND_RELIGION3, GRADIEND_RELIGION2, GRADIEND_RELIGION1


def print_rank_table(models=list(models), metrics=[SS, SEAT], performance_metrics=[LMS, GLUE, SUPER_GLUE], variants=[BASE] + list(VARIANTS), use_proportional=True, bias_type='gender'):
    results = read_results(metrics=metrics, variants=variants, models=models, bias_type=bias_type)
    performance_results = read_results(metrics=performance_metrics, variants=variants, models=models, bias_type=bias_type)

    # Prepare: ranks[metric][variant] = list of ranks across models
    ranks = {metric.id: {variant: [] for variant in variants} for metric in metrics}

    for metric in metrics:
        for model in models:
            # Collect all variant scores for this model and metric
            variant_scores = []
            for variant in variants:
                if variant in results[model]:
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
    best_overall = None
    for variant in sorted_variants:
        if variant.changes_weights and not variant.is_post_processing and (
                best_weight_change is None or overall_score.get(variant) > overall_score.get(best_weight_change)):
            best_weight_change = variant
        if variant.is_post_processing and not variant.changes_weights and (
                best_post_process is None or overall_score.get(variant) > overall_score.get(best_post_process)):
            best_post_process = variant
        if variant.changes_weights and variant.is_post_processing  and (
                best_overall is None or overall_score.get(variant) > overall_score.get(best_overall)):
            best_overall = variant

    # Precompute best scores per metric **after aggregation per variant**
    # Precompute best scores per metric across all variants
    best_perf_scores = {}
    for perf_metric in performance_metrics:
        all_scores = []
        for variant in sorted_variants:
            for m in models:
                try:
                    val = performance_results[m][variant].scores[perf_metric.id].score
                    if 'ss' in perf_metric.id.lower():  # smaller-is-better adjustment
                        val = abs(val - 50)
                    all_scores.append(val)
                except (KeyError, AttributeError, TypeError):
                    continue

        if all_scores:
            if perf_metric in [SS, SEAT]:  # smaller is better
                best_perf_scores[perf_metric.id] = min(all_scores)
            else:  # larger is better
                best_perf_scores[perf_metric.id] = max(all_scores)


    print(' & '.join(header) + ' \\ \midrule')
    for variant in sorted_variants:
        row = [variant.pretty]

        # Add the Delta W and PP columns
        best_variant_type = variant == best_overall or variant == best_post_process or variant == best_weight_change
        row.append(map_bool(variant.changes_weights, is_best=(variant.changes_weights and best_variant_type)))
        row.append(map_bool(variant.is_post_processing, is_best=(variant.is_post_processing and best_variant_type)))

        fmt = '.2f'
        mean_val = overall_score.get(variant)
        if mean_val == best_overall:
            row.append(f"\\textbf{{{mean_val:{fmt}}}}")
        else:
            row.append(f"{mean_val:{fmt}}")

        for metric in metrics:
            val = mean_ranks[variant].get(metric.id)
            if val is None:
                row.append('--')
            elif val == best_per_metric[metric.id]:
                row.append(f"\\textbf{{{val:{fmt}}}}")
            else:
                row.append(f"{val:{fmt}}")

        fmt = '.2f'

        # Add performance metrics: value + delta
        for perf_metric in performance_metrics:
            try:
                # Gather variant scores
                scores = []
                base_scores = []
                for m in models:
                    try:
                        val = performance_results[m][variant].scores[perf_metric.id].score
                        base_val = performance_results[m][BASE].scores[perf_metric.id].score
                        if 'ss' in perf_metric.id.lower():
                            val = abs(val - 50)
                            base_val = abs(base_val - 50)
                        scores.append(val)
                        base_scores.append(base_val)
                    except (KeyError, AttributeError, TypeError):
                        continue

                if scores and base_scores:
                    score = np.mean(scores)
                    base_score = np.mean(base_scores)
                    delta = score - base_score
                    is_best = f"{score:{fmt}}" == f"{best_perf_scores[perf_metric.id]:{fmt}}"

                    # Determine arrow
                    if perf_metric in [SS, SEAT]:
                        arrow = "\\uan" if delta > 0 else "\\dan" if delta < 0 else ""
                    else:
                        arrow = "\\uagn" if delta > 0 else "\\dabn" if delta < 0 else ""


                    if variant == BASE:
                        # dont print delta
                        if is_best:
                            row.append(f"\\textbf{{{score:{fmt}}}}")
                        else:
                            row.append(f"{score:{fmt}}")
                    else:
                        # Append formatted value
                        if is_best:
                            row.append(f"{arrow}{{\\textbf{{{delta:{fmt}}}}}} \\textbf{{{score:{fmt}}}}")
                        else:
                            row.append(f"{arrow}{{{delta:{fmt}}}} {score:{fmt}}")
                else:
                    row.append("--")
            except Exception:
                row.append("--")

        print(' & '.join(row) + r' \\')
    print('\\bottomrule')

def print_rank_table_gender():
    print_rank_table(use_proportional=True)

def print_rank_table_race():
    print_rank_table(
        variants=[BASE, GRADIEND_RACE1, GRADIEND_RACE2, GRADIEND_RACE3, CDA, DROPOUT, INLP, SELF_DEBIAS, SENTENCE_DEBIAS],
        bias_type='race'
    )


def print_rank_table_religion():
    print_rank_table(
        variants=[BASE, GRADIEND_RELIGION1, GRADIEND_RELIGION2, GRADIEND_RELIGION3, CDA, DROPOUT, INLP, SELF_DEBIAS, SENTENCE_DEBIAS],
        bias_type='religion'
    )

if __name__ == '__main__':
    print_rank_table_gender()
    print_rank_table_race()
    print_rank_table_religion()