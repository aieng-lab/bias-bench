import re


def generate_experiment_id(
    name,
    model=None,
    model_name_or_path=None,
    bias_type=None,
    seed=None,
):
    experiment_id = f"{name}"

    # Build the experiment ID.
    if isinstance(model, str):
        experiment_id += f"_m-{model}"
    if isinstance(model_name_or_path, str):
        # remove everything from the path before the gradiend folder
        model_name_or_path = re.sub(r'.*gradiend[\\/]', '', model_name_or_path)
        model_name_or_path = re.sub(r'.*gradient[\\/]', '', model_name_or_path) # todo deprecate
        model_name_or_path = re.sub(r'.*bench[\\/]', '', model_name_or_path)
        model_name_or_path = (model_name_or_path.replace('results/', '')
                              .replace('checkpoints/', '')
                              .replace('changed_models/', '')
                              .replace('../', '')
                              .replace('/', '-')
                              .replace('\\', '-'))
        experiment_id += f"_c-{model_name_or_path}"
    if isinstance(bias_type, str):
        experiment_id += f"_t-{bias_type}"
    if isinstance(seed, int):
        experiment_id += f"_s-{seed}"

    return experiment_id
