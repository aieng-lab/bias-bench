import argparse
import os

import torch

from bias_bench.debias.rlace import compute_projection_matrix
from bias_bench.model import models, load_tokenizer
from bias_bench.util import generate_experiment_id

thisdir = os.path.dirname(os.path.realpath(__file__))
parser = argparse.ArgumentParser(description="Computes the projection matrix for RLACE.")
parser.add_argument(
    "--persistent_dir",
    action="store",
    type=str,
    default=os.path.realpath(os.path.join(thisdir, "..")),
    help="Directory where all persistent data will be stored.",
)
parser.add_argument(
    "--model",
    action="store",
    type=str,
    default="BertModel",
    #choices=["BertModel", "AlbertModel", "RobertaModel", "GPT2Model"],
    help="Model (e.g., BertModel) to compute the RLACE projection matrix for. "
    "Typically, these correspond to a HuggingFace class.",
)
parser.add_argument(
    "--model_name_or_path",
    action="store",
    type=str,
    default="bert-base-cased",
    #choices=["bert-base-cased", "albert-base-v2", "roberta-base", "gpt2"],
    help="HuggingFace model name or path (e.g., bert-base-cased). Checkpoint from which a "
    "model is instantiated.",
)
parser.add_argument(
    "--bias_type",
    action="store",
    default="gender",
    choices=["gender", "race", "religion"],
    help="What type of bias to compute the RLACE projection matrix for.",
)
parser.add_argument("--seed", action="store", type=int, default=0, help="Seed for RNG.")


if __name__ == "__main__":
    args = parser.parse_args()

    experiment_id = generate_experiment_id(
        name="rlace_projection",
        model=args.model,
        model_name_or_path=args.model_name_or_path,
        bias_type=args.bias_type,
        seed=args.seed,
    )

    print("Computing projection matrix:")
    print(f" - persistent_dir: {args.persistent_dir}")
    print(f" - model: {args.model}")
    print(f" - model_name_or_path: {args.model_name_or_path}")
    print(f" - bias_type: {args.bias_type}")
    print(f" - seed: {args.seed}")

    # Load model and tokenizer.
    model = getattr(models, args.model)(args.model_name_or_path)
    model.eval()
    tokenizer = load_tokenizer(args.model_name_or_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    projection_matrix = compute_projection_matrix(
        model,
        tokenizer,
        bias_type=args.bias_type,
    )

    print(
        f"Saving computed projection matrix to: {args.persistent_dir}/results/projection_matrix/{experiment_id}.pt"
    )
    os.makedirs(f"{args.persistent_dir}/results/projection_matrix", exist_ok=True)
    torch.save(
        projection_matrix,
        f"{args.persistent_dir}/results/projection_matrix/{experiment_id}.pt",
    )
