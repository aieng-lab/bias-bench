import os
import pickle
import random

import numpy as np
import sklearn
import torch.optim
from sklearn.decomposition import PCA
from tqdm import tqdm

from .rlace import solve_adv_game


def set_seeds(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


def _encode(model, batch_encoding, device):
    import torch

    # Move encoding to device
    input_ids = torch.tensor(batch_encoding["input_ids"]).to(device)
    attention_mask = torch.tensor(batch_encoding["attention_mask"]).to(device)

    # Handle models that do not use token_type_ids
    token_type_ids = batch_encoding.get("token_type_ids", None)
    if token_type_ids is not None:
        token_type_ids = torch.tensor(token_type_ids).to(device)

    # Determine model type and call appropriately
    if 'distilbert' in model.config.model_type:
        H = model(input_ids=input_ids,
                  attention_mask=attention_mask)["last_hidden_state"][:, 0, :]
    elif "bert" in model.config.model_type:
        H = model(input_ids=input_ids,
                  attention_mask=attention_mask,
                  token_type_ids=token_type_ids)["pooler_output"]
    elif "roberta" in model.config.model_type:
        H = model(input_ids=input_ids,
                  attention_mask=attention_mask)["pooler_output"]  # No token_type_ids
    elif "gpt" in model.config.model_type:
        H = model(input_ids=input_ids)["last_hidden_state"][:, -1,
            :]  # GPT models return last hidden state, take last token
    elif "llama" in model.config.model_type:
        H = model(input_ids=input_ids,
                  attention_mask=attention_mask)["last_hidden_state"][:, -1, :]
    else:
        raise ValueError(f"Unsupported model type: {model.config.model_type}")

    return H

# Code based on https://github.com/shauli-ravfogel/rlace-icml/blob/master/bios/encode.py
def encode(model, tokenizer, texts):
    all_H = []
    model.eval()
    device = model.device
    with torch.no_grad():

        print("Encoding...")
        batch_size = 100
        pbar = tqdm(range(len(texts)), ascii=True)

        for i in range(0, len(texts) - batch_size, batch_size):
            batch_texts = texts[i: i + batch_size]

            batch_encoding = tokenizer.batch_encode_plus(batch_texts, padding=True, max_length=512, truncation=True)

            H = _encode(model, batch_encoding, device)
            assert len(H.shape) == 2
            all_H.append(H.detach().cpu().numpy())

            pbar.update(batch_size)

        remaining = texts[(len(texts) // batch_size) * batch_size:]
        print(len(remaining))
        if len(remaining) > 0:
            batch_encoding = tokenizer.batch_encode_plus(remaining, padding=True, max_length=512, truncation=True)
            H = _encode(model, batch_encoding, device)
            assert len(H.shape) == 2
            all_H.append(H.detach().cpu().numpy())

    H_np = np.concatenate(all_H)
    assert len(H_np.shape) == 2
    assert len(H_np) == len(texts)
    return H_np

# based on https://github.com/shauli-ravfogel/rlace-icml/blob/master/bios/run_bios.py
def load_bios(group, model, tokenizer):
    with open("data/rlace/{}.pickle".format(group), "rb") as f:
        bios_data = pickle.load(f)

    model_id = os.path.basename(model.name_or_path)
    encoded_cache = f"data/rlace/encoded/{model_id}/{group}_encoded.pickle"
    if os.path.exists(encoded_cache):
        with open(encoded_cache, "rb") as f:
            X = pickle.load(f)
        print("Loaded encoded data from cache")
    else:
        texts = [d["hard_text_untokenized"] for d in bios_data]
        X = encode(model, tokenizer, texts)
        os.makedirs(os.path.dirname(encoded_cache), exist_ok=True)
        with open(encoded_cache, "wb") as f:
            pickle.dump(X, f)
        print("Saved encoded data to cache")

    Y = np.array([1 if d["g"] == "f" else 0 for d in bios_data])
    professions = np.array([d["p"] for d in bios_data])
    txts = [d["hard_text_untokenized"] for d in bios_data]
    random.seed(0)
    np.random.seed(0)
    X, Y, professions, txts, bios_data = sklearn.utils.shuffle(X, Y, professions, txts, bios_data)
    X = X[:]
    Y = Y[:]

    return X, Y, txts, professions, bios_data

def compute_projection_matrix(model, tokenizer, bias_type, rank=1, run_id=0, pca=False):
    assert bias_type == 'gender'

    optimizer_class = torch.optim.SGD
    optimizer_params_P = {"lr": 0.005, "weight_decay": 1e-4, "momentum": 0.0}
    optimizer_params_predictor = {"lr": 0.005, "weight_decay": 1e-5, "momentum": 0.9}
    finetune_mode = 'none'


    X, y, txts, professions, bios_data = load_bios("train", model, tokenizer)
    X, y = X[:100000], y[:100000]

    if pca:
        if not os.path.exists("pca"):
            os.makedirs("pca")
        pca = PCA(random_state=run_id, n_components=300)
        pca.fit(X)
        output = "results/pca/rlace/pca_{}_{}.pickle".format(finetune_mode, run_id)
        os.makedirs(os.path.dirname(output), exist_ok=True)
        with open(output, "wb") as f:
            pickle.dump(pca, f)
        X = pca.transform(X)

    output = solve_adv_game(X, y, X, y, rank=rank, device=model.device, out_iters=60000,
                            optimizer_class=optimizer_class, optimizer_params_P=optimizer_params_P,
                            optimizer_params_predictor=optimizer_params_predictor, epsilon=0.002,
                            batch_size=256)

    P = output["P"].to(torch.float32)

    return P