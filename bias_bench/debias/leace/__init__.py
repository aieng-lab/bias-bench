import os
import pickle

import torch
from sklearn.decomposition import PCA

from .leace import LeaceEraser
from ..rlace import load_bios


def compute_projection_matrix(model, tokenizer, bias_type, run_id=0, pca=False):
    assert bias_type == 'gender'

    finetune_mode = 'none'
    X, y, txts, professions, bios_data = load_bios("train", model, tokenizer)
    X, y = X[:100000], y[:100000]

    if pca:
        if not os.path.exists("pca"):
            os.makedirs("pca")
        pca = PCA(random_state=run_id, n_components=300)
        pca.fit(X)
        output = "results/pca/leace/pca_{}_{}.pickle".format(finetune_mode, run_id)
        os.makedirs(os.path.dirname(output), exist_ok=True)
        with open(output, "wb") as f:
            pickle.dump(pca, f)
        X = pca.transform(X)

    X_t = torch.from_numpy(X)
    y_t = torch.from_numpy(y)

    eraser = LeaceEraser.fit(X_t, y_t)

    P = eraser.P

    return P