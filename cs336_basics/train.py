# import torch
# from torch import nn


# # Deliverable: Write a script that runs a training loop to train your model on user-provided input.
# # In particular, we recommend that your training script allow for (at least) the following:
# # • Ability to configure and control the various model and optimizer hyperparameters.
# # • Memory-eﬀicient loading of training and validation large datasets with np.memmap.
# # • Serializing checkpoints to a user-provided path.
# # • Periodically logging training and validation performance (e.g., to console and/or an external service like Weights and Biases).a


# # model hyperparameters
# vocab_size: int,
# context_length: int,
# d_model: int,
# num_layers: int,
# num_heads: int,
# d_ff: int,
# rope_theta: float,
# batch_size

# # optimizer hyperparameters
# params: Iterable[torch.nn.Parameter],
# lr: Float = 1e-3,
# weight_decay: Float = 0.01,
# betas: tuple[Float, Float] = (0.9, 0.999),



# def train(name: str, get_batch,
#           D: int, num_layers: int,
#           B: int, num_train_steps: int, lr: float):
#     model = Cruncher(dim=D, num_layers=0).to(get_device())
#     optimizer = SGD(model.parameters(), lr=0.01)
#     for t in range(num_train_steps):
#         # Get data
#         x, y = get_batch(B=B)
#         # Forward (compute loss)
#         pred_y = model(x)
#         loss = F.mse_loss(pred_y, y)
#         # Backward (compute gradients)
#         loss.backward()
#         # Update parameters
#         optimizer.step()
#         optimizer.zero_grad(set_to_none=True)



# vocab_size: int,
# context_length: int,
# d_model: int,
# num_layers: int,
# num_heads: int,
# d_ff: int,
# rope_theta: float,
# batch_size

#!/usr/bin/env python3

import argparse
import json
import pickle
from tests.adapters import Transformer_LM, AdamW, GetBatch, CrossEntropy
from einops import rearrange
import numpy as np
import torch

def get_device(index: int = 0) -> torch.device:
    """Try to use the GPU if possible, otherwise, use CPU."""
    if torch.cuda.is_available():
        return torch.device(f"cuda:{index}")
    else:
        return torch.device("cpu")

def main():
    parser = argparse.ArgumentParser(description="Load configuration from a JSON file.")
    parser.add_argument(
        'json_file',
        nargs='?',
        type=argparse.FileType('r'),
        default='/Users/kewang/assignment1-basics/cs336_basics/hyperparameters.json',
        help="Path to the JSON parameter file."
    )
    args = parser.parse_args()

    with args.json_file as f:
        params = json.load(f)

    print("--- Parameters ---")
    for key, value in params.items():
        print(f"{key}: {value}")

    params["betas"] = tuple(beta for beta in params["betas"])

    print("initilize the LM")
    model = Transformer_LM(params["d_model"],
                                    params["num_heads"],
                                    params["d_ff"],
                                    params["rope_theta"],
                                    params["context_length"],
                                    params["vocab_size"],
                                    params["num_layers"])\
                                    .to(get_device())

    print("initial the optimizer")
    optimizer = AdamW(model.parameters(),
                      params["lr"],
                      params["weight_decay"],
                      params["betas"])


    print("initilize the data")
    with open(params["DATA_PATH_TRAIN"], "rb") as f:
        dataset = pickle.load(f)
    # dataset = np.load(params["DATA_PATH_TRAIN"], mmap_mode="r")
    batches = GetBatch(dataset,
                    params["batch_size"],
                    params["context_length"],
                    get_device())

    cross_entropy = CrossEntropy()

    for _ in range(params["num_train_steps"]):
        # Get the batched dataset
        x, y = batches.get_batch()
        # Forward (compute loss)
        pred_y = model(x)
        loss = cross_entropy(
                rearrange(pred_y, "batch_size seq_len d_model -> (batch_size seq_len) d_model"),
                rearrange(y, "batch_size seq_len -> (batch_size seq_len)"))
        # Backward (compute gradients)
        loss.backward()
        # Update parameters
        optimizer.step()
        # Set gradient to 0
        optimizer.zero_grad(set_to_none=True)

if __name__ == '__main__':
    main()