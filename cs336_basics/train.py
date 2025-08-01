import argparse
import json
from tests.adapters import Transformer_LM, AdamW, GetBatch, CrossEntropy, run_save_checkpoint
from einops import rearrange
import numpy as np
import torch
import pathlib
import wandb

wandb.login()

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

    run = wandb.init(
        project="llm-assignment-hw1",
        config=params,
    )

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
    train_dataset = np.memmap(params["DATA_PATH_TRAIN"], dtype=np.uint16)
    valid_dataset = np.memmap(params["DATA_PATH_VALID"], dtype=np.uint16)
    train_batches = GetBatch(train_dataset,
                    params["batch_size"],
                    params["context_length"],
                    get_device())
    valid_batches = GetBatch(valid_dataset,
                    params["batch_size"],
                    params["context_length"],
                    get_device())

    cross_entropy = CrossEntropy()

    for iter in range(params["num_train_steps"]):
        # Get the batched dataset
        x, y = train_batches.get_batch()
        # Forward (compute loss)
        pred_y = model(x)
        train_loss = cross_entropy(
                rearrange(pred_y, "batch_size seq_len d_model -> (batch_size seq_len) d_model"),
                rearrange(y, "batch_size seq_len -> (batch_size seq_len)"))
        # Backward (compute gradients)
        train_loss.backward()
        # Update parameters
        optimizer.step()
        # Set gradient to 0
        optimizer.zero_grad(set_to_none=True)

        if iter != 0 and iter % params["save_interval"] == 0:
            run_save_checkpoint(model, optimizer, iter, pathlib.Path(params["MODEL_CHECK_POINT_PATH"]) / f"checkpoint_{iter}.pt")

        if iter != 0 and iter % params["val_interval"] == 0:
            print(f"training loss {train_loss.item()}")
            run.log({"training loss": train_loss.item()})

            model.eval()  # Set the model to evaluation mode

            with torch.no_grad():
                valid_losses = []
                for _ in range(params["num_eval_steps"]):
                    x, y = valid_batches.get_batch()
                    pred_y = model(x)
                    valid_loss = cross_entropy(
                        rearrange(pred_y, "batch_size seq_len d_model -> (batch_size seq_len) d_model"),
                        rearrange(y, "batch_size seq_len -> (batch_size seq_len)")
                    )
                    valid_losses.append(valid_loss.item())
                
                print(f"validation loss {np.mean(valid_losses)}")
                run.log({"valid loss": valid_loss.item()})

            model.train()  # Set the model back to training mode



if __name__ == '__main__':
    main()