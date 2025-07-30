import torch    
from tests.adapters import Tokenizer, Transformer_LM, run_load_checkpoint, AdamW
import pathlib
import json
import pdb

def get_device(index: int = 0) -> torch.device:
    """Try to use the GPU if possible, otherwise, use CPU."""
    if torch.cuda.is_available():
        return torch.device(f"cuda:{index}")
    else:
        return torch.device("cpu")

def decoding(
    model: torch.nn.Module,
    tokenizor: Tokenizer,
    prompt: str,
    device: str,
    max_token_len: int = 256,
    temp: float = 1.0,
    top_k: float = 0.95
) -> str:

    model.eval()

    prompt = tokenizor.encode(prompt)
    prompt = torch.tensor(prompt, dtype=torch.long, device=device)
    end_token_id = tokenizor.encode("<|endoftext|>")[0] # the tokenizer has an issue when generating the vocab
    
    output_tokens = model.generate(prompt, max_token_len, temp, top_k, end_token_id)
    output_text = tokenizor.decode(output_tokens)

    model.train()

    return output_text

if __name__ == '__main__':

    with open('/Users/kewang/assignment1-basics/cs336_basics/hyperparameters.json', "r") as f:
        params = json.load(f)

    DIR_PATH = pathlib.Path(__file__).resolve().parent.parent
    INPUT_PATH_VOCAB = DIR_PATH / "results" / "TinyStories" / "vocab.pkl"
    INPUT_PATH_MERGES = DIR_PATH / "results" / "TinyStories" / "merges.pkl"

    #define the tokenizer
    tokenizer = Tokenizer.from_files(INPUT_PATH_VOCAB, INPUT_PATH_MERGES, ["<|endoftext|>"])
    
    #define the model
    model = Transformer_LM(d_model=params["d_model"],
                           num_heads=params["num_heads"],
                           d_ff=params["d_ff"],
                           theta=params["rope_theta"],
                           context_length=params["context_length"],
                           vocab_size=params["vocab_size"],
                           num_layers=params["num_layers"]
                           )
    
    #define the optimizer
    optimizer = AdamW(model.parameters(),
                      params["lr"],
                      params["weight_decay"],
                      params["betas"])
    
    #load current saved weight and optimizer
    run_load_checkpoint("/Users/kewang/assignment1-basics/models/checkpoint_10.pt", model, optimizer)

    #set device
    device = get_device()

    #doing inference and generate the output
    prompt = "The weather is good today. Let us"
    output_text = decoding(model, tokenizer, prompt, device)
    print(prompt + output_text)

