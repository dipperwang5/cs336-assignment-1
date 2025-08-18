from tests.adapters import run_train_bpe
import pickle
import pathlib

if __name__ == '__main__':

    TASK = "TinyStories"
    DIR_PATH = pathlib.Path(__file__).resolve().parent.parent
    
    INPUT_PATH = DIR_PATH / "data" / "TinyStoriesV2-GPT4-train.txt"
    VOCAB_OUPUT_PATH = DIR_PATH / "results" / TASK / "vocab.pkl"
    MERGES_OUTPUT_PATH = DIR_PATH / "results" / TASK / "merges.pkl"
    
    vocab_size = 10_000
    special_tokens = ["<|endoftext|>"]

    vocab, merges = run_train_bpe(INPUT_PATH, vocab_size, special_tokens)
    
    VOCAB_OUPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    MERGES_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with VOCAB_OUPUT_PATH.open("wb") as f:
        pickle.dump(vocab, f)
    
    with MERGES_OUTPUT_PATH.open("wb") as f:
        pickle.dump(merges, f)
    
    longest_token = max(vocab.values(), key=len)
    print("longest_token:", longest_token, "length:", len(longest_token))