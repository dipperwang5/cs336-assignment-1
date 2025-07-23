from tests.adapters import Tokenizer
import numpy as np
import pathlib
import pickle

if __name__ == '__main__':
    DIR_PATH = pathlib.Path(__file__).resolve().parent.parent
    # INPUT_PATH_VOCAB = DIR_PATH / "results" / "TinyStories" / "vocab.pkl"
    # INPUT_PATH_MERGES = DIR_PATH / "results" / "TinyStories" / "merges.pkl"
    # DATA_PATH_TRAIN = DIR_PATH / "data" / "TinyStoriesV2-GPT4-train.txt"
    # DATA_PATH_VALID = DIR_PATH / "data" / "TinyStoriesV2-GPT4-valid.txt"
    # OUTPUT_PATH_TRAIN = DIR_PATH / "data" / "TinyStoriesV2-GPT4-train-tokenized.npy"
    # OUTPUT_PATH_VALID = DIR_PATH / "data" / "TinyStoriesV2-GPT4-valid-tokenized.npy"

    INPUT_PATH_VOCAB = DIR_PATH / "results" / "owt" / "vocab.pkl"
    INPUT_PATH_MERGES = DIR_PATH / "results" / "owt" / "merges.pkl"
    DATA_PATH_TRAIN = DIR_PATH / "data" / "owt_train.txt"
    DATA_PATH_VALID = DIR_PATH / "data" / "owt_valid.txt"
    OUTPUT_PATH_TRAIN = DIR_PATH / "data" / "owt_train-tokenized.npy"
    OUTPUT_PATH_VALID = DIR_PATH / "data" / "owt_valid-tokenized.npy"

    tokenizer = Tokenizer.from_files(INPUT_PATH_VOCAB, INPUT_PATH_MERGES)

    content = []
    with open(DATA_PATH_VALID, "rb") as f:
        for sentence in f:
            tokenizer.encode_iterable(sentence)
            content.extend(sentence)

    content = np.array(content, dtype=np.uint16)
    np.save(OUTPUT_PATH_VALID, content)
    # with open(OUTPUT_PATH_VALID, "wb") as f:
    #     pickle.dump(content, f)


    content = []
    with open(DATA_PATH_TRAIN, "rb") as f:
        for sentence in f:
            tokenizer.encode_iterable(sentence)
            content.extend(sentence)

    content = np.array(content, dtype=np.uint16)

    np.save(OUTPUT_PATH_TRAIN, content)
    # with open(OUTPUT_PATH_TRAIN, "wb") as f:
    #     pickle.dump(content, f)