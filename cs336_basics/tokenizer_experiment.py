from tests.adapters import Tokenizer
import numpy as np
import pathlib
import random
import time
random.seed(0)

if __name__ == '__main__':
    DIR_PATH = pathlib.Path(__file__).resolve().parent.parent
    INPUT_PATH_VOCAB = DIR_PATH / "results" / "TinyStories" / "vocab.pkl"
    INPUT_PATH_MERGES = DIR_PATH / "results" / "TinyStories" / "merges.pkl"
    DATA_PATH = DIR_PATH / "data" / "TinyStoriesV2-GPT4-valid.txt"
    SAMPLE_NUM = 10

    tokenizer = Tokenizer.from_files(INPUT_PATH_VOCAB, INPUT_PATH_MERGES)

    # sampling for sanity check
    content = []
    with open(DATA_PATH, "rb") as f:
        for i in range(10000):
            line_content = f.readline().decode("utf-8")
            content.append(line_content)
    
    sampled_content = random.sample(content, 10000)
    text = "".join(line_content for line_content in sampled_content)

    start_time = time.time()
    tokens = tokenizer.encode(text)
    end_time = time.time()
    
    print(f"length of the text {len(text)}")
    print(f"length of the tokens {len(tokens)}")
    print(f"compression ratio {len(text) / len(tokens)}")
    print(f"throughput {len(text) / (end_time - start_time)}")

    # process and save the tokenzied data
    DATA_PATH_TRAIN = DIR_PATH / "data" / "TinyStoriesV2-GPT4-train.txt"
    DATA_PATH_VALID = DIR_PATH / "data" / "TinyStoriesV2-GPT4-valid.txt"
    OUTPUT_PATH_TRAIN = DIR_PATH / "data" / "TinyStoriesV2-GPT4-train-tokenized.npy"
    OUTPUT_PATH_VALID = DIR_PATH / "data" / "TinyStoriesV2-GPT4-valid-tokenized.npy"

    content = []
    with open(DATA_PATH_VALID, "rb") as f:
        for sentence in f:
            tokenizer.encode_iterable(sentence)
            content.extend(sentence)

    content = np.array(content, dtype=np.uint16)
    np.save(OUTPUT_PATH_VALID, content)

    content = []
    with open(DATA_PATH_TRAIN, "rb") as f:
        for sentence in f:
            tokenizer.encode_iterable(sentence)
            content.extend(sentence)

    content = np.array(content, dtype=np.uint16)

    np.save(OUTPUT_PATH_TRAIN, content)