from tests.adapters import Tokenizer
import pathlib
import random
random.seed(0)

if __name__ == '__main__':
    DIR_PATH = pathlib.Path(__file__).resolve().parent.parent
    INPUT_PATH_VOCAB = DIR_PATH / "results" / "TinyStories" / "vocab.pkl"
    INPUT_PATH_MERGES = DIR_PATH / "results" / "TinyStories" / "merges.pkl"
    DATA_PATH = DIR_PATH / "data" / "owt_valid.txt"
    SAMPLE_NUM = 10

    tokenizer = Tokenizer.from_files(INPUT_PATH_VOCAB, INPUT_PATH_MERGES)
    # print(tokenizer.vocab)
    # print(tokenizer.merges)
    # print(tokenizer.encode("hello world!"))
    # print(tokenizer.decode([1,2,9979,4,5,6,8,9994]))

    content = []
    with open(DATA_PATH, "rb") as f:
        for i in range(1000):
            line_content = f.readline().decode("utf-8")
            content.append(line_content)
    
    sampled_content = random.sample(content, 10)
    text = "".join(line_content for line_content in sampled_content)

    tokens = tokenizer.encode(text)
    print(f"length of the text {len(text)}")
    print(f"length of the tokens {len(tokens)}")
    print(f"compression ratio {len(text) / len(tokens)}")
    
