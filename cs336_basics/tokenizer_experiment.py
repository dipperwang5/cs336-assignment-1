# from tests.adapters import Tokenizer
# import numpy as np
# import pathlib
# import random
# import time
# random.seed(0)

# if __name__ == '__main__':
#     DIR_PATH = pathlib.Path(__file__).resolve().parent.parent
#     INPUT_PATH_VOCAB = DIR_PATH / "results" / "TinyStories" / "vocab.pkl"
#     INPUT_PATH_MERGES = DIR_PATH / "results" / "TinyStories" / "merges.pkl"
#     DATA_PATH = DIR_PATH / "data" / "TinyStoriesV2-GPT4-valid.txt"
#     SAMPLE_NUM = 10

#     tokenizer = Tokenizer.from_files(INPUT_PATH_VOCAB, INPUT_PATH_MERGES)

#     # sampling for sanity check
#     content = []
#     with open(DATA_PATH, "rb") as f:
#         for i in range(10000):
#             line_content = f.readline().decode("utf-8")
#             content.append(line_content)
    
#     sampled_content = random.sample(content, 10000)
#     text = "".join(line_content for line_content in sampled_content)

#     start_time = time.time()
#     tokens = tokenizer.encode(text)
#     end_time = time.time()
    
#     print(f"length of the text {len(text)}")
#     print(f"length of the tokens {len(tokens)}")
#     print(f"compression ratio {len(text) / len(tokens)}")
#     print(f"throughput {len(text) / (end_time - start_time)}")

#     # process and save the tokenzied data
#     DATA_PATH_TRAIN = DIR_PATH / "data" / "TinyStoriesV2-GPT4-train.txt"
#     DATA_PATH_VALID = DIR_PATH / "data" / "TinyStoriesV2-GPT4-valid.txt"
#     OUTPUT_PATH_TRAIN = DIR_PATH / "data" / "TinyStoriesV2-GPT4-train-tokenized.npy"
#     OUTPUT_PATH_VALID = DIR_PATH / "data" / "TinyStoriesV2-GPT4-valid-tokenized.npy"

#     content = []
#     with open(DATA_PATH_VALID, "rb") as f:
#         for sentence in f:
#             tokenizer.encode_iterable(sentence)
#             content.extend(sentence)

#     content = np.array(content, dtype=np.uint16)
#     np.save(OUTPUT_PATH_VALID, content)

#     content = []
#     with open(DATA_PATH_TRAIN, "rb") as f:
#         for sentence in f:
#             tokenizer.encode_iterable(sentence)
#             content.extend(sentence)

#     content = np.array(content, dtype=np.uint16)

#     np.save(OUTPUT_PATH_TRAIN, content)


from tests.adapters import Tokenizer
import numpy as np
import pathlib
import os
from tqdm import tqdm

def tokenize_and_save_in_chunks(
    tokenizer: Tokenizer,
    input_path: pathlib.Path,
    output_dir: pathlib.Path,
    chunk_size: int = 2_000_000, # 2 million tokens per chunk file
):
    """
    Reads a large text file, tokenizes it, and saves the tokens in separate
    chunk files to avoid using too much memory.
    """
    # Ensure the output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    token_buffer = []
    chunk_idx = 0

    print(f"Tokenizing {input_path.name} into chunks of {chunk_size} tokens...")
    
    with open(input_path, "rb") as f:
        # Use tqdm to show file reading progress
        pbar = tqdm(f, unit="B", unit_scale=True, desc=f"Reading {input_path.name}")
        for line in pbar:
            # Decode the line from bytes to string and encode to tokens
            # This was the main bug: you must use the output of the tokenizer.
            tokens = tokenizer.encode(line.decode("utf-8", errors="ignore"))
            token_buffer.extend(tokens)
            
            # When the buffer is full, save a chunk to a file
            while len(token_buffer) >= chunk_size:
                # Slice off a chunk from the buffer
                chunk_to_save = token_buffer[:chunk_size]
                
                # Convert to a NumPy array
                np_chunk = np.array(chunk_to_save, dtype=np.uint16)
                
                # Define the output path and save the chunk
                output_path = output_dir / f"chunk_{chunk_idx}.npy"
                np.save(output_path, np_chunk)
                print(f"Saved {output_path}")
                
                # Remove the saved chunk from the buffer
                token_buffer = token_buffer[chunk_size:]
                chunk_idx += 1

    # Save any remaining tokens in the buffer as the last chunk
    if token_buffer:
        np_chunk = np.array(token_buffer, dtype=np.uint16)
        output_path = output_dir / f"chunk_{chunk_idx}.npy"
        np.save(output_path, np_chunk)
        print(f"Saved final chunk to {output_path}")

if __name__ == '__main__':
    DIR_PATH = pathlib.Path(__file__).resolve().parent.parent
    
    # --- Define Paths ---
    INPUT_PATH_VOCAB = DIR_PATH / "results" / "TinyStories" / "vocab.pkl"
    INPUT_PATH_MERGES = DIR_PATH / "results" / "TinyStories" / "merges.pkl"
    DATA_PATH_TRAIN = DIR_PATH / "data" / "TinyStoriesV2-GPT4-train.txt"
    DATA_PATH_VALID = DIR_PATH / "data" / "TinyStoriesV2-GPT4-valid.txt"
    
    # --- Define Output Directories for Chunks ---
    OUTPUT_DIR_TRAIN = DIR_PATH / "data" / "train_tokenized_chunks"
    OUTPUT_DIR_VALID = DIR_PATH / "data" / "valid_tokenized_chunks"

    # --- Load Tokenizer ---
    tokenizer = Tokenizer.from_files(INPUT_PATH_VOCAB, INPUT_PATH_MERGES)

    # --- Process and Save Files in Chunks ---
    # Process the validation file
    tokenize_and_save_in_chunks(
        tokenizer=tokenizer,
        input_path=DATA_PATH_VALID,
        output_dir=OUTPUT_DIR_VALID,
        chunk_size=20_000_000 # You can adjust this chunk size
    )

    # Process the training file
    tokenize_and_save_in_chunks(
        tokenizer=tokenizer,
        input_path=DATA_PATH_TRAIN,
        output_dir=OUTPUT_DIR_TRAIN,
        chunk_size=20_000_000 # Keeping chunk size consistent
    )

    print("\nTokenization complete. Chunks are saved in the respective directories.")