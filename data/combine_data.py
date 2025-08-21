import numpy as np
import pathlib
from tqdm import tqdm

def combine_npy_chunks(chunks_dir: str | pathlib.Path, output_path: str | pathlib.Path):
    """
    Finds all 'chunk_*.npy' files in a directory, concatenates them in
    sorted order, and saves them to a single .npy file.
    """
    chunks_dir = pathlib.Path(chunks_dir)
    output_path = pathlib.Path(output_path)

    # 1. Find and sort all chunk files to ensure they are in the correct order
    chunk_paths = sorted(chunks_dir.glob("chunk_*.npy"))
    
    if not chunk_paths:
        print(f"No chunk files found in {chunks_dir}")
        return

    print(f"Found {len(chunk_paths)} chunks. Combining them now...")

    # 2. Load each chunk into a list
    all_chunks = []
    for path in tqdm(chunk_paths, desc="Loading chunks"):
        chunk = np.load(path)
        all_chunks.append(chunk)

    # 3. Concatenate all chunks into a single NumPy array
    print("Concatenating all chunks...")
    combined_array = np.concatenate(all_chunks)

    # 4. Save the final combined array to the output path
    print(f"Saving combined array of shape {combined_array.shape} to {output_path}...")
    np.save(output_path, combined_array)
    
    print("Done!")

if __name__ == '__main__':
    DIR_PATH = pathlib.Path(__file__).resolve().parent.parent
    
    # --- CONFIGURE YOUR PATHS HERE ---
    
    # For the training data
    TRAIN_CHUNKS_DIR = DIR_PATH / "data" / "train_tokenized_chunks"
    TRAIN_OUTPUT_FILE = DIR_PATH / "data" / "train_tokenized.npy"
    combine_npy_chunks(TRAIN_CHUNKS_DIR, TRAIN_OUTPUT_FILE)

    # For the validation data
    VALID_CHUNKS_DIR = DIR_PATH / "data" / "valid_tokenized_chunks"
    VALID_OUTPUT_FILE = DIR_PATH / "data" / "valid_tokenized.npy"
    combine_npy_chunks(VALID_CHUNKS_DIR, VALID_OUTPUT_FILE)