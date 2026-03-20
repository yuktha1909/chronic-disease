import os
import pickle

import pandas as pd

import config


def main() -> None:
    df = pd.read_csv(config.DATASET_PATH)
    print("DATASET_PATH:", config.DATASET_PATH)
    print("dataset_shape:", df.shape)

    model_path = str(config.MODEL_PATH)
    print("MODEL_PATH:", model_path)
    print("model_exists:", os.path.exists(model_path))

    # Optional deeper check: try to unpickle (can be slow, but validates wiring).
    # If you want a faster check, comment this block out.
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    print("model_type:", type(model))


if __name__ == "__main__":
    main()

