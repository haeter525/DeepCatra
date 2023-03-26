import sys
from DeepCatra.learning.lstm_preprocess import encoding
from DeepCatra.model.model_to_predict import (
    get_split_dataset,
    test,
)

opcode_dict = encoding()


def main():
    test_dataset_path = sys.argv[1]
    test_dataset = get_split_dataset(test_dataset_path, 13, 100)
    test(test_dataset, "task")


if __name__ == "__main__":
    main()
