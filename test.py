from DeepCatra.learning.data_reader import *
from DeepCatra.results.model_test import get_split_dataset

test_dataset_path = "Features"
dataset = get_split_dataset(test_dataset_path, 13, 100)
