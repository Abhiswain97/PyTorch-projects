import torch

# Paths
TRAIN_PATH = "../data/Train"
TEST_PATH = "../data/Test"

# Hyperparameters
BATCH_SIZE = 128

# Miscellanous
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
