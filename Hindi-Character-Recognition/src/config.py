import torch

# Paths
TRAIN_PATH = "../data/Train"
TEST_PATH = "../data/Test"

# Hyperparameters
BATCH_SIZE = 32
EPOCHS = 10
LR = 1e-5

# Miscellanous
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
INTERVAL = 100
