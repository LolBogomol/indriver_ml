import os
import torch
import random
import numpy as np


DATA_DIR = "data"
OUTPUT_DIR = "outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Hyperparams
BACKBONE = "efficientnet_b4"
IMG_SIZE = 512
BATCH_SIZE = 16
EPOCHS = 20
LR = 1e-4
WEIGHT_DECAY = 1e-2
NUM_CLASSES = 3                # 0: not_damaged, 1: minor, 2: severe

# Mixed-crop parameters
N_RANDOM_CROPS = 6             # per image, during dataset sampling (mixing)
CROP_SCALES = [0.5, 0.7, 1.0]  # possible scales for random crops (relative to car bbox)
