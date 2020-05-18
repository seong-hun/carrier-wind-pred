import os

DATASET_PATH = os.path.join("data", "video")
LOG_DIR = os.path.join("logs")
GENERATED_DIR = os.path.join("data", "generated")
MODELS_DIR = os.path.join("data", "models")

# GPU
GPU = {
    'Embedder': True,
    'Generator': False,
    'Discriminator': False,
    'LossEG': True,
    'LossD': True,
}

# Dataset hyperparameters
NANVALUE = 50
MAPSIZE = 500
GRIDSIZE = 32
VIDEONUM = 20
FRAMENUM = 250
CHANNEL = 5

# Training Hyperparameters
K = 8
BATCHSIZE = 3
EPOCHS = 1000
LEARNING_RATE_E_G = 1e-4
LEARNING_RATE_D = 5e-4
FEED_FORWARD = False
LOSS_CNT_WEIGHT = 1e-2
LOSS_MCH_WEIGHT = 1e0
LOSS_FM_WEIGHT = 1e1

# Model Parameters
E_VECTOR_LENGTH = 128
