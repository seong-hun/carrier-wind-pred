import os

DATASET_PATH = os.path.join("data", "video")
LOG_DIR = os.path.join("logs")
GENERATED_DIR = os.path.join("data", "generated")
MODELS_DIR = os.path.join("data", "models")
VGG_FACE = os.path.join("src", "vgg_face_dag.pth")

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
MAPSIZE = 1000
GRIDSIZE = 256
LANDMARK_RANGE = 20
VIDEONUM = 20
FRAMENUM = 100
# VIDEONUM = 2
# FRAMENUM = 5
CHANNEL = 4

# Training Hyperparameters
K = 8
BATCHSIZE = 3
EPOCHS = 1000
LEARNING_RATE_E_G = 5e-5
LEARNING_RATE_D = 2e-4
LOSS_VGG_FACE_WEIGHT = 2e-3
LOSS_VGG19_WEIGHT = 1e-2
LOSS_MASK_WEIGHT = 5e-1
LOSS_MCH_WEIGHT = 8e1
LOSS_FM_WEIGHT = 1e1
LOSS_LAN_WEIGHT = 1e-2
FEED_FORWARD = False

# Model Parameters
E_VECTOR_LENGTH = 512
