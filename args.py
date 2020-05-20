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
NANVALUE = 0
MAPSIZE = 1000
GRIDSIZE = 64
LANDMARK_RANGE = 50
VIDEONUM = 50
FRAMENUM = 50
# VIDEONUM = 2
# FRAMENUM = 5
CHANNEL = 4

# Training Hyperparameters
K = 8
BATCHSIZE = 5
EPOCHS = 1000
LEARNING_RATE_E_G = 5e-5
LEARNING_RATE_D = 2e-4
LOSS_VGG19_WEIGHT = 1e-2
LOSS_VGG_FACE_WEIGHT = 2e-3
LOSS_IMAGE_WEIGHT = 5e-1
LOSS_MCH_WEIGHT = 1e1
LOSS_FM_WEIGHT = 1e1
LOSS_LAN_WEIGHT = 1e-2
FEED_FORWARD = False

# Model Parameters
E_VECTOR_LENGTH = 256
