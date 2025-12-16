"""Configuration constants for training.

This file consolidates the parameters used during training. Adjusting these
values lets you tune how the data are split, the scale of the quantum circuit,
and the number of optimization iterations.
"""

# Proportion of the dataset reserved for testing
TEST_SIZE = 0.3
# Random seed used for the train/test split
RANDOM_STATE = 0
# Random seed for generating the initial quantum circuit parameters
SEED = 1
# Number of qubits to use
NQUBIT = 4
# Depth of the output circuit
C_DEPTH = 14
# Maximum number of iterations for parameter optimization
MAX_ITER = 100

# Number of samples in each LLP bag
BAG_SIZE = 10

# Paths to pre-extracted feature datasets
TRAIN_DATA_PATH = "trial_data/CIFAR10_DINOextract_train_5class_100instace.pt"
TEST_DATA_PATH = "trial_data/CIFAR10_DINOextract_test_5class_100instace.pt"

# Dimensionality after PCA compression
PCA_DIM = NQUBIT

# Whether to use GPU acceleration with qulacs
USE_GPU = False

# Gradient clipping threshold for training (``None`` to disable clipping)
GRAD_CLIP_NORM = None

# Learning rate decay configuration
# The learning rate used in training is multiplied by ``LR_DECAY_RATE``
# every ``LR_DECAY_STEPS`` optimization steps.
START_LR = 0.3
LR_DECAY_STEPS = 10
LR_DECAY_RATE = 0.95


# Directory for storing trained model artifacts
MODEL_DIR = "model"

# File name for saving the trained circuit as JSON
TRAINED_CIRCUIT_JSON = f"{MODEL_DIR}/depth={C_DEPTH}_ite={MAX_ITER}_circuit.json"
# File name for saving the trained circuit diagram as PNG
TRAINED_CIRCUIT_PNG = f"{MODEL_DIR}/depth={C_DEPTH}_ite={MAX_ITER}_circuit.png"

# File name for saving the trained circuit as QPY
TRAINED_CIRCUIT_QPY = f"{MODEL_DIR}/depth={C_DEPTH}_ite={MAX_ITER}_circuit.qpy"

# File name for persisting the optimized parameter vector
TRAINED_PARAMETERS_NPY = f"{MODEL_DIR}/depth={C_DEPTH}_ite={MAX_ITER}_parameters.npy"

# Path to the trained circuit JSON used for inference
WEIGHT_JSON_PATH = TRAINED_CIRCUIT_JSON

# Path to the trained circuit serialized as a QPY file for inference
WEIGHT_QPY_PATH = "model/depth=16_ite=100_circuit.qpy"

# IBM Quantum runtime configuration
# Set ``IBM_API_KEY`` to the API key associated with your IBM Quantum account.
IBM_API_KEY = ""
IBM_INSTANCE_CRN = ""

# Runtime channel, typically ``ibm_quantum`` for real hardware or ``ibm_cloud`` for IBM Cloud.
IBM_CHANNEL = "ibm_quantum"
# Backend to use for hardware execution (e.g., ``ibm_oslo``); override as needed.
IBM_BACKEND = "ibm_oslo"
# Number of shots to collect when sampling circuits on hardware.
IBM_SHOTS = 4000
