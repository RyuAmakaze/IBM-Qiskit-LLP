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
C_DEPTH = 32
# Maximum number of iterations for parameter optimization
MAX_ITER = 100

# Number of samples in each LLP bag
BAG_SIZE = 10

# Paths to pre-extracted feature datasets
TEST_DATA_PATH = "trial_data/CIFAR10_DINOextract_test_5class_100instace.pt"

# Dimensionality after PCA compression
PCA_DIM = NQUBIT

# Gradient clipping threshold for training (``None`` to disable clipping)
GRAD_CLIP_NORM = None

# Path to the trained circuit serialized as a QPY file for inference
WEIGHT_QPY_PATH = "model/trained_circuit.qpy"
