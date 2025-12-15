"""Inference script for circuits saved as QPY files.

This utility loads a trained quantum circuit serialized in QPY format and
evaluates it on the test split. The script mirrors the preprocessing used in
training so the feature scaling is consistent.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score
from qiskit import qpy

from config import C_DEPTH, NQUBIT, PCA_DIM, TEST_DATA_PATH, TRAIN_DATA_PATH, WEIGHT_QPY_PATH
from data_utils import load_pt_features
from qcl_classification import QclClassification


def load_trained_circuit(qpy_path: str):
    """Load a single trained circuit from a QPY file."""

    with open(qpy_path, "rb") as f:
        circuits = qpy.load(f)

    if not circuits:
        raise ValueError(f"No circuits found in QPY file: {qpy_path}")
    if len(circuits) > 1:
        print(
            f"[warning] Multiple circuits found in {qpy_path}; using the first entry for inference."
        )

    return circuits[0]


def run_inference():
    """Load the trained circuit and report test accuracy."""

    x_train, x_test, y_train, y_test = load_pt_features(
        TRAIN_DATA_PATH, TEST_DATA_PATH, PCA_DIM
    )
    num_class = len(np.unique(y_train))

    circuit = load_trained_circuit(WEIGHT_QPY_PATH)

    model = QclClassification(NQUBIT, C_DEPTH, num_class)
    model.set_output_gate(circuit)

    preds = model.pred_amplitude(x_test)
    pred_labels = np.argmax(preds, axis=1)
    acc = accuracy_score(y_test, pred_labels)

    print(f"QPY inference accuracy: {acc:.3f}")
    return acc


if __name__ == "__main__":
    print(f"Loading trained circuit from {WEIGHT_QPY_PATH}")
    run_inference()
