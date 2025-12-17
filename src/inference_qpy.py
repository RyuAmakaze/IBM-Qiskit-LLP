"""Inference script for circuits saved as QPY files.

This utility loads a trained quantum circuit serialized in QPY format and
evaluates it on the test split. The script mirrors the preprocessing used in
training so the feature scaling is consistent.

In addition to the amplitude-based inference used for accuracy evaluation, the
script can also run sampling on a local simulator to produce hardware-like
measurement counts and save them for inspection.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List

import numpy as np
from qiskit import qpy, transpile
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator
from sklearn.metrics import accuracy_score

from config import (
    C_DEPTH,
    NQUBIT,
    PCA_DIM,
    SIMULATOR_RAW_RESULT_PATH,
    SIMULATOR_SHOTS,
    TEST_DATA_PATH,
    TRAIN_DATA_PATH,
    WEIGHT_QPY_PATH,
)
from data_utils import load_pt_features
from qcl_classification import QclClassification
from qcl_utils import min_max_scaling


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


def _build_sampling_circuit(model: QclClassification, x_sample: np.ndarray):
    """Encode a single sample and attach measurements for simulator execution."""

    x_scaled = min_max_scaling(np.asarray([x_sample]))[0]
    input_gate = model.create_input_gate(x_scaled)

    circuit = input_gate.compose(model.output_gate)
    circuit.measure_all()
    return circuit


def _counts_to_probs(counts: Dict[str, int], num_class: int) -> np.ndarray:
    """Convert sampled bitstring counts into class probabilities.

    Each computational-basis bitstring is treated as the integer label for the
    corresponding class. The probability for class ``k`` is the fraction of
    shots whose bitstring decodes to ``k``.
    """

    shots = sum(counts.values()) if counts else 1
    quasi = {int(bitstr, 2): c / shots for bitstr, c in counts.items()}
    return np.array([quasi.get(cls, 0.0) for cls in range(num_class)])


def _state_to_probs(state: Statevector, num_class: int) -> np.ndarray:
    """Convert a ``Statevector`` to class probabilities.

    This mirrors the measurement-count postprocessing by using the probability
    of each computational-basis state (without sampling noise) as the class
    likelihood. It keeps the mapping where the measured bitstring encodes the
    class index.
    """

    full_probs = state.probabilities()
    probs = np.array([full_probs[cls] if cls < len(full_probs) else 0.0 for cls in range(num_class)])
    return probs / probs.sum() if probs.sum() > 0 else probs


def run_inference():
    """Load the trained circuit and report test accuracy using amplitudes."""

    _, x_test, _, y_test = load_pt_features(
        TRAIN_DATA_PATH, TEST_DATA_PATH, PCA_DIM
    )
    num_class = len(np.unique(y_test))

    circuit = load_trained_circuit(WEIGHT_QPY_PATH)

    model = QclClassification(NQUBIT, C_DEPTH, num_class)
    model.set_output_gate(circuit)

    preds = []
    for x in min_max_scaling(x_test):
        state = model._state_for_x(x, model.theta)
        probs = _state_to_probs(state, num_class)
        preds.append(probs)

    preds = np.array(preds)
    pred_labels = np.argmax(preds, axis=1)
    acc = accuracy_score(y_test, pred_labels)

    print(f"QPY inference accuracy: {acc:.3f}")
    return acc


def run_sampling_inference(sample_count: int = 5) -> List[int]:
    """Execute the trained circuit on AerSimulator and store raw counts."""

    _, x_test, _, y_test = load_pt_features(TRAIN_DATA_PATH, TEST_DATA_PATH, PCA_DIM)
    num_class = len(np.unique(y_test))

    circuit = load_trained_circuit(WEIGHT_QPY_PATH)
    model = QclClassification(NQUBIT, C_DEPTH, num_class)
    model.set_output_gate(circuit)

    simulator = AerSimulator()

    predictions: List[int] = []
    raw_results = []

    for idx in range(min(sample_count, len(x_test))):
        qc = _build_sampling_circuit(model, x_test[idx])
        compiled = transpile(qc, simulator)
        job = simulator.run(compiled, shots=SIMULATOR_SHOTS)
        res = job.result()
        counts = res.get_counts(0)

        probs = _counts_to_probs(counts, num_class)
        pred_label = int(np.argmax(probs))

        predictions.append(pred_label)
        raw_results.append(
            {
                "sample_index": idx,
                "counts": counts,
                "shots": sum(counts.values()) if counts else 0,
                "probabilities": probs.tolist(),
                "predicted_label": pred_label,
                "true_label": int(y_test[idx]),
            }
        )
        print(
            f"sample {idx}: predicted={pred_label}, true={int(y_test[idx])}, distribution={probs}"
        )

    os.makedirs(os.path.dirname(SIMULATOR_RAW_RESULT_PATH) or ".", exist_ok=True)
    with open(SIMULATOR_RAW_RESULT_PATH, "w", encoding="utf-8") as f:
        json.dump(raw_results, f, ensure_ascii=False, indent=2)
    print(
        f"Raw simulator data saved to {SIMULATOR_RAW_RESULT_PATH} (shots={SIMULATOR_SHOTS})."
    )

    return predictions


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run QPY inference.")
    parser.add_argument(
        "--mode",
        choices=["amplitude", "sampling"],
        default="amplitude",
        help="Inference mode: amplitude-based accuracy or sampling counts on AerSimulator.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=5,
        help="Number of test samples to evaluate when using sampling mode.",
    )
    return parser


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    print(f"Loading trained circuit from {WEIGHT_QPY_PATH}")

    if args.mode == "sampling":
        print(
            f"Starting simulator inference with {SIMULATOR_SHOTS} shots per circuit (samples={args.samples})."
        )
        run_sampling_inference(sample_count=args.samples)
    else:
        run_inference()


if __name__ == "__main__":
    main()
