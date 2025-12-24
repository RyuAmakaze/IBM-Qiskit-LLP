"""Shot-based inference on a local simulator.

This script runs the trained QPY circuit on ``AerSimulator`` with a specified
number of shots, collects raw measurement counts, converts them to class
probabilities, and saves the results to disk. It mirrors the bitstring-to-class
mapping used by hardware inference, keeping sampling separate from the
amplitude-based accuracy evaluation in ``inference_qpy.py``.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Tuple

import numpy as np
from qiskit import qpy, transpile
from qiskit_aer import AerSimulator

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
    """Encode a pre-scaled sample and attach measurements for simulator execution."""

    input_gate = model.create_input_gate(x_sample)

    circuit = input_gate.compose(model.output_gate)
    circuit.measure_all()
    return circuit


def _counts_to_probs(counts: Dict[str, int], num_class: int) -> np.ndarray:
    """Convert sampled bitstring counts into class probabilities.

    Bitstrings are decoded to integers and folded into the label range via
    ``value % num_class``. This prevents losing probability mass when the
    measured state index exceeds the number of target classes, which otherwise
    produced zero vectors and a constant prediction of class ``0``.
    """

    class_counts = np.zeros(num_class, dtype=float)
    for bitstr, count in counts.items():
        cls = int(bitstr, 2) % num_class
        class_counts[cls] += count

    shots = class_counts.sum()
    if shots == 0:
        return np.full(num_class, 1.0 / num_class)

    return class_counts / shots


def run_shot_inference(shots: int = SIMULATOR_SHOTS) -> Tuple[List[int], float]:
    """Execute the trained circuit on AerSimulator for every test sample.

    Returns
    -------
    Tuple[List[int], float]
        Predicted labels for each test sample and the overall accuracy.
    """

    _, x_test, _, y_test = load_pt_features(TRAIN_DATA_PATH, TEST_DATA_PATH, PCA_DIM)
    x_scaled = min_max_scaling(x_test)
    num_class = len(np.unique(y_test))

    circuit = load_trained_circuit(WEIGHT_QPY_PATH)
    model = QclClassification(NQUBIT, C_DEPTH, num_class)
    model.set_output_gate(circuit)

    simulator = AerSimulator()

    predictions: List[int] = []
    raw_results = []

    for idx, x_sample in enumerate(x_scaled):
        qc = _build_sampling_circuit(model, x_sample)
        compiled = transpile(qc, simulator)
        job = simulator.run(compiled, shots=shots)
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
    print(f"Raw simulator data saved to {SIMULATOR_RAW_RESULT_PATH} (shots={shots}).")

    accuracy = float(np.mean(np.array(predictions) == y_test))
    print(f"Simulator sampling accuracy: {accuracy:.3f} ({len(predictions)} samples)")

    return predictions, accuracy


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run shot-based inference on AerSimulator.")
    parser.add_argument(
        "--shots",
        type=int,
        default=SIMULATOR_SHOTS,
        help="Number of shots to execute per circuit.",
    )
    return parser


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    print(f"Loading trained circuit from {WEIGHT_QPY_PATH}")
    print(
        f"Starting shot-based simulator inference with {args.shots} shots per circuit for all test samples."
    )
    run_shot_inference(shots=args.shots)


if __name__ == "__main__":
    main()
